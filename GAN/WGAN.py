import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from dataset import MNIST_DS, MyDataset

# model architecture: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py
class generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False), # get the entries
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, int(np.prod(self.img_shape))),
            nn.Tanh() # pixels in [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

class critic(nn.Module):
    def __init__(self, img_shape):
        super(critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
        
        self._initialize_weights()

    def forward(self, img):
        x = img.view(img.shape[0], -1)
        validity = self.model(x)
        return validity

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

class WGAN():
    def __init__(self, latent_dim, n_ch, n_w, n_h, device, lr, clip):
        self.latent_dim = latent_dim
        self.n_ch = n_ch
        self.n_w = n_w
        self.n_h = n_h

        self.device = device
        self.lr = lr
        self.clip = clip

        self.build_network()

    def build_network(self):
        self.critic = critic((self.n_ch, self.n_w, self.n_h))
        self.generator = generator(self.latent_dim, (self.n_ch, self.n_w, self.n_h))
        if self.device == torch.device('cuda'):
            self.critic.cuda()
            self.generator.cuda()
        self.optimizer_G = optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.optimizer_C = optim.RMSprop(self.critic.parameters(),    lr=self.lr)

    def train(self, train_data, ep_num=10, n_critic=5, batch_size=128, sample_period=5):
        self.critic.train()
        self.generator.train()
        label_real = np.ones((train_data.shape[0],))
        label_fake = -1 * np.ones((batch_size,))
        data_progress = []
        loss_C_progress = []
        loss_G_progress = []

        #== Data Loader ==
        train_dataset  = MyDataset(train_data, label_real)
        train_loader  = DataLoader( dataset = train_dataset,
                                    batch_size = batch_size,
                                    shuffle = True,
                                    pin_memory=torch.cuda.is_available())
        #== Train ==
        # min_G max_C E_x[C(x)] - E_z[C(G(z))]
        # Iterative Two-Step Optimization
        # min_C - E_x[C(x)] + E_z[C(G(z))]
        # min_G - E_z[C(G(z))]
        iter_num = int(np.ceil(len(train_loader)/batch_size))
        for ep in range(1, ep_num+1):
            for i, (data_real, _) in enumerate(train_loader):
                #===================
                #== Update Critic ==
                #===================
                if self.device == torch.device('cuda'):
                    data_real = data_real.cuda()

                z = torch.randn((batch_size, self.latent_dim), dtype=torch.float, device=self.device)
                with torch.no_grad():
                    data_fake = self.generator(z).detach()

                loss_C = -torch.mean(self.critic(data_real)) + torch.mean(self.critic(data_fake))

                self.optimizer_C.zero_grad()
                loss_C.backward()
                self.optimizer_C.step()

                for w in self.critic.parameters(): # to guarantee Lipschitz assumption
                    w.data.clamp_(-self.clip, self.clip)

                if self.device == torch.device('cuda'):
                    loss_C_progress.append(loss_C.data.cpu().numpy())
                else:
                    loss_C_progress.append(loss_C.data.numpy())
                #======================
                #== Update Generator ==
                #======================
                if i % n_critic == 0:
                    z = torch.randn((batch_size, self.latent_dim), dtype=torch.float, device=self.device)
                    data_fake = self.generator(z)

                    loss_G = -torch.mean(self.critic(data_fake))

                    self.optimizer_G.zero_grad()
                    loss_G.backward()
                    self.optimizer_G.step()
                    if self.device == torch.device('cuda'):
                        loss_G_progress.append(loss_G.data.cpu().numpy())
                    else:
                        loss_G_progress.append(loss_G.data.numpy())

            if ep % sample_period == 0:
                print("[Ep {:d}/{:d}] [loss_critic: {:.2e}] [loss_generator: {:.2e}]".format( \
                         ep, ep_num, loss_C.item(), loss_G.item()))
                if self.device == torch.device('cuda'):
                    data_fake = data_fake.cpu()
                data_progress.append(data_fake.data.numpy())
        return data_progress, loss_C_progress, loss_G_progress
