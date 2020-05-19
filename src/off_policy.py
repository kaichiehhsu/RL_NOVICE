from collections import namedtuple

def off_policy_train(env, agent, CONFIG, r_enhance):
    TrainingRecord = namedtuple('TrainingRecord', ['ep', 'avg_reward', 'reward'])
    running_reward = -1000
    training_records = []
    
    for ep in range(CONFIG.MAX_EPISODES):
        s = env.reset()
        ep_reward = 0.
        for step_num in range(CONFIG.MAX_EP_STEPS):
            print('\r {:d}'.format(step_num), end='')
            if CONFIG.RENDER:
                env.render()
            # action selection
            a, a_idx = agent.select_action(s)

            # interact with env
            s_, r, done, _ = env.step(a)
            ep_reward += r
            if done:
                s_ = None

            # Store the transition in memory
            agent.store_transition(s, a_idx, r_enhance(r), s_)
            s = s_

            # Perform one step of the optimization (on the target network)
            q = agent.update_Q_network()
            if done:
                break
        running_reward = running_reward * 0.9 + ep_reward * 0.1
        training_records.append(TrainingRecord(ep, running_reward, ep_reward))

        if ep % 10 == 0:
            print('\rEp[{:3.0f}]: Running Reward: {:.2f} \t Real Reward: {:.2f}'.format(ep, running_reward, ep_reward))
        if running_reward > -150:
            print("Solved! Running reward is now {:.2f}!".format(running_reward))
            env.close()
            break
    return training_records