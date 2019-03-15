import numpy as np
import gym
import random
from gym.envs.registration import register


register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.8196, # optimum = .8196, changing this seems have no influence
)

env = gym.make("FrozenLakeNotSlippery-v0")
action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))
print(qtable)

total_episodes = 20000 # max number of training episodes
alpha = 0.8 # learning rate
max_steps = 99  # max number of step per episode
gamma = 0.95 # discount factor

epsilon = 1.0 # exploration rate in epsilon-greedy
max_epsilon = 1.0 # exploration probability at the beginning
min_epsilon = 0.01 # after decaying the min exploration probability
decay_rate = 0.001 # decay rate of exploration probability

rewards = [] # list of rewards

# training with q-learning 
for episode in range(total_episodes):
    # ata every beginning of an episode reset the environment
    state = env.reset()
    # initialize the number of step
    step = 0 
    done = False
    total_rewards = 0

    for step in range(max_steps):
        # randomize a number
        tradeoff = random.uniform(0, 1)

        # if tradeoff > epsilon then do exploitation
        if tradeoff > epsilon:
            # qtable[state, :] represent all the actions we can take at this state 
            action = np.argmax(qtable[state, :])

        # else do exploration -> select a random action
        else:
            action = env.action_space.sample()
        
        # take action and save the outcome
        new_state, reward, done, info = env.step(action)

        # update q(s,a) = q(s,a)+alpha[r(s,a)+gamma*max q(s',a') - q(s,a)]
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])       
        total_rewards = total_rewards + reward
        state = new_state # update state

        # if dead then finish this episode
        if done == True:
            break
    episode += 1

    # when we've got some q, then we can decay the exploration probability
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print('score over time:' + str(sum(rewards)/total_episodes))
print(qtable)
print(epsilon)        


# print the action in every place
# LEFT = 0 DOWN = 1 RIGHT = 2 UP = 3
env.reset()
env.render()
print(np.argmax(qtable,axis=1).reshape(4,4))

#All the episoded is the same
env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            break
        state = new_state
env.close()