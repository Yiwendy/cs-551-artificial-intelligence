import gymnasium as gym
import sys
import QLearningAgents

TRAIN_EPISODES = 50000
TEST_EPISODES = 5000

if __name__ == "__main__":
    env = gym.make('Blackjack-v1', natural=True, sab=False)

    # Decide whether to use a simple QLearning agent or a Deep Q-Learning agent
    if '--use-dqn' in sys.argv:
        agent = QLearningAgents.DQNAgent(env.observation_space, env.action_space)
    else:
        agent = QLearningAgents.DictQLearningAgent(
            env.action_space,
            learning_rate=0.05,
            discount=0.95,
            exploration_rate=0.5,
            exploration_decay_rate=0.99)

    mode = 'train'  # default mode
    if '-test' in sys.argv:
        mode = 'test'
        agent.load()  # Assuming your agent has a load method

    if mode == 'train':
        total_reward = 0
        for i_episode in range(TRAIN_EPISODES):
            observation, info = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(observation)
                new_observation, reward, terminated, truncated, info = env.step(action)
                agent.update(observation, action, new_observation, reward, terminal=terminated)
                done = terminated or truncated
                observation = new_observation
                episode_reward += reward
            total_reward += episode_reward
            if (i_episode % 1000) == 0:
                print("Episode: ", i_episode, " Average Reward: ", total_reward / (i_episode + 1))

        print("Training complete. Total Average Reward: ", total_reward / TRAIN_EPISODES)
        agent.save()  # Assuming your agent has a save method

    if mode == 'test':
        winnings = 0
        wins = 0
        for i in range(TEST_EPISODES):
            observation, info = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(observation)
                new_observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                observation = new_observation
                done = terminated or truncated
            print("Game Reward: ", episode_reward)
            winnings += episode_reward
            if episode_reward > 0:  # Count wins
                wins += 1

        average_winnings = winnings / TEST_EPISODES
        win_rate = wins / TEST_EPISODES * 100  # Calculate win rate
        print(f"Testing complete. Average Winnings: {average_winnings}")
        print(f"Win Rate: {win_rate}%")
