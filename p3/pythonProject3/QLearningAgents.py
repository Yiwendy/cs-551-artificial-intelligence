import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import operator
from random import shuffle
import pickle


class DQNAgent:
    def __init__(self, observation_space, action_space, learning_rate=0.001, discount=0.95, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01, memory_size=10000, batch_size=64):
        self.action_space = action_space.n
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.discount = discount

        self.model = self.create_model(observation_space.shape[0], action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def create_model(self, input_size, output_size):
        model = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, output_size)
        )
        return model

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_space)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy())

    def update(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.discount * np.amax(self.model(next_state).detach().numpy()))
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay

    def save(self, filename="./dqn_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename="./dqn_model.pth"):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
class DictQLearningAgent(object):
    def __init__(self, action_space, learning_rate=0.1, discount=0.95,
                 exploration_rate=0.5, exploration_decay_rate=0.99):
        self._q_table = dict()
        self._n_actions = action_space.n
        self._learning_rate = learning_rate
        self._discount = discount
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay_rate
        self._updatesSum = 0.0
        self._maxChange = 0.0

    def reset(self):
        self._exploration_rate *= self._exploration_decay
        self._updatesSum = 0.0
        self._maxChange = 0.0

    def save(self, filename="./qtable.pkl"):
        with open(filename, 'wb') as f:  # Note 'wb' for write binary mode
            pickle.dump(self._q_table, f)

    def load(self, filename="./qtable.pkl"):
        with open(filename, 'rb') as f:  # Note 'rb' for read binary mode
            self._q_table = pickle.load(f)
        print("Loaded qtable size: ", len(self._q_table), " from file: ", filename)

    def act(self, observation):
        if np.random.random_sample() < self._exploration_rate:
            return np.random.randint(0, self._n_actions)
        else:
            action_vals = [(self._q_table.get((observation, action), 0), action)
                           for action in range(self._n_actions)]
            shuffle(action_vals)
            return max(action_vals, key=operator.itemgetter(0))[1]

    def update(self, observation, action, new_observation, reward, terminal=False):
        observation_action = (observation, action)
        if terminal:
            val = 0
        else:
            val = max(self._q_table.get((new_observation, a), 0) for a in
                      range(self._n_actions))
        old = self._q_table.get(observation_action, 0)
        self._q_table[observation_action] = \
            self._q_table.get(observation_action, 0) + self._learning_rate * \
            (reward + self._discount *
             val -
             self._q_table.get(observation_action, 0))
        delta = abs(self._q_table[observation_action] - old)
        self._updatesSum += delta
        if (delta > self._maxChange):
            self._maxChange = delta


class DiscretizingQLearningAgent(DictQLearningAgent):
    def __init__(self, bins, action_space,
                 learning_rate=0.1,
                 discount=0.95,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99):
        DictQLearningAgent.__init__(self, action_space,
                                    learning_rate=learning_rate,
                                    discount=discount,
                                    exploration_rate=exploration_rate,
                                    exploration_decay_rate=exploration_decay_rate)
        self._bins = bins

    def _discretize(self, observation):
        # res = sum(np.digitize(x=val, bins=self._bins[i]).item() * (10**i)
        # for i, val in enumerate(observation))
        res = 0
        for i, val in enumerate(observation):
            temp = np.digitize(x=val, bins=self._bins[i])
            temp = temp * (10 ** i)  # shift to encode state id
            res += temp
        return res

    def reset(self):
        DictQLearningAgent.reset(self)

    def act(self, observation):
        return DictQLearningAgent.act(self, self._discretize(observation))

    def update(self, observation, action, new_observation, reward):
        DictQLearningAgent.update(self, self._discretize(observation), action,
                                  self._discretize(new_observation), reward)


class ContinuousQLearningAgent(object):
    def __init__(self, observation_space_dim, action_space,
                 learning_rate=0.1,
                 discount=0.95,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99):
        self._w = np.zeros([observation_space_dim, action_space.n])
        self._target_w = np.copy(self._w)
        self._n_actions = action_space.n
        self._learning_rate = learning_rate
        self._discount = discount
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay_rate
        self._updates_count = 0

    def reset(self):
        self._exploration_rate *= self._exploration_decay

    def _compute_q(self, observation, weights):
        return np.matmul(observation, weights)

    def act(self, observation):
        if np.random.random_sample() < self._exploration_rate:
            return np.random.randint(0, self._n_actions)
        else:
            q_vals = self._compute_q(observation, self._w)
            return np.argmax(q_vals)

    def update(self, observation, action, new_observation, reward):
        val = np.max(self._compute_q(new_observation, self._target_w))

        updates = reward + self._discount * val - self._compute_q(observation,
                                                                  self._w)[action]
        self._w[:, action] += self._learning_rate * updates * observation
        self._updates_count += 1
        if self._updates_count % 10 == 0:
            self._target_w = np.copy(self._w)


# class ContinuousTensorflowQLearningAgent(object):
# def __init__(self, observation_space_dim, action_space, hidden_dim=4,
# learning_rate=0.1, discount=0.95, exploration_rate=0.5,
# exploration_decay_rate = 0.99):
# self._sess = tf.Session()
# self._discount = tf.constant(discount)
# self._state = tf.placeholder(dtype=tf.float32, shape=[1,
# observation_space_dim], name = 'observation')
# self._action = tf.placeholder(dtype=tf.int32, shape=1, name='action')
# self._next_state = tf.placeholder(dtype=tf.float32, shape=[1,
# observation_space_dim], name = 'next_observation')
# self._reward = tf.placeholder(dtype=tf.float32, shape=[1], name='reward')
#
# self._w1 = tf.get_variable('w1', shape=[observation_space_dim,
# hidden_dim],
#
# initializer = tf.contrib.layers.xavier_initializer())
# self._b1 = tf.get_variable('b1', shape=[hidden_dim])
#
# self._w2 = tf.get_variable('w2', shape=[hidden_dim, action_space.n],
#
# initializer = tf.contrib.layers.xavier_initializer())
# self._b2 = tf.get_variable('b2', shape=[action_space.n])
#
# self._next_value = tf.reduce_max(self._compute_q(self._next_state),
# reduction_indices = 1)
# self._q_func = self._compute_q(self._state)
# self._policy = tf.argmax(self._q_func, dimension=1)
#
# prev_q = tf.slice(self._q_func[0, :], self._action, [1])
# self._loss = tf.reduce_mean(tf.square(self._reward + self._discount *
# self._next_value - prev_q))
# self._update =
# tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss)
#
# self._sess.run([tf.initialize_all_variables()])
#
# self._n_actions = action_space.n
# self._exploration_rate = exploration_rate
# self._exploration_decay = exploration_decay_rate
#
# def _compute_q(self, state):
# return tf.matmul(tf.nn.relu(tf.matmul(state, self._w1) + self._b1),
# self._w2) + self._b2
#
# def reset(self):
# self._exploration_rate *= self._exploration_decay
#
# def act(self, observation):
# if np.random.random_sample() < self._exploration_rate:
# return np.random.randint(0, self._n_actions)
# else:
# res = self._sess.run([self._policy], feed_dict={self._state:
# np.array([observation])})
# return res[0][0]
#
# def update(self, observation, action, new_observation, reward):
# self._sess.run([self._update], feed_dict={
# self._state: np.array([observation]),
# self._action: np.array([action]),
# self._next_state: np.array([new_observation]),
# self._reward: np.array([reward])
# })
if __name__ == "__main__":
    pass
