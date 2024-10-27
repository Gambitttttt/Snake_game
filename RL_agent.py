import torch       
import random
import numpy as np
from env_for_RL import Snake
from model import DQN, trainer
from collections import deque
from visualizer import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
EPSILON = 0.5
GAMMA = 0.9
DECAY = 1000

class Agent():
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN()
        self.trainer = trainer(self.model, lr=LR, gamma=self.gamma)
        self.decay = DECAY

    # Получить от среды информацию для модели
    def get_state(self, env):
        head = env.snake_body[0]

        new_head_up = [head[0], head[1] - env.blocksize]
        new_head_down = [head[0], head[1] + env.blocksize]
        new_head_right = [head[0] + env.blocksize, head[1]]
        new_head_left = [head[0] - env.blocksize, head[1]]
        
        # текущее направление
        dir_up = (env.dir == 'UP')
        dir_down = (env.dir == 'DOWN')
        dir_right = (env.dir == 'RIGHT')
        dir_left = (env.dir == 'LEFT')    

        state = [
            # опасность прямо
            ((dir_up and env.check_collision(new_head_up)) or
            (dir_down and env.check_collision(new_head_down)) or
            (dir_left and env.check_collision(new_head_left)) or
            (dir_right and env.check_collision(new_head_right))),
            # опасность справа
            ((dir_up and env.check_collision(new_head_right)) or
            (dir_down and env.check_collision(new_head_left)) or
            (dir_left and env.check_collision(new_head_up)) or
            (dir_right and env.check_collision(new_head_down))),
            # опасность слева
            ((dir_up and env.check_collision(new_head_left)) or
            (dir_down and env.check_collision(new_head_right)) or
            (dir_left and env.check_collision(new_head_down)) or
            (dir_right and env.check_collision(new_head_up))),

            # направление в данный момент
            (dir_up),
            (dir_down),
            (dir_right),
            (dir_left),
            # где яблоко относительно головы
            (head[1] > env.apple[1]),
            (head[1] < env.apple[1]),
            (head[0] > env.apple[0]),
            (head[0] < env.apple[0])
        ]

        return list(np.array(state, dtype = float))
    # получаем действие, выдаваемое моделью, epsilon-greedy policy
    def get_action(self, state):
        random_num = np.random.uniform(0, 1)
        action = [0, 0, 0]
        if random_num < self.epsilon - self.n_games / self.decay:
            random_coord = random.randint(0, 2)
            action[random_coord] = 1 
        else:
            cur_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(cur_state)
            action_coord = torch.argmax(prediction).item()
            action[action_coord] = 1

        return action
    # запоминаем 
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    top_score = 0
    agent = Agent()
    env = Snake()
    while True:       
        state_old = agent.get_state(env)

        action = agent.get_action(state_old)

        reward, done, score = env.play_step(action)
        state_new = agent.get_state(env)

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            env.reset_env()
            agent.n_games += 1
            agent.train_long_memory()

            if score > top_score:
                top_score = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Top score:', top_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()
