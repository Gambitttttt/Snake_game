import torch       
import random
import numpy as np
import statistics
import matplotlib.pyplot as plt
from env_for_RL import Snake
from model import DQN, trainer
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.98

class Agent():
    
    def __init__(self):
        self.n_games = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = DQN()
        self.trainer = trainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, env):
        head = env.snake_body[0]
        corner = [env.w - env.blocksize, env.h - env.blocksize]

        new_head_up = [head[0], head[1] - env.blocksize]
        new_head_down = [head[0], head[1] + env.blocksize]
        new_head_right = [head[0] + env.blocksize, head[1]]
        new_head_left = [head[0] - env.blocksize, head[1]]
        
        # текущее направление

        dir_up = (env.dir == 'UP')
        dir_down = (env.dir == 'DOWN')
        dir_right = (env.dir == 'RIGHT')
        dir_left = (env.dir == 'LEFT')    

        # относительно яблока


        state = [
            # danger where 
            ((dir_up and env.check_collision(new_head_up)) or
            (dir_down and env.check_collision(new_head_down)) or
            (dir_left and env.check_collision(new_head_left)) or
            (dir_right and env.check_collision(new_head_right))),

            ((dir_up and env.check_collision(new_head_right)) or
            (dir_down and env.check_collision(new_head_left)) or
            (dir_left and env.check_collision(new_head_up)) or
            (dir_right and env.check_collision(new_head_down))),
            
            ((dir_up and env.check_collision(new_head_left)) or
            (dir_down and env.check_collision(new_head_right)) or
            (dir_left and env.check_collision(new_head_down)) or
            (dir_right and env.check_collision(new_head_up))),

            # cur dir
            (dir_up),
            (dir_down),
            (dir_right),
            (dir_left),
            # wheres da apple
            (head[1] > env.apple[1]),
            (head[1] < env.apple[1]),
            (head[0] > env.apple[0]),
            (head[0] < env.apple[0])
        ]

        return list(np.array(state, dtype = float))
    
    def get_action(self, state):
        action = [0, 0, 0]
        cur_state = torch.tensor(state, dtype=torch.float)
        prediction = self.model(cur_state)
        action_coord = torch.argmax(prediction).item()
        action[action_coord] = 1

        return action
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


def evaluate():
    global steps_till_end
    global steps_till_apple
    steps_till_end = []
    steps_till_apple = []
    agent = Agent()
    env = Snake()
    agent.model.load()

    while agent.n_games != 500:
        if env.score == 0:
            waiting = True

        state_old = agent.get_state(env)
        action = agent.get_action(state_old)
        reward, game_over, score = env.play_step(action)

        if env.score == 1 and waiting:
            waiting = False
            steps_till_apple.append(env.iteration)

        if game_over:
            steps_till_end.append(env.iteration)
            env.reset_env()
            print('Score', score)
            agent.n_games += 1

if __name__ == '__main__':
    evaluate()
    steps_till_end = np.array(steps_till_end)
    steps_till_apple = np.array(steps_till_apple)

    print(f'Mean of steps till the end: {np.mean(steps_till_end)}')
    print(f'Min of steps till the end: {min(steps_till_end)}')
    print(f'Max of steps till the end: {max(steps_till_end)}')
    print(f'Mode of steps till the end: {statistics.mode(steps_till_end)}')
    print(f'Median of steps till the end: {np.median(steps_till_end)}')
    print(f'Standard Deviation of steps till the end: {np.std(steps_till_end)}')
    print()


    print(f'Mean of steps till the first apple: {np.mean(steps_till_apple)}')
    print(f'Min of steps till the first apple: {min(steps_till_apple)}')
    print(f'Max of steps till the first apple: {max(steps_till_apple)}')
    print(f'Mode of steps till the first apple: {statistics.mode(steps_till_apple)}')
    print(f'Median of steps till the first apple: {np.median(steps_till_apple)}')
    print(f'Standard Deviation of steps till the first apple: {np.std(steps_till_apple)}')

    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    # plt.clf()

    plt.figure(figsize = (12, 7))
    plt.subplot(1, 2, 1)
    plt.hist(steps_till_apple, color = 'blue', edgecolor = 'black')
    plt.xlabel('Число шагов')
    plt.ylabel('Частота признака')
    plt.title('Распределение числа шагов до первого съеденного яблока')
    plt.subplot(1, 2, 2)
    plt.hist(steps_till_end, color = 'blue', edgecolor = 'black')
    plt.xlabel('Число шагов')
    plt.ylabel('Частота признака')
    plt.title('Распределение числа шагов до конца игры')

    plt.show()