import random
import numpy as np
import statistics
import matplotlib.pyplot as plt
from env_no_RL import Snake

class Random_Agent():
    
    def __init__(self):
        self.n_games = 0

    def get_action(self):
        action = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return action

def showcase():
    
    global steps_till_end
    global steps_till_apple
    steps_till_end = []
    steps_till_apple = []
    agent = Random_Agent()
    env = Snake()

    while agent.n_games != 5000:
        if env.score == 0:
            waiting = True
        
        action = agent.get_action()
        reward, done, score = env.play_step(action)

        if env.score == 1 and waiting:
            waiting = False
            steps_till_apple.append(env.iteration)

        if done:
            steps_till_end.append(env.iteration)
            env.reset_env()
            print('Score', score)
            agent.n_games += 1

    
if __name__ == '__main__':
    showcase()
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