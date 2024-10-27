import random
import numpy as np
import statistics
import matplotlib.pyplot as plt
from IPython import display
from env_no_RL import Snake



class Hamilton_Agent():
    
    def __init__(self):
        self.random = 1
        self.n_games = 0

    def define_net_connection(self, env):
        height = env.h
        width = env.w
        blocksize = env.blocksize

        self.width_steps = (width - blocksize) // blocksize
        self.height_steps = (height - blocksize) // blocksize

        if (self.height_steps % 2 == 0):
            connection = 'not full'
        else:
            connection = 'full'
        return connection 

    def create_net(self, connection):
        net = []
        if connection == 'full':
            for i in range(self.height_steps + 1):
                if i == 0:
                    row = [k for k in range(self.width_steps + 1)]
                    idx = row[-1] + 1
                else:
                    if i % 2 != 0:
                        row = [k for k in range(idx, idx + self.width_steps)]
                        idx = row[-1] + 1
                        row.append((self.width_steps + 1) * (self.height_steps + 1) - i)
                        row = row[::-1]
                    else:
                        row = [k for k in range(idx, idx + self.width_steps)]
                        idx = row[-1] + 1
                        row.insert(0, (self.width_steps + 1) * (self.height_steps + 1) - i)
                net.append(row)

            net = np.array(net)
            return net
        
        elif connection == 'not full' and self.width_steps % 2 == 0:
            for i in range(self.height_steps + 1):
                if i == 0:
                    row = [k for k in range(self.width_steps + 1)]
                    idx = row[-1] + 1
                elif i <= self.height_steps - 2:
                    if i % 2 != 0:
                        row = [k for k in range(idx, idx + self.width_steps)]
                        idx = row[-1] + 1
                        row.append((self.width_steps + 1) * (self.height_steps + 1) - i - 1)
                        row = row[::-1]
                    else:
                        row = [k for k in range(idx, idx + self.width_steps)]
                        idx = row[-1] + 1
                        row.insert(0, (self.width_steps + 1) * (self.height_steps + 1) - i - 1)
                elif i == self.height_steps - 1:
                    row = [idx + 3, idx]
                    t = 2
                    while t != self.width_steps:
                        row.insert(0, row[0] + 1)
                        row.insert(0, row[0] + 3)
                        t += 2
                    row.insert(0, row[0] + 1)
                elif i == self.height_steps:
                    row = [idx + 2, idx + 1]
                    t = 2
                    while t != self.width_steps:
                        row.insert(0, row[0] + 3)
                        row.insert(0, row[0] + 1)
                        t += 2
                    row.insert(0, row[0] + 1)
                net.append(row)
            
            net = np.array(net)
            return net

        elif connection == 'not full' and self.width_steps % 2 == 1:
            for i in range(self.height_steps + 1):
                if i == 0:
                    row = [k for k in range(self.width_steps + 1)]
                    idx = row[-1] + 1
                elif i <= self.height_steps - 2:
                    if i % 2 != 0:
                        row = [k for k in range(idx, idx + self.width_steps)]
                        idx = row[-1] + 1
                        row.append((self.width_steps + 1) * (self.height_steps + 1) - i)
                        row = row[::-1]
                    else:
                        row = [k for k in range(idx, idx + self.width_steps)]
                        idx = row[-1] + 1
                        row.insert(0, (self.width_steps + 1) * (self.height_steps + 1) - i)
                elif i == self.height_steps - 1:
                    row = [idx + 3, idx]
                    t = 2
                    while t < self.width_steps:
                        row.insert(0, row[0] + 1)
                        row.insert(0, row[0] + 3)
                        t += 2
                    row = row[1:]
                    row.insert(0, row[0] + 3)
                elif i == self.height_steps:
                    row = [idx + 2, idx + 1]
                    t = 2
                    while t < self.width_steps:
                        row.insert(0, row[0] + 3)
                        row.insert(0, row[0] + 1)
                        t += 2
                    row = row[1:]
                    row.insert(0, row[0] + 1)
                net.append(row)
            
            net = np.array(net)
            return net    

    def get_options_at_state(self, env):
        dir_up = (env.dir == 'UP')
        dir_down = (env.dir == 'DOWN')
        dir_right = (env.dir == 'RIGHT')
        dir_left = (env.dir == 'LEFT')

        head = env.snake_body[0]

        if dir_up:
            options = [[head[0], head[1] - env.blocksize], 
                       [head[0] + env.blocksize, head[1]], 
                       [head[0] - env.blocksize, head[1]]]
        elif dir_down:
            options = [[head[0], head[1] + env.blocksize], 
                       [head[0] - env.blocksize, head[1]], 
                       [head[0] + env.blocksize, head[1]]]
        elif dir_right:
            options = [[head[0] + env.blocksize, head[1]], 
                       [head[0], head[1] + env.blocksize], 
                       [head[0], head[1] - env.blocksize]]
        elif dir_left:
            options = [[head[0] - env.blocksize, head[1]], 
                       [head[0], head[1] - env.blocksize], 
                       [head[0], head[1] + env.blocksize]]
        return options

    def random_shortcut(self, env, connection):
        rand_threshold = 0.99
        rand_num = random.random()
        if (env.dir == 'RIGHT' or env.dir == 'LEFT') and env.snake_body[0][1] < (env.h - 2 * env.blocksize):
            if rand_num < rand_threshold:
                if env.dir == 'RIGHT':
                    return [0, 1, 0], 'LEFT'
                elif env.dir == 'LEFT':
                    return [0, 0, 1], 'RIGHT'
            else:
                return None, env.dir
        else:
            return None, env.dir

    def bad_shortcut(self, env, net):
        apple_value = net[env.apple[1] // env.blocksize][env.apple[0] // env.blocksize]
        head_shortcut = [env.snake_body[0][0], env.snake_body[0][1] + env.blocksize]
        shortcut_value = net[(head_shortcut[1] % env.h) // env.blocksize][(head_shortcut[0] % env.w) // env.blocksize]
        if (shortcut_value > apple_value) or env.check_collision(head_shortcut):
            return True
        else:
            return False

    def get_net_action(self, env, options, net):
        head = env.snake_body[0]
        temp_head = [(head[0] // env.blocksize), 
                    (head[1] // env.blocksize)]
        max_value = net.max()
        cur_value = net[temp_head[1]][temp_head[0]]
        if net[temp_head[1]][temp_head[0]] != max_value:
            value_change = []
            for elem in options:
                value_change.append(net[(elem[1] % env.h) // env.blocksize][(elem[0] % env.w) // env.blocksize] - cur_value)
            if value_change[0] == 1 and value_change[1] == 1:
                if self.random % 2 == 0:
                    action = [1, 0, 0]
                else:
                    action = [0, 1, 0]
            elif value_change[0] == 1 and value_change[1] != 1:
                action = [1, 0, 0]
            elif value_change[0] != 1 and value_change[1] == 1:
                action = [0, 1, 0]
            elif value_change[2] == 1:
                action = [0, 0, 1]
        else:
            action = [1, 0, 0]
        return action

def showcase():
    global steps_till_end
    global steps_till_apple
    steps_till_end = []
    steps_till_apple = []
    agent = Hamilton_Agent()
    env = Snake()
    connection = agent.define_net_connection(env)
    net = agent.create_net(connection)

    while agent.n_games != 500:
        if env.score == 0:
            waiting = True

        if env.snake_body[0] == [0, 0]:
            agent.random += 1

        cur_options = agent.get_options_at_state(env)

        if len(env.snake_body) < (env.h // env.blocksize) * 2 - 2:
            action, new_dir = agent.random_shortcut(env, connection)

            if (action is None) or agent.bad_shortcut(env, net):
                change_dir = False
                action = agent.get_net_action(env, cur_options, net)
            else:
                change_dir = True
            
        # perform move and get new state
            reward, done, score = env.play_step(action)
    
            if change_dir:
                env.dir = new_dir

        else:
            action = agent.get_net_action(env, cur_options, net)
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