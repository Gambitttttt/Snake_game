import sys
import pygame            
import random
import numpy as np

pygame.init()
font = pygame.font.SysFont('couriernew', 30)

WIDTH = 600
HEIGHT = 600
FPS = 5000

BLOCKSIZE = 40

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Snake():
    def __init__(self, w = WIDTH, h = HEIGHT):
        self.w = WIDTH
        self.h = HEIGHT
        self.blocksize = BLOCKSIZE
        self.fps = FPS
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        # Начальное положение змейки - левый верхний угол
        self.dir = 'RIGHT'
        self.snake_body = [[2 * self.blocksize, 0], 
                           [self.blocksize, 0], 
                           [0, 0]]
        self.tail = self.snake_body[-1]

        #Создание множества координат, на которых могут появиться яблоки
        self.map = []
        for i in range(0, self.w, self.blocksize):
            for j in range(0, self.h, self.blocksize):
                self.map.append([i, j])

        self.iteration = 0
        self.score = 0
        self._apple_spawn()
        self.reset_env()
    # Ф-ция спавна яблок вне змейки
    def _apple_spawn(self):
        temp_map = self.map.copy()
        for elem in self.snake_body:
            temp_map.remove(elem)
        x_apple, y_apple = random.choice(temp_map)
        self.apple = [x_apple, y_apple]
    # Ф-ция, которая получает действие и выдает либо новые координаты головы, либо смещает всю змейку согласно action
    def move_in_cur_dir(self, action, perform_move):

        dict_right = {'RIGHT': 'DOWN', 'LEFT': 'UP', 'UP': 'RIGHT', 'DOWN': 'LEFT'}
        dict_straight = {'RIGHT': 'RIGHT', 'LEFT': 'LEFT', 'UP': 'UP', 'DOWN': 'DOWN'}
        dict_left = {'RIGHT': 'UP', 'LEFT': 'DOWN', 'UP': 'LEFT', 'DOWN': 'RIGHT'}

        if np.array_equal(action, [1, 0, 0]):
            self.dir = dict_straight[self.dir]
        elif np.array_equal(action, [0, 1, 0]):
            self.dir = dict_right[self.dir]
        elif np.array_equal(action, [0, 0, 1]):
            self.dir = dict_left[self.dir]

        new_head = [self.snake_body[0][0], self.snake_body[0][1]] 
        if self.dir == 'LEFT':
            new_head[0] -= self.blocksize
        elif self.dir == 'RIGHT':
            new_head[0] += self.blocksize
        elif self.dir == 'UP':
            new_head[1] -= self.blocksize
        elif self.dir == 'DOWN':
            new_head[1] += self.blocksize

        if perform_move == 'make a move':
            self.tail = self.snake_body[-1]
            self.snake_body = [self.snake_body[-1]] + self.snake_body[:-1]
            self.snake_body[0] = new_head
        elif perform_move == 'receive coord':
            return new_head
    # Ф-ция, проверяющая наличие столкновения с собой или стеной. Также проверяет, пора ли закончить игру (если змейка закрыла собой все поле)
    # В случае передачи координат головы, проверяет, будет ли столкновение в этой точке
    def check_collision(self, head = None):
        if head is None:
            head = self.snake_body[0]
        if ((not (0 <= head[0] < self.w)) or (not (0 <= head[1] < self.h))) or (head in self.snake_body[1:]):
            return True
        elif self.score == (self.w // self.blocksize) * (self.h // self.blocksize) - 3:
            return True
        else:
            return False
    # Роль этой ф-ции: завершать игру, если слишком долго действия агента не приводили к съедению аблока 
    def check_game_length(self):
        if self.iteration > len(self.snake_body) * 100:
            return True
        else:
            return False
    # Ф-ция съедания яблока и спавна нового (если можно заспавнить еще одно)
    def _eat(self):
        if self.snake_body[0] == self.apple: 
            self.score += 1
            reward = 10
            if self.score != ((self.w // self.blocksize) - 1) * ((self.h // self.blocksize) - 1) - 3:
                self.snake_body.append(self.tail)
                self._apple_spawn()
    # Ф-ция, меняющая направление движения змейки
    def _change_dir(self, key):
        if (self.key == pygame.K_LEFT) and (self.dir == 'UP' or self.dir == 'DOWN'):
            self.dir = 'LEFT'
        elif self.key == pygame.K_RIGHT and (self.dir == 'UP' or self.dir == 'DOWN'):
            self.dir = 'RIGHT'
        elif self.key == pygame.K_UP and (self.dir == 'LEFT' or self.dir == 'RIGHT'):
            self.dir = 'UP'
        elif self.key == pygame.K_DOWN and (self.dir == 'LEFT' or self.dir == 'RIGHT'):
            self.dir = 'DOWN'
    # Ф-ция отрисовки среды        
    def _update_ui(self):
        self.screen.fill(BLACK)
        self.screen.blit(font.render(f'Score: {self.score}', True, WHITE), (0, 0))

        pygame.draw.rect(self.screen, GREEN, pygame.Rect(self.apple[0], self.apple[1], BLOCKSIZE, BLOCKSIZE), 0)

        for i in range(len(self.snake_body)):
            if i == 0:
                snake_part = pygame.Rect(self.snake_body[i][0], self.snake_body[i][1], BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(self.screen, RED, snake_part, 0)
            else:
                snake_part = pygame.Rect(self.snake_body[i][0], self.snake_body[i][1], BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(self.screen, RED, snake_part, 10)

        pygame.display.flip()
    # Ф-ция полного обновления среды
    def reset_env(self):
        self.dir = 'RIGHT'
        self.snake_body = [[2 * self.blocksize, 0], 
                           [self.blocksize, 0], 
                           [0, 0]]
        self.tail = self.snake_body[-1]

        self.score = 0
        self._apple_spawn()
        self.iteration = 0

    # Ф-ция, воспроизводящая шаг игры: получение действия -> результат
    def play_step(self, action):
        self.iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()



        self.move_in_cur_dir(action, 'make a move')

        reward = 0
        
        self._eat()

        game_over = False
        if self.check_collision() or self.check_game_length():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        self._update_ui()
        self.clock.tick(self.fps)

        return reward, game_over, self.score
    