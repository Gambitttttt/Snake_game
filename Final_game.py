import sys
import pygame            
import random

pygame.init()
font = pygame.font.SysFont('couriernew', 30)

WIDTH = 800
HEIGHT = 800
FPS = 10

BLOCKSIZE = 40

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Snake():
    def __init__(self, w = WIDTH, h = HEIGHT):
        self.w = w
        self.h = h
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        self.dir = 'RIGHT'
        self.snake_body = [[2 * BLOCKSIZE, 0], [BLOCKSIZE, 0], [0, 0]]
        self.tail = self.snake_body[-1]

        self.score = 0
        self._apple_spawn()

    def _apple_spawn(self):
        x_apple = (random.randint(0, self.w) // BLOCKSIZE) * BLOCKSIZE
        y_apple = (random.randint(0, self.h) // BLOCKSIZE) * BLOCKSIZE
        self.apple = [x_apple, y_apple]
        if self.apple in self.snake_body:
            self._apple_spawn()

    def _move_in_cur_dir(self, dir):
        new_head = [self.snake_body[0][0], self.snake_body[0][1]]
        if self.dir == 'LEFT':
            new_head[0] -= BLOCKSIZE
        elif self.dir == 'RIGHT':
            new_head[0] += BLOCKSIZE
        elif self.dir == 'UP':
            new_head[1] -= BLOCKSIZE
        elif self.dir == 'DOWN':
            new_head[1] += BLOCKSIZE
        self.tail = self.snake_body[-1]
        self.snake_body = [self.snake_body[-1]] + self.snake_body[:-1]
        self.snake_body[0] = new_head

    def check_collision(self):
        if (not (0 <= self.snake_body[0][0] < self.w)) or (not (0 <= self.snake_body[0][1] < self.h)):
            return True
        if self.snake_body[0] in self.snake_body[1:]:
            return True

    def _eat(self):
        if self.snake_body[0][0] == self.apple[0] and self.snake_body[0][1] == self.apple[1]: 
            self.score += 1
            self.snake_body.append(self.tail)
            self._apple_spawn()

    def _change_dir(self, key):
        if (self.key == pygame.K_LEFT) and (self.dir == 'UP' or self.dir == 'DOWN'):
            self.dir = 'LEFT'
        elif self.key == pygame.K_RIGHT and (self.dir == 'UP' or self.dir == 'DOWN'):
            self.dir = 'RIGHT'
        elif self.key == pygame.K_UP and (self.dir == 'LEFT' or self.dir == 'RIGHT'):
            self.dir = 'UP'
        elif self.key == pygame.K_DOWN and (self.dir == 'LEFT' or self.dir == 'RIGHT'):
            self.dir = 'DOWN'

    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or self.check_collision():
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                self.key = event.key
                self._change_dir(self.key)

        self._move_in_cur_dir(self.dir)

        game_over = False
        if self.check_collision():
            game_over = True
            return game_over, self.score

        self._eat()

        self._update_ui()
        self.clock.tick(FPS)

        return game_over, self.score
    

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
    
if __name__ == '__main__':
    game = Snake()

    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()