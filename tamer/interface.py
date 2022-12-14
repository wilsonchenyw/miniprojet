import os
import pygame
from tamer.gest import get_gest


class Interface:
    """ Pygame interface for training TAMER """

    def __init__(self, action_map):
        self.action_map = action_map
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 32)

        # set position of pygame window (so it doesn't overlap with gym)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
        os.environ["SDL_VIDEO_CENTERED"] = "0"

        self.screen = pygame.display.set_mode((200, 100))
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)

    def get_scalar_feedback(self):
        """
        Get human input. 'W' key for positive, 'A' key for negative.
        Returns: scalar reward (1 for positive, -1 for negative)
        """
        reward = 0
        area = None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    area = self.screen.fill((0, 255, 0))
                    reward = 1
                    break
                elif event.key == pygame.K_a:
                    area = self.screen.fill((255, 0, 0))
                    reward = -1
                    break
        pygame.display.update(area)
        return reward

    def get_gest_feedback(self,cap,mpHands,hands,mpDraw):
        """
        Get human input. l1>l2 for positive, l1<l2 for negative.
        Returns: scalar reward (1 for positive, -1 for negative)
        """
        reward = 0
        area = None
        l1,l2 = get_gest(cap,mpHands,hands,mpDraw)
        if l1>l2:
            area = self.screen.fill((0, 255, 0))
            reward = 1
        else:
            area = self.screen.fill((255, 0, 0))
            reward = -1
        pygame.display.update(area)
        return reward

    def get_parole_feedback(self,r,mic):
        """
        Get human input. 'yes' key for positive, 'no' key for negative.
        Returns: scalar reward (1 for positive, -1 for negative)
        """
        reward = 0
        area = None
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        test = r.recognize_google(audio)
        print(test)
        if 'yes' in test:
            area = self.screen.fill((0, 255, 0))
            reward = 1
        elif 'no' in test:
            area = self.screen.fill((255, 0, 0))
            reward = -1
        pygame.display.update(area)
        return reward

    def show_action(self, action):
        """
        Show agent's action on pygame screen
        Args:
            action: numerical action (for MountainCar environment only currently)
        """
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)
        text = self.font.render(self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        area = self.screen.blit(text, text_rect)
        pygame.display.update(area)
