# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:04:24 2025

@author: n.zhilenkova
"""

import pygame
import numpy as np
import addintion_func_dqn as af

import agent_q_l_exp as agent
import statistics
import random


WIDTH = 480
HIGTH = 640
BIRD_X = 185
CHANGE_GROUND = 560
COOR_BACKG = (0, 0)
ANGLE_TUBE = 180
DELTA_COL_Y = 506
DELTA_COL_X = 275
TUBE_WIDTH = 78
TUBE_HIGH_BEETW = (230, 465)
BIRD_SIZE = (55, 40)
MOVING_SCREEN = 480
SPEED_SCREEN = 2.5
GRAVITY = 2
ROTATION_SPEED = 1
JUMP = 90
SPEED_UP = 0.1
SPEED_BIRD = 2.05
DEVICE = "cuda:0"
BATCH_SIZE = 32
DELTA = 5
GAP_COL = 150

clock = pygame.time.Clock()
pygame.init()
pygame.font.init()


class Bird:
    """
    Create the bird

    """

    def __init__(self):
        self.y = 300
        self.rotation = 0
        self.bird_animation = 0
        self.speed = SPEED_BIRD
        self.bird = [
            pygame.transform.scale(
                pygame.image.load("image/bird1.png").convert_alpha(), BIRD_SIZE
            ),
            pygame.transform.scale(
                pygame.image.load("image/bird2.png").convert_alpha(), BIRD_SIZE
            ),
            pygame.transform.scale(
                pygame.image.load("image/bird3.png").convert_alpha(), BIRD_SIZE
            ),
        ]
        self.steps = 1
        self.jump_count = 0
        self.is_jump = False

    def jump(self, action: np.array) -> None:  # check this
        """
        Jump action

        Args:
            action: actions of the bird (jump or fall)

        """

        if action[1] == 1:
            self.is_jump = True
            self.jump_count = 12
        if self.jump_count - 3.3 > 0:
            self.y -= (self.jump_count - 3.3) ** 0.5
            self.jump_count -= 0.2
            self.rotation = 15
        else:
            self.is_jump = False
        self.y = max(-5, self.y)

    def bird_unit(self) -> dict:
        """
        Create parameters of the ellise(bird) for
        checking the interception with surface

        Returns:
            dict_el : parameters of thr ellips.

        """
        dict_el = {}
        dict_el["k"] = 0
        dict_el["y_c"] = self.y + 25
        dict_el["x_c"] = 216.5
        dict_el["a"] = 27.5 * 0.98
        dict_el["b"] = 20 * 0.95
        dict_el["y"] = self.y
        if self.rotation < -40:
            dict_el["a"], dict_el["b"] = dict_el["b"], dict_el["a"]
        return dict_el

    def update(self, window: pygame.surface.Surface) -> None:
        """
        Update condition of the bird

        Args:
            window: screen of the game

        """
        window.blit(
            pygame.transform.rotate(
                self.bird[self.bird_animation], self.rotation
            ),
            (BIRD_X, self.y),
        )
        self.steps = (self.steps + 1) % 20
        if not self.steps:
            self.bird_animation = (self.bird_animation + 1) % 3
        if not self.is_jump:
            self.y += GRAVITY * self.speed
            self.rotation = max(self.rotation - ROTATION_SPEED, -90)

    def speed_up(self, score: int) -> None:
        """
        Spped up falling the bird

        Args:
            score: score of the game

        """
        self.speed = min(SPEED_BIRD + SPEED_UP * score, 14)


class Tube:
    """
    Create tubes

    """

    def __init__(self):
        self.tube_up = pygame.image.load("image/tube.png").convert_alpha()
        self.tube_low = pygame.transform.rotate(self.tube_up, ANGLE_TUBE)
        self.x = 687.9
        self.y1 = np.random.randint(*TUBE_HIGH_BEETW)
        self.y2 = np.random.randint(*TUBE_HIGH_BEETW)

    def update(self, window: pygame.surface.Surface) -> None:
        """
        Updates drawings and tubes positions

        Args:
            window: screen of the game

        """
        window.blit(self.tube_up, (self.x, self.y1 - DELTA_COL_Y))
        window.blit(self.tube_low, (self.x, self.y1))
        window.blit(
            self.tube_up, (self.x + DELTA_COL_X, self.y2 - DELTA_COL_Y)
        )
        window.blit(self.tube_low, (self.x + DELTA_COL_X, self.y2))
        self.rule_tube()

    def rule_tube(self) -> None:
        """
        Update tubes positions

        """
        if self.x < -TUBE_WIDTH:
            self.x += DELTA_COL_X
            self.y1 = self.y2
            self.y2 = np.random.randint(*TUBE_HIGH_BEETW)
        self.x -= SPEED_SCREEN

    def scoring(self) -> int:
        """
        Check for getting a point in the game

        Return:
            1 if take point else 0
        """
        if abs(self.x + TUBE_WIDTH / 2 - BIRD_X) < SPEED_SCREEN / 2:
            return 1
        return 0

    def lost(self, bird_elip: dict) -> bool:
        """
        Check for losing the game through equation
            of intersection of a line and an ellipse

        Args:
            bird_elip : dictionary of parameters of the ellipse (bird)

        Returns:
            True if lost the game else False

        """
        x_tube = self.x + DELTA_COL_X if self.x < 100 else self.x
        y_tube = self.y2 if self.x < 100 else self.y1
        #   print(x_tube)
        if 107 < x_tube <= 235:
            for tube in [y_tube + 3, y_tube - 155]:
                if 106 < x_tube < 118:
                    break
                S = tube - bird_elip["k"] * x_tube - bird_elip["y_c"]
                A = bird_elip["b"] ** 2 + bird_elip["a"] ** 2 * bird_elip["k"]
                B = 2 * (
                    bird_elip["a"] ** 2 * bird_elip["k"] * S
                    - bird_elip["b"] ** 2 * bird_elip["x_c"]
                )
                C = (
                    (bird_elip["b"] * bird_elip["x_c"]) ** 2
                    + (bird_elip["a"] * S) ** 2
                    - (bird_elip["a"] * bird_elip["b"]) ** 2
                )
                D = B**2 - 4 * A * C
                D1 = -10
                if (
                    bird_elip["y"] + 39 > y_tube
                    or bird_elip["y"] + 2 < y_tube - 157
                ):
                    S1 = x_tube - bird_elip["x_c"]
                    A1 = bird_elip["a"] ** 2
                    B1 = 2 * (-bird_elip["a"] ** 2 * bird_elip["y_c"])
                    C1 = (
                        (bird_elip["a"] * bird_elip["y_c"]) ** 2
                        + (bird_elip["a"] * S1) ** 2
                        - (bird_elip["a"] * bird_elip["b"]) ** 2
                    )
                    D1 = B1**2 - 4 * A1 * C1
                if D > 0 or D1 > 0 or not D or not D1:
                    return True
        return False


class Ground:
    """
    Create ground surface

    """

    def __init__(self):
        self.bg_lower_part = pygame.image.load(
            "image/bg_lower_part.png"
        ).convert()
        self.x = 0

    def lost(self, bird_y: float, rot: float) -> bool:
        """
        Check for losing the game through intersection with ground

        Args:
            bird_y : y coordinate of the bird
            rot : angle of the bird ellipse

        Returns:
            True if lost  else False

        """

        gip_1 = np.round(np.cos(np.deg2rad(-rot)), 3) * BIRD_SIZE[1] * 0.9
        gip_2 = np.round(np.sin(np.deg2rad(-rot)), 3) * BIRD_SIZE[0] * 0.85

        if bird_y + gip_1 + gip_2 >= CHANGE_GROUND:
            return True
        return False

    def update(self, window: pygame.surface.Surface) -> None:
        """
        Update condition the ground

        Args:
            window : screen of the game

        """
        self.x = 0 if self.x <= -MOVING_SCREEN else self.x
        self.x -= SPEED_SCREEN
        window.blit(self.bg_lower_part, (self.x, CHANGE_GROUND))
        window.blit(
            self.bg_lower_part, (self.x + MOVING_SCREEN, CHANGE_GROUND)
        )


class Background:
    """
    Create the background.

    """

    def __init__(self):
        self.bg_upper_part = pygame.image.load(
            "image/bg_upper_part.png"
        ).convert()

    def update(self, window: pygame.surface.Surface) -> None:
        """
        Update background

        Args:
            window : screen of the game

        """
        window.blit(self.bg_upper_part, COOR_BACKG)


def restart() -> None:
    """
    Start new game

    """
    main()


def main(game_ag, window: pygame.surface.Surface, learning: bool) -> int:
    """
    Function create the game and its logic for agent's training

    Args:
        game_ag: training agent, from class 'agent_q_l_exp.Agent'
        window: screen of the game

    Returns:
        game_score : return score of the game

    """
    bird = Bird()
    tube = Tube()
    back = Background()
    g_d = Ground()
    game_score = 0
    max_sc = af.max_score()
    lost = False
    # state the system in start moment
    state = state_func(bird.y, tube.y1, tube.y2, tube.x, bird.speed)
    reward = 0
    while not lost:
        # choose action
        if np.random.random() < game_ag.epsilon:
            game_ag.epsilon = (
                game_ag.epsilon / 1.01 if game_ag.epsilon > 0 else 0
            )
            move_res = random.choices([0, 1])[0]
            game_ag.create_q(state)
            final_move = np.eye(2, dtype=int)[move_res]
        else:
            move_res = np.argmax(game_ag.create_q(state))
            # print(game_ag.create_q(state))
            final_move = np.eye(2, dtype=int)[move_res]
        bird.jump(final_move)
        [i.update(window) for i in [back, bird, tube, g_d]]
        if tube.scoring():
            reward += 1
            game_score += 1
        af.score_num(window, game_score)
        bird.speed_up(game_score)
        clock.tick()
        next_state = state_func(bird.y, tube.y1, tube.y2, tube.x, bird.speed)
        if tube.lost(bird.bird_unit()) or g_d.lost(bird.y, bird.rotation):
            reward = -1
            lost = True
            if max_sc < game_score:
                max_sc = game_score
        pygame.display.update()
        game_ag.train_step(state, final_move, reward, next_state, lost)
        state = next_state

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    return game_score


def state_func(
    bird_y: float, tube_y1: float, tube_y2: float, tube_x: float, speed: float
) -> tuple:
    """
    Create states of the system

    Args:
        bird_y : y coordinate of the bird.
        tube_y1 : y coordinate of the gap between columns 1
        tube_y2 :  y coordinate of the gap between columns 2
        tube_x : x  coordinate of the gap between column
        speed : speed of falling.

    Returns:
        state : tuple of bools values of the system.

    """
    y_tube = tube_y2 if tube_x <= 100 else tube_y1
    state = (
        int(bird_y > y_tube - BIRD_SIZE[1] - GRAVITY * speed - DELTA),
        int(bird_y < y_tube - GAP_COL + DELTA),
        int(tube_y1 > tube_y2),
    )

    return state


def training(epochs: int = 1000, learning: bool = False) -> None:
    """
    Creating and record parametes of the agent

    Args:
        epochs : numbers of games for training (the default is 1000)
        learning : True for learnin value func else False

    Returns:
        total_score : sum of scores for all games
        mean : mean value of the score in games
        stdev : stdev of the scores in games

    """
    games = 0
    score_plot = []
    counter_plot = []
    record = 0
    total_score = 0
    agent_fb = agent.Agent(
        learning=learning, load_q=True, path="weights_exper_ql.npy"
    )
    print(type(agent_fb))
    window = pygame.display.set_mode((WIDTH, HIGTH), vsync=1)
    for epoch in range(epochs):
        score = main(agent_fb, window)
        record = record if score < record else score
        games += 1
        if not games % 100:
            agent_fb.lr = max(agent_fb.lr / 1.1, 1e-5)
            np.save("weights_exper_ql.npy", agent_fb.Q)
            print("saved", agent_fb.lr)
        total_score += score
        print(
            f"""Game {games}, Score {score},
            Total score {total_score}, Eps{agent_fb.epsilon},
            Eps{agent_fb.lr}, {len(agent_fb.Q)}"""
        )
        score_plot.append(score)
        counter_plot.append(games)
        af.plot_seaborn(counter_plot, score_plot)
    mean, stdev = statistics.mean(score_plot), statistics.stdev(score_plot)
    np.save("weights_exper_ql.npy", agent_fb.Q)


if __name__ == "__main__":
    training(1000, False)
