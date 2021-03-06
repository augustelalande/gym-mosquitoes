from __future__ import division, print_function, absolute_import

import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np

from gym_mosquitoes.envs.plotter import Plotter

import logging


logger = logging.getLogger(__name__)


BETA_M = None
BETA_H = None
MU_M = None
MU_H = None
K_M = None
K_H = None
OMEGA = None
SIGMA_A = None
SIGMA_S = None
ZETA = 0.75
_A = None
_P = 0.2
_R = 1 / 6
_R_S = _R
_R_A = _R
_B = 0.5
LAMBDA = None
UPSILON = None
THETA = 1
PHI = 1
TAU = 0.5
TAU_A = TAU
TAU_S = TAU
_Q = None

MUTATION_RATE = 0.001


class RandMosquitoesEnv(gym.Env):

    def __init__(self):
        self.reward_range = (-np.inf, 0)
        self.action_space = spaces.Box(
            low=0, high=0.1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32)

        self.plotter = None
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action), \
            "Action {} is not a valid action".format(action)

        delta_S = self.delta_S(*action)
        delta_I_s = self.delta_I_s(*action)
        delta_I_a = self.delta_I_a(*action)
        delta_J_s = self.delta_J_s(*action)
        delta_J_a = self.delta_J_a(*action)
        delta_T_s = self.delta_T_s(*action)
        delta_T = self.delta_T(*action)
        delta_T_a = self.delta_T_a(*action)
        delta_R = self.delta_R(*action)
        delta_M_s = self.delta_M_s(*action)
        delta_M_r = self.delta_M_r(*action)

        self.S += delta_S
        self.I_s += delta_I_s
        self.I_a += delta_I_a
        self.J_s += delta_J_s
        self.J_a += delta_J_a
        self.T_s += delta_T_s
        self.T += delta_T
        self.T_a += delta_T_a
        self.R += delta_R
        self.M_s += delta_M_s
        self.M_r += delta_M_r

        self.t += 1

        reward = -(self.I_s + self.J_s)
        state = self.get_state()

        return state, reward, False, {}

    def reset(self, soft_reset=False):
        if self.plotter is not None:
            self.plotter.reset()

        if not soft_reset:
            global BETA_M, BETA_H, MU_M, MU_H, K_M, K_H, OMEGA, \
                SIGMA_A, SIGMA_S, _A, LAMBDA, UPSILON, _Q
            BETA_M = self._randrange(0.03, 0.2)
            BETA_H = self._randrange(0.18, 0.9)
            MU_M = self._randrange(1/21, 1/7)
            MU_H = self._randrange(1/(60*365), 1/(45*365))
            K_M = self._randrange(0, 1)
            K_H = self._randrange(0, 1)
            OMEGA = self._randrange(1/370, 1/28)
            SIGMA_A = self._randrange(1/180, 1/33)
            SIGMA_S = self._randrange(1/180, 1/33)
            _A = self._randrange(1/10, 1/3)
            LAMBDA = self._randrange(0.25, 0.5)
            UPSILON = self._randrange(0.01, 0.05)
            _Q = self._randrange(0.15, 0.2)

        self.t = 0

        self.M_s = 0.0
        self.M_r = 0.0
        self.S = 0.9
        self.I_s = 0.04
        self.I_a = 0.04
        self.J_s = 0.01
        self.J_a = 0.01
        self.T_s = 0.0
        self.T = 0.0
        self.T_a = 0.0
        self.R = 0.0

        return self.get_state()

    def render(self, max_x=20000):
        if self.plotter is None:
            self.plotter = Plotter(
                [
                    "Sensitive strain infections",
                    "Resistant strain infections",
                    "Total infections"
                ],
                ['g', 'r', 'b'],
                max_x=max_x
            )

        self.plotter.update(
            self.t,
            [
                self.I_s,
                self.J_s,
                self.I_s + self.J_s
            ]
        )

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _randrange(self, low, high):
        rand = self.np_random.random_sample()
        return low + rand * (high - low)

    def get_state(self):
        return np.array([
            self.I_s + self.J_s,  # symptomatic population
            self.T_s,  # treated symptomatic
            self.T + self.T_a,  # treated asymptomatic or healthy
            self.S + self.I_a + self.J_a + self.R  # the rest
        ])

    def delta_S(self, C):
        a = MU_H - MU_H * self.S
        b = BETA_H * (self.M_s + K_H * self.M_r) * self.S
        c = _Q * C * self.S
        d = (1 - ZETA) * SIGMA_A * (self.I_a + PHI * self.J_a)
        e = (1 - _B) * _R_A * self.T_a
        f = _R * self.T + OMEGA * self.R
        return a - b - c + d + e + f

    def delta_I_s(self, C):
        a = LAMBDA * BETA_H * self.M_s * self.S
        b = UPSILON * self.I_a
        c = (_P * _A + SIGMA_S) * self.I_s
        d = MU_H * self.I_s
        return a + b - c - d

    def delta_I_a(self, C):
        a = (1 - LAMBDA) * BETA_H * self.M_s * self.S
        b = _Q * C * self.I_a
        c = UPSILON * self.I_a
        d = SIGMA_A * self.I_a
        e = MU_H * self.I_a
        return a - b - c - d - e

    def delta_J_s(self, C):
        a_1 = LAMBDA * K_H * BETA_H * self.M_r
        a_2 = self.S + TAU_S * self.T_s + TAU * self.T + TAU_A * self.T_a
        a = a_1 * a_2
        b = THETA * UPSILON * self.J_a
        c = PHI * SIGMA_S * self.J_s
        d = MU_H * self.J_s
        return a + b - c - d

    def delta_J_a(self, C):
        a_1 = (1 - LAMBDA) * K_H * BETA_H * self.M_r
        a_2 = self.S + TAU_S * self.T_s + TAU * self.T + TAU_A * self.T_a
        a = a_1 * a_2
        b = PHI * SIGMA_A * self.J_a
        c = THETA * UPSILON * self.J_a
        d = MU_H * self.J_a
        return a - b - c - d

    def delta_T_s(self, C):
        a = _P * _A * self.I_s
        b = _R_S * self.T_s
        c = TAU_S * K_H * BETA_H * self.M_r * self.T_s
        d = MU_H * self.T_s
        return a - b - c - d

    def delta_T(self, C):
        a = _Q * C * self.S
        b = _R * self.T
        c = TAU * K_H * BETA_H * self.T * self.M_r
        d = MU_H * self.T
        return a - b - c - d

    def delta_T_a(self, C):
        a = _Q * C * self.I_a
        b = _R_A * self.T_a
        c = TAU_A * K_H * BETA_H * self.T_a * self.M_r
        d = MU_H * self.T_a
        return a - b - c - d

    def delta_R(self, C):
        a = _R_S * self.T_s
        b = _B * _R_A * self.T_a
        c = ZETA * SIGMA_A * (self.I_a + PHI * self.J_a)
        d = SIGMA_S * self.I_s
        e = PHI * SIGMA_S * self.J_s
        f = OMEGA * self.R
        g = MU_H * self.R
        return a + b + c + d + e - f - g

    def delta_M_s(self, C):
        a = BETA_M * (1 - self.M_s - self.M_r) * (self.I_a + self.I_s)
        b = MU_M * self.M_s
        return a - b

    def delta_M_r(self, C):
        a = K_M * BETA_M * (1 - self.M_s - self.M_r) * (self.J_a + self.J_s)
        b = MU_M * self.M_r
        return a - b + MUTATION_RATE * self.M_s
