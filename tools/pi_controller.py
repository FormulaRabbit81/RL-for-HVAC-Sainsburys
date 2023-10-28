# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:16:49 2021

@author: maxbi
"""


def apply_system_constraints(pi_signal: float, q_max: int) -> float:
    """Heat loads are in [W]"""
    heat_load = pi_signal
    # min/max heat duty constraints
    if heat_load < 0:
        heat_load = 0
    elif heat_load > q_max:
        heat_load = q_max

    return heat_load


class PIController:
    def __init__(self, max_heat_duty: int, proportional_gain=1e6, integral_gain=5e4):
        self.q_max = max_heat_duty
        self.p_gain = proportional_gain
        self.i_gain = integral_gain
        self.ei_t = 0  # current value of integral error
        self.errors = []  # list of instantaneous errors

    def calc_pi_signal(self):
        return self.p_gain * self.errors[-1] + self.i_gain * self.ei_t

    def pi_signal(self, setpoint: float, measured: float) -> float:
        # calculate errors and PI signal
        self.errors.append(setpoint - measured)
        self.ei_t += self.errors[-1]
        pi_signal = self.calc_pi_signal()

        # anti-windup
        if pi_signal > self.q_max or pi_signal < 0:
            self.ei_t -= self.errors[-1]  # subtract error from this timestep to prevent excessive error accumulation
            pi_signal = self.calc_pi_signal()

        return pi_signal
