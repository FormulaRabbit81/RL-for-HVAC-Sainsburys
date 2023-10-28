# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:47:50 2023

Initial code to investigate basic RL model for price minimisation of HVAC system.

States:
    - Room temp
    - External temp
    - Elec price

Action:
    - Choose temp setpoint

Training methodology

1. For each hourly timestep:
    - Load internal temp, external temp, and current electricity price (states)
    - Choose temperature setpoint (action)
    - Simulate 60 mins of PI controller working
        - Re calculating required heat load (Q_hvac) at each minute
        - Step building thermal model 1 min (simulation_enviornment) to calc new internal temperature
    - Final internal temperature calculation from thermal model (60th value) is internal temperature for next step
    - Calculate reward

Measured data required:
    - Only need measurements for external temperature and electricity price
    - Internal temperature can be initialised to any value close to desired setpoint,
      then calculated from building thermal model
    - Temperature setpoint chosen by RL agent
    - HVAC heat load calculated from PI controller


@author: Max
"""
import numpy as np
import simulation_environment as sim_env
from pi_controller import PIController, apply_system_constraints


class EpsilonGreedy:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def run(self, alpha, gamma, exttempsdf, pricingdf, mintemp=17, maxtemp=22, priceweight=1, comfortweight=1, initialvals=None):
        """Epsilon greedy policy
        alpha: Step size for Q-Learning algorithm
        gamma: Discount value for Q-Learning algorithm
        exttempsdf: Dataframe of external temperatures (index 0)
        pricingdf: Dataframe of energy pricing in kWh (index 1)
        mintemp: Minimum desired temperature
        maxtemp: Maximum desired temperature
        priceweight: Weighting agent uses for pricing priority,
        comfortweight: Weighting agent uses for temperature priority
        (initialvals: Starter State-Action value table for agent)
        """

        def rewardfunc(price, temp, priceweight, comfortweight, qhvac):
            if temp > maxtemp or temp < mintemp:
                temp_comp = -1
            else:
                temp_comp = 0
            return comfortweight * temp_comp - priceweight * price * qhvac

        def findstate(tempin, tempout, price):
            """Convert state to matrix coordinates."""
            outtemp = int((rounddegrees(tempout) - rounddegrees(min(exttempsdf.iloc[:, 0]))) * 2)
            intemp = int((rounddegrees(tempin) - mintemp - 5) * 2)  # -1 for 1 degree of allowed comfort violation
            prc = int((roundtopenny(price) - roundtopenny(min(pricingdf.iloc[:, 1]))) * 10)
            return [outtemp, intemp, prc]

        # Initialise values for later
        tempsets = []
        energy = []
        buildingtemps = []
        rewardslog = []
        tempstates = (maxtemp - mintemp) * 2 + 4  # Allows a comfort violation of 1 degree
        timesselected = np.zeros((discretisetemp(exttempsdf), tempstates, discretisemoney(pricingdf), tempstates))
        qtable = initialvals if initialvals is not None else np.zeros((discretisetemp(exttempsdf), tempstates, discretisemoney(pricingdf), tempstates))
        print(qtable.shape)
        t_sensors = [16] * 13
        rc_vals = [0.0496, 0.5862, 0.1383, 59.01, 247.8, 0.1576, 0.0289, 1749100, 388300]
        sales_area_dims = [117, 64.866, 9]  # width, length, height
        tempin = t_sensors[-1]
        q_max = 500000  # [W] maximum combined heat load for AHUs
        newstate = findstate(tempin, exttempsdf.iloc[0, 0], pricingdf.iloc[0, 1])
        thermal_model = sim_env.ThermalNetwork(rc_vals, sales_area_dims, t_sensors, dt=60)  # number of second for each timestemp. Needs to be 60

        # Loop every hour
        for t in range(len(exttempsdf) - 1):
            currentstate = newstate
            # Choose an action from epsilon greedy policy
            if self.epsilon > np.random.uniform(0, 1):
                i = np.random.randint(0, high=tempstates - 1)
            else:
                i = np.argmax(qtable[currentstate[0], currentstate[1], currentstate[2], :])
            timesselected[currentstate[0], currentstate[1], currentstate[2], i] += 1
            t_sp = i / 2 + mintemp - 1  # setpoint temperature

            # Take action
            pi_controller = PIController(max_heat_duty=q_max)
            pi_signal = pi_controller.pi_signal(t_sp, tempin)
            q_hvac = apply_system_constraints(pi_signal, q_max)
            model_inputs = np.array([exttempsdf.iloc[t, 0]] * 5 + [q_hvac])
            t_sensors = thermal_model.calcNextTimestep(model_inputs)
            tempin = t_sensors[-1]
            reward = rewardfunc(pricingdf.iloc[t, 1], tempin, priceweight, comfortweight, q_hvac)  # Observe R
            newstate = findstate(tempin, exttempsdf.iloc[t + 1, 0], pricingdf.iloc[t + 1, 1])  # Observe S'
            # Q(S, A) ← Q(S, A) + α[R + γargmax Q(S', a) − Q(S, A)]
            qtable[currentstate[0], currentstate[1], currentstate[2], i] += alpha * (reward + gamma * np.max(qtable[newstate[0], newstate[1], newstate[2], :]) - qtable[currentstate[0], currentstate[1], currentstate[2], i])
            tempsets.append(t_sp)
            energy.append(q_hvac)
            buildingtemps.append(tempin)
            rewardslog.append(reward)
        self.tempsets = tempsets
        self.energy = energy
        self.buildingtemps = buildingtemps
        self.qtable = qtable
        # print(np.count_nonzero(qtable == 0))
        return tempsets

    def greedyrun(self, runlength, exttempsdf, pricingdf, mintemp=17, maxtemp=22, priceweight=1, comfortweight=1, initialvals=None):
        """To be used once behaviour is learned from a run."""
        def findstate(tempin, tempout, price):
            """Convert state to matrix coordinates."""
            outtemp = int((rounddegrees(tempout) - rounddegrees(min(exttempsdf.iloc[:, 0]))) * 2)
            intemp = int((rounddegrees(tempin) - mintemp - 5) * 2)  # -1 for 1 degree of allowed comfort violation
            prc = int((roundtopenny(price) - roundtopenny(min(pricingdf.iloc[:, 1]))) * 10)
            return [outtemp, intemp, prc]

        # Initialise values for later
        tempsets = []
        energy = []
        buildingtemps = []
        tempstates = (maxtemp - mintemp) * 2 + 4  # Allows a comfort violation of 1 degree
        timesselected = np.zeros((discretisetemp(exttempsdf), tempstates, discretisemoney(pricingdf), tempstates))
        qtable = self.qtable
        t_sensors = [16] * 13
        rc_vals = [0.0496, 0.5862, 0.1383, 59.01, 247.8, 0.1576, 0.0289, 1749100, 388300]
        sales_area_dims = [117, 64.866, 9]  # width, length, height
        tempin = t_sensors[-1]
        q_max = 500000  # [W] maximum combined heat load for AHUs
        newstate = findstate(tempin, exttempsdf.iloc[0, 0], pricingdf.iloc[0, 1])
        thermal_model = sim_env.ThermalNetwork(rc_vals, sales_area_dims, t_sensors, dt=60)  # number of second for each timestemp. Needs to be 60

        # Loop every hour
        for t in range(runlength - 1):
            currentstate = newstate
            i = np.argmax(qtable[currentstate[0], currentstate[1], currentstate[2], :])
            timesselected[currentstate[0], currentstate[1], currentstate[2], i] += 1
            t_sp = i / 2 + mintemp - 1  # setpoint temperature
            # Take action
            pi_controller = PIController(max_heat_duty=q_max)
            pi_signal = pi_controller.pi_signal(t_sp, tempin)
            q_hvac = apply_system_constraints(pi_signal, q_max)
            model_inputs = np.array([exttempsdf.iloc[t, 0]] * 5 + [q_hvac])
            t_sensors = thermal_model.calcNextTimestep(model_inputs)
            tempin = t_sensors[-1]
            newstate = findstate(tempin, exttempsdf.iloc[t + 1, 0], pricingdf.iloc[t + 1, 1])  # Observe S'
            tempsets.append(t_sp)
            energy.append(q_hvac)
            buildingtemps.append(tempin)
        self.learned_tempsets = tempsets
        self.learned_energy = energy
        self.learned_buildingtemps = buildingtemps
        return tempsets


def roundtopenny(price):
    return round(price, 1)


def rounddegrees(temp):
    return round(temp * 2) / 2


def discretisetemp(dataframe):
    ranje = rounddegrees(np.max(dataframe.iloc[:, 0])) - rounddegrees(np.min(dataframe.iloc[:, 0]))
    return int(2 * ranje) + 1


def discretisemoney(dataframe):
    ranje = roundtopenny(np.max(dataframe.iloc[:, 1])) - roundtopenny(np.min(dataframe.iloc[:, 1]))
    return int(10 * ranje) + 1


"""
x generalise state-action space by taking mins and maxes and use for dimensions
x avoid recalculating states
x make reward function weighted
x Add docstring for all parameters
- Make epsilon step function?
x Tidy up file
"""
"""
            a=currentstate[0]
            b=currentstate[1]
            c=currentstate[2]
discretise after rounding, not before
"""
