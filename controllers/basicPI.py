"""Baseline controller for comparison"""


import numpy as np
import simulation_environment as sim_env
from pi_controller import PIController, apply_system_constraints


class PI:

    def __init__(self):
        pass

    def run(self, exttempsdf, pricingdf, cmintemp=17, omintemp=20, maxtemp=22):
        tempsets = []
        energy = []
        buildingtemps = []
        t_sensors = [16] * 13
        rc_vals = [0.0496, 0.5862, 0.1383, 59.01, 247.8, 0.1576, 0.0289, 1749100, 388300]
        sales_area_dims = [117, 64.866, 9]  # width, length, height
        tempin = t_sensors[-1]
        t_sp = 17
        q_max = 500000  # [W] maximum combined heat load for AHUs
        seconds = 60
        thermal_model = sim_env.ThermalNetwork(rc_vals, sales_area_dims, t_sensors, dt=seconds)  # number of second for each timestemp. Needs to be 60
        pi_controller = PIController(max_heat_duty=q_max)
        moneys = []
        bmoneys = []
        comfortviols = []
        minutetemp = tempin
        # Loop every hour
        for t in range(len(exttempsdf) - 1):
            if 4 < (t) % 24 < 23 and t % 168 < 144:
                t_sp = omintemp
            elif 7 < (t) % 24 < 17 and t % 168 > 144:
                t_sp = omintemp
            else:
                t_sp = cmintemp  # setpoint temperature
            measuredtemps = []
            qhvacs = []
            for step in range(3600 // seconds):
                pi_signal = pi_controller.pi_signal(t_sp, minutetemp)
                q_hvac = apply_system_constraints(pi_signal, q_max)
                model_inputs = np.array([exttempsdf.iloc[t, 0]] * 5 + [q_hvac])
                t_sensors = thermal_model.calcNextTimestep(model_inputs)
                minutetemp = t_sensors[-1]
                measuredtemps.append(minutetemp)
                qhvacs.append(q_hvac)
            qhvac = sum(qhvacs) * seconds / 3600
            tempin = sum(measuredtemps) * seconds / 3600
            tempsets.append(t_sp)
            energy.append(qhvac)
            buildingtemps.append(tempin)
            moneys.append(pricingdf.iloc[t, 1] * qhvac / 2)
            bmoneys.append(pricecalculator(t) * qhvac / 2)
            if 6 < (t) % 24 < 23 and t % 168 < 144:
                if omintemp > tempin:
                    comfortviols.append(omintemp - tempin)
                else:
                    comfortviols.append(0)
            elif 9 < (t) % 24 < 17 and t % 168 > 144:
                if omintemp > tempin:
                    comfortviols.append(omintemp - tempin)
                else:
                    comfortviols.append(0)
            else:
                comfortviols.append(0)
        self.bmoneys = bmoneys
        self.tempsets = tempsets
        self.energy = energy
        self.buildingtemps = buildingtemps
        self.moneys = moneys
        self.comfortviols = comfortviols


def pricecalculator(timestep):
    if 17 <= (timestep % 24) < 21:
        price = 0.3
    elif (timestep % 24) < 8 or (timestep % 24) > 23:
        price = 0.1
    else:
        price = 0.2
    return price
