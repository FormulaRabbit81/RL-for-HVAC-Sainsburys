# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 11:12:52 2021

@author: maxbi
"""
import numpy as np
import pandas as pd
from scipy.signal import StateSpace


class ExternalSurface:
    # individual wall or roof represented by 3R2C model
    def __init__(self, r1, r3, r5, c2, c4, width, height):
        self.R1 = r1/(width*height)
        self.R3 = r3/(width*height)
        self.R5 = r5/(width*height)
        self.C2 = c2*width*height
        self.C4 = c4*width*height   

class InternalSurface:
    # all internal mass of store represented by one 2R2C model
    def __init__(self, r1, r2, c1, c2, surface_area):
        self.R1 = r1/surface_area
        self.R2 = r2/surface_area
        self.C1 = c1*surface_area
        self.C2 = c2*surface_area

class InternalAir:
    # class to represent characteristics of internal air within store
    def __init__(self, width, height, depth):
        cp_air = 1.012 * 1.1644 * 1000 # kJ/(kg.C) * kg/m3 * 1000 J/kJ = J/(m3.C)
        self.Cin = cp_air*width*height*depth

class ThermalNetwork:
    # system of linear heat transfer ODEs which we use to model the store
    # all units in the object must be supplied in SI form, units of equations are [W]
    def __init__(self, rc_vals, sales_area_dims, T_init, dt, occupantHeatGain=False, solar_gain=False):
        """
        T_init: numpy array of temperatures to initialise thermal model - ["T_w12", "T_w14", "T_w22", "T_w24", "T_w32", "T_w34", "T_w42", "T_w44", "T_r2", "T_r4", "T_i1", "T_i2", "T_in"]
        dt: Timestep for discretised approximation of ODE system [s]
        occupantHeatGain: not coded in yet
        solarGain: not coded in yet
        """
        width, depth, height = sales_area_dims
        internal_surface_area = 2*width*depth # estimate for the surface area of lumped internal mass of the store

        self.walls = [ExternalSurface(rc_vals[0], rc_vals[1], rc_vals[2], rc_vals[3], rc_vals[4], width, height) for x in range(2)]
        self.walls += [ExternalSurface(rc_vals[0], rc_vals[1], rc_vals[2], rc_vals[3], rc_vals[4], depth, height) for x in range(2)]
        self.roof = [ExternalSurface(rc_vals[0], rc_vals[1], rc_vals[2], rc_vals[3], rc_vals[4], width, depth)]
        self.internalMass = [InternalSurface(rc_vals[5], rc_vals[6], rc_vals[7], rc_vals[8], internal_surface_area)]
        self.internalAir = InternalAir(width, depth, height)

        self.T_init = T_init # store initial temperature values
        self.x = T_init # initialise state variables
        self.dt = dt 
        self.solar_gain_bool = solar_gain

        # store the number of walls/roofs/internal mass
        self.n_walls = len(self.walls)
        self.n_roof = len(self.roof)
        self.n_internalMass = len(self.internalMass)

        # define number of states and inputs for model
        self.n_states = self.n_walls*2 + self.n_roof*2 + self.n_internalMass*2 + 1 # number of states for the system (Temperatures within the RC network)
        self.n_inputs = self.n_walls + self.n_roof + self.solar_gain_bool + 1 # number of control variables [controls + disturbances]

        # each row is one equation, each column is the state/control/disturbance variable - dx/dt = Ax + Bu, dy/dt = Cx + Du
        # T_in is always the bottom variable in the vector by definition 
        self.A = np.zeros((self.n_states, self.n_states)) # [T_w12, T_w14, T_w22, T_w24, T_w32, T_w34, T_w42, T_w44, T_r2, T_r4, T_m1, T_m2, T_in]
        self.B = np.zeros((self.n_states, self.n_inputs)) # [T_o1, T_o2, T_o3, T_o4, T_or, Q_solar, Q_hvac]
        self.C = np.zeros((1, self.n_states))
        self.C[0][-1] = 1 # ammend value for T_in term, thats the only control variable we can measure
        self.D = np.zeros((1, self.n_inputs))
        
        # initialise thermal model with specified design, first add walls and roof, then add internal mass, then add internal air - MUST KEEP THIS ORDER
        # these functions populate the self.A, self.B, self.C, self.D matricies in order to solve the linear system of equations using later class methods
        self.add_walls_and_roof()
        self.add_internal_mass()
        self.add_internal_air()


    def add_walls_and_roof(self):
        # function to update A and B matricies based on number of walls/roofs specified in the instance initialisation
        # models walls and roofs using 3R2C model

        walls_and_roof = self.walls + self.roof # list of all walls and roofs
        for x in range(self.n_walls + self.n_roof):
            w = walls_and_roof[x]
            # node 2 ODE coefficients
            self.A[0+2*x][0+2*x] = -1/(w.R1*w.C2) - 1/(w.R3*w.C2) # T_wi2 coeff 
            self.A[0+2*x][1+2*x] = 1/(w.R3*w.C2) # T_wi4 coeff 
            self.B[0+2*x][x] = 1/(w.R1*w.C2) # T_out,wi coeff 
            # node 4 ODE coefficients
            self.A[1+2*x][0+2*x] = 1/(w.C4*w.R3) # T_wi2 coeff 
            self.A[1+2*x][1+2*x] = -1/(w.R3*w.C4) - 1/(w.R5*w.C4) # T_wi4 coeff
            self.A[1+2*x][-1] = 1/(w.R5*w.C4) # T_in coeff 

    def add_internal_mass(self):
        # function to update A and E matricies for internal mass of building
        # models internal mass using 2R2C model

        start_row = 2*(self.n_walls + self.n_roof) # each wall/roof has 2 ODEs, so we want to add to the 2n row of A to get coeffs in right place
        solar_gain_col = self.n_walls + self.n_roof


        for x in range(self.n_internalMass):
            i = self.internalMass[x]
            # node 1 of the 2R2C model for the internal mass
            self.A[start_row+2*x][start_row+2*x] = -1/(i.R1*i.C1)
            self.A[start_row+2*x][start_row+1+2*x] = 1/(i.R1*i.C1)
            if self.solar_gain_bool:
                self.B[start_row+2*x][solar_gain_col] = 1/(2*i.C1) # 1/2 of total solar_gain in this node

            # node 2 of the 2R2C model for the internal mass
            self.A[start_row+1+2*x][start_row+2*x] = 1/(i.R1*i.C2)
            self.A[start_row+1+2*x][start_row+1+2*x] = -1/(i.R1*i.C2) - 1/(i.R2*i.C2)
            self.A[start_row+1+2*x][-1] = 1/(i.R2*i.C2)
            if self.solar_gain_bool:
                self.B[start_row+2*x][solar_gain_col] = 1/(2*i.C2) # 1/2 of total solar_gain in this node

    def add_internal_air(self):       
        # one big ODE which represents internal air volume within the store
        row_num = -1 # T_in equation will always be last row of system of ODEs by definition. 

        ## wall and roof terms
        walls_and_roof = self.walls + self.roof   
        for x in range(self.n_walls + self.n_roof):
            col_num = 1+2*x # start column number for T_w4 values
            w = walls_and_roof[x]
            self.A[row_num][col_num] =  1/(self.internalAir.Cin*w.R5) # need one of these for each wall/roof temp

        # internal mass T_i2 terms
        for x in range(self.n_internalMass):
            col_num = 2*(self.n_walls + self.n_roof) + 1 + 2*x # column number for T_i2 values
            i = self.internalMass[x]
            self.A[row_num][col_num] = 1/(self.internalAir.Cin*i.R2)

        # T_in terms - most complicated
        T_wall_component = -sum([1/(self.internalAir.Cin*w.R5) for w in self.walls])
        T_roof_component = -sum([1/(self.internalAir.Cin*r.R5) for r in self.roof])
        T_internalMass_component = -sum([1/(self.internalAir.Cin*i.R2) for r in self.internalMass])
        self.A[row_num][-1] = T_wall_component + T_roof_component + T_internalMass_component

        # Input terms
        self.B[row_num][-1] = 1/self.internalAir.Cin # Q_hvac      

    def continuousSystem(self):
        return(StateSpace(self.A, self.B, self.C, self.D))

    def discreteSystem(self, dt):
        return(self.continuousSystem().to_discrete(dt))

    def calcNextTimestep(self, u: np.array):
        """
        u: column vector of control variable inputs (control + disturbances) - [T_o1, T_o2, T_o3, T_o4, T_or, Q_solar, Q_hvac] - HAVE TO KEEP THIS ORDER
        dt: timestep to calculate k+1 values [s]

        """
        sysd = self.discreteSystem(self.dt)
        self.x = np.dot(sysd.A, self.x) + np.dot(sysd.B, u) # update state variables 

        return self.x

    def simulateTemperature(self, u_measured):
        """
        u_measured: numpy matrix of measured u control/disturbance values over the simulation period - [T_o1, T_o2, T_o3, T_o4, T_or, Q_hvac]
                    Number of rows in u_measured is the number of timeintervals which will be simulated, needs to match with dt size for calcNextTimestep
        """
        tempProfiles = np.zeros(shape=(len(u_measured), len(self.T_init))) # rows for each timestep, columns for each x value 
        tempProfiles[0] = self.T_init

        for step in range(len(tempProfiles)-1):
            tempProfiles[step+1] = self.calcNextTimestep(u_measured[step])

        return tempProfiles


def main(rc_vals, T_init, dt, sales_area_dims):  

    thermal_network = ThermalNetwork(rc_vals, sales_area_dims, T_init, dt)

    u0=np.array([10, 10, 10, 10, 10, 300])
    nextTimestep = thermal_network.calcNextTimestep(T_init, u0)

    return thermal_network, nextTimestep

if __name__ == "__main__":

    sales_area_dims = pd.read_excel("./data/master_data.xlsx", sheet_name="sales_area_dims")
    rc_vals = rc_init = np.array(pd.read_excel("./data/master_data.xlsx", sheet_name="rc_init")).flatten()
    T_init = np.array([12, 15, 12, 15, 12, 15, 12, 15, 12, 15, 22, 20, 19])
    dt = 15*60

    thermal_network, next_timestep_solar = main(rc_vals, T_init, dt, sales_area_dims)
