from math import *
import random
import numpy as np
import numba

@numba.jit('f8(f8)', nopython=True)
def potential(x):
    w = 30
    p = 0.5 * exp(-(x-12)**2/w) + \
        0.8 * exp(-(x-37)**2/w) + \
        1.0 * exp(-(x-65)**2/w) + \
        0.5 * exp(-(x-87)**2/w)
    return -log(p)

@numba.jit('f8(f8, f8, f8, f8)', nopython=True)
def new_position(pos, stepsize, min, max):
    direction = 1 if (random.random() > 0.5) else -1
    new_pos = pos + (random.random()*stepsize) * direction
    if new_pos < min or new_pos > max:
        return new_position(pos, stepsize, min, max)
    else:
        return new_pos

@numba.jit('f8[:](f8, i8, i8, f8, f8, f8)', nopython=True)
def run_mc(pos, step, interval, stepsize, min, max):
    pot = potential(pos)
    traj = []
    for i in range(step):
        new_pos = new_position(pos, stepsize, min, max)
        new_pot = potential(new_pos)
        diff = new_pot - pot
        crit = True if diff < 0 or exp(-diff/0.6) > random.random() else False
        if crit:
            pos = new_pos
            pot = new_pot
        if i % interval == 0:
            traj.append(pos)
    return np.array(traj)

class simulation():
    def __init__(self):
        self.traj = []
        self.stepsize = 0.5
        self.boundary = [0, 100]
        self.set_position(0)
        self.interval = 10
    
    def set_position(self, x):
        self.pos = x
        self.pot = potential(x)
        
    def run_mc(self, step):
        traj = run_mc(self.pos, step, self.interval, self.stepsize, self.boundary[0], self.boundary[1])
        self.traj = traj
        self.pos = traj[-1]
        self.pot = potential(self.pos)