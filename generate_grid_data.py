import numpy as np
import torch
from torch.nn import functional as F

LEFT = -1
RIGHT = 1
UP = -10
DOWN = 10
possible_actions = [LEFT, RIGHT, UP, DOWN]

state = 50

all_states = []
all_actions = []
all_next_states = []

for _ in range(1000000):
    all_states.append(state)
    action = np.random.randint(0, 4)
    all_actions.append(action)
    state += possible_actions[action]
    state = state % 100
    all_next_states.append(state)

np.array(all_states).dump("grid_data/n_states.npz")
F.one_hot(torch.tensor(all_actions)).squeeze().detach().numpy().dump("grid_data/n_actions_one_hot.npz")
np.array(all_next_states).dump("grid_data/n_next_states.npz")
