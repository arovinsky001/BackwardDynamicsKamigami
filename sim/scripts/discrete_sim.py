import numpy as np
import torch
from torch.nn import functional as F

LEFT = -1
RIGHT = 1
UP = -10
DOWN = 10
possible_actions = [LEFT, RIGHT, UP, DOWN]

if __name__ == "__main__":
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

    # dump data as one hot vectors
    F.one_hot(torch.tensor(all_states), num_classes=100).squeeze().detach().numpy().dump("grid_data/states_one_hot.npz")
    F.one_hot(torch.tensor(all_actions), num_classes=100).squeeze().detach().numpy().dump("grid_data/actions_one_hot.npz")
    F.one_hot(torch.tensor(all_next_states), num_classes=100).squeeze().detach().numpy().dump("grid_data/next_states_one_hot.npz")
