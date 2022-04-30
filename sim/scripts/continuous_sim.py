import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from ..params.continuous_sim_params import *


def generate_data(stochastic=False):
    all_actions = np.random.rand(N_THREADS, N_STEPS, 2)
    all_actions *= TRAINING_MAX_MAGNITUDE
    all_actions -= TRAINING_MAX_MAGNITUDE / 2

    if stochastic:
        noise = np.random.normal(loc=0.0, scale=NOISE_STD, size=all_actions.shape)

    states = np.meshgrid(np.arange(SQRT_N_THREADS), np.arange(SQRT_N_THREADS))
    states = np.stack(states, axis=2).reshape(-1, 2).astype("float64")
    assert states.shape[1] == 2 and len(states.shape) == 2 and all_actions.shape[0] == states.shape[0]

    if LIMIT:
        all_states = np.empty_like(all_actions)
        all_next_states = np.empty_like(all_actions)

        for i in trange(N_STEPS):
            all_states[:, i] = states
            states += all_actions[:, i] + noise[:, i] if stochastic else all_actions[:, i]
            states = np.clip(states, MIN_STATE, MAX_STATE)
            all_next_states[:, i] = states
    else:
        all_next_states = states[:, None, :] + all_actions.cumsum(axis=1) + noise.cumsum(axis=1)
        all_states = np.append(states[:, None, :], all_next_states[:, :-1, :], axis=1)

    all_states = all_states.reshape(-1, 2)
    all_actions = all_actions.reshape(-1, 2)
    all_next_states = all_next_states.reshape(-1, 2)

    print("samples collected:", len(all_states))
    
    suffix = "stochastic" if stochastic else "deterministic"
    np.savez_compressed(DATA_PATH + f"data_continuous_{suffix}.npz", states=all_states,
                        actions=all_actions, next_states=all_next_states)

def visualize_data(n_steps):
    data = np.load(DATA_PATH + f"data_continuous_stochastic.npz")
    states = data['states'][:n_steps*SQRT_N_THREADS**2:SQRT_N_THREADS**2][:n_steps]
    actions = data['actions'][:n_steps*SQRT_N_THREADS**2:SQRT_N_THREADS**2][:n_steps]
    next_states = data['next_states'][:n_steps*SQRT_N_THREADS**2:SQRT_N_THREADS**2][:n_steps]

    plt.figure()
    plt.plot(states[:, 0], states[:, 1], 'bo')
    plt.plot(next_states[:, 0], next_states[:, 1], 'ro')
    for i in range(len(states)):
        plt.arrow(states[i, 0], states[i, 1], actions[i, 0], actions[i, 1], width=0.1, head_width=1.0)
    plt.show()


if __name__ == '__main__':
    generate_data(stochastic=STOCHASTIC)
    # visualize_data(10)
