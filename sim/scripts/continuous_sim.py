import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
import matplotlib.cm as cm

MAGNITUDE = 20.0
SQRT_N_THREADS = 100
N_ITERS = 1000
DATA_PATH = "/Users/Obsidian/Desktop/eecs106b/projects/BackwardDynamicsKamigami/sim/data/continuous/"

def generate_data():
    all_actions = np.random.rand(N_ITERS, SQRT_N_THREADS**2, 2)
    all_actions *= MAGNITUDE
    all_actions -= MAGNITUDE / 2

    states = np.meshgrid(np.arange(SQRT_N_THREADS), np.arange(SQRT_N_THREADS))
    states = np.stack(states, axis=2).reshape(-1, 2).astype("float64")
    assert states.shape[1] == 2 and len(states.shape) == 2 and all_actions.shape[1] == states.shape[0]

    all_states = np.empty((N_ITERS, SQRT_N_THREADS**2, 2))
    all_next_states = np.empty((N_ITERS, SQRT_N_THREADS**2, 2))

    for i in trange(N_ITERS):
        all_states[i] = states
        states += all_actions[i]
        states = np.clip(states, 0, 100)
        all_next_states[i] = states

    all_states = all_states.reshape(-1, 2)
    all_actions = all_actions.reshape(-1, 2)
    all_next_states = all_next_states.reshape(-1, 2)

    print("samples collected:", len(all_states))
    
    all_states.dump(DATA_PATH + "states_continuous.npz")
    all_actions.dump(DATA_PATH + "actions_continuous.npz")
    all_next_states.dump(DATA_PATH + "next_states_continuous.npz")

def visualize_data(n_steps):
    states = np.load(DATA_PATH + "states_continuous.npz", allow_pickle=True)[:n_steps*SQRT_N_THREADS**2:SQRT_N_THREADS**2][:10]
    actions = np.load(DATA_PATH + "actions_continuous.npz", allow_pickle=True)[:n_steps*SQRT_N_THREADS**2:SQRT_N_THREADS**2][:10]
    next_states = np.load(DATA_PATH + "next_states_continuous.npz", allow_pickle=True)[:n_steps*SQRT_N_THREADS**2:SQRT_N_THREADS**2][:10]

    plt.figure()
    plt.plot(states[:, 0], states[:, 1], 'bo')
    plt.plot(next_states[:, 0], next_states[:, 1], 'ro')
    for i in range(len(states)):
        plt.arrow(states[i, 0], states[i, 1], actions[i, 0], actions[i, 1], width=0.1, head_width=1.0)
    plt.show()


if __name__ == '__main__':
    generate_data()
    visualize_data(100)
