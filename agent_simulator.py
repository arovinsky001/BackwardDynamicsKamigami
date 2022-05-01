import argparse
import pickle as pkl
from tkinter.tix import MAX

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import animation

from forward_mpc_agent import *
from sim.params.continuous_sim_params import *

MODES = ['follow', 'cluster']

class MPCSim:
    def __init__(self, agent_path, fig, mode, swarm_weight=10., tol=2., n_agents=3):
        assert mode in MODES
        self.fig = fig
        self.mode = mode
        self.ani = None
        self.states = None
        self.starts, self.goal = None, None
        self.noises = None
        self.swarm_weight = swarm_weight
        self.dones = np.full(n_agents, False)
        self.tol = tol
        self.state_range = np.array([[MIN_STATE]*2, [MAX_STATE]*2])
        self.action_range = np.array([[-1]*2, [1]*2]) * MAGNITUDE
        self.colors = ['b', 'm', 'k', 'c', 'y']
        self.agents = []
        for _ in range(n_agents):
            with open(agent_path, "rb") as f:
                self.agents.append(pkl.load(f))
        for i, agent in enumerate(self.agents):
            for j, neighbor in enumerate(self.agents):
                if i != j:
                    agent.neighbors.append(neighbor)
    
    def sim_and_animate(self, step):
        distances = np.linalg.norm(self.goal - self.states, axis=-1)
        self.dones[distances < self.tol] = True

        if np.all(self.dones):
            self.ani = None
            return
        
        if self.mode == 'follow':
            leader_idx = distances.argmin()
            leader_state = self.states[leader_idx]

        for i, agent in enumerate(self.agents):
            if not self.dones[i]:
                if self.mode == 'follow':
                    goal = self.goal if i == leader_idx else leader_state
                    action = agent.mpc_action(self.states[i], goal, self.state_range,
                                            self.action_range, n_steps=self.mpc_steps,
                                            n_samples=self.mpc_samples, swarm=False).detach().numpy()
                elif self.mode == 'cluster':
                    action = agent.mpc_action(self.states[i], self.goal, self.state_range,
                                            self.action_range, n_steps=self.mpc_steps,
                                            n_samples=self.mpc_samples, swarm=True, swarm_weight=self.swarm_weight).detach().numpy()
                noise = self.noises[i, step]
                self.states[i] += (action + noise)

        self.states = np.clip(self.states, MIN_STATE, MAX_STATE)
        
        plt.clf()
        plt.xlim(MIN_STATE, MAX_STATE)
        plt.ylim(MIN_STATE, MAX_STATE)
        plt.grid()
        plt.plot(self.starts[:, 0], self.starts[:, 1], color='r', linestyle='None', marker='>', markersize=7)
        plt.plot(self.goal[0], self.goal[1], color='g', linestyle='None', marker='<', markersize=7)
        plt.plot(self.states[:, 0], self.states[:, 1], color=self.colors[i], linestyle='None', marker='*', markersize=7)
        plt.legend(['Starting State', 'Goal State', 'Current State'])

    def run(self, starts, goal, n_agents, n_steps, mpc_steps, mpc_samples, noise_std, interval):
        self.starts, self.goal = starts, goal
        self.states = self.starts.copy()
        for i, agent in enumerate(self.agents):
            agent.state = torch.tensor(starts[i])
        self.mpc_steps, self.mpc_samples = mpc_steps, mpc_samples
        self.noises = np.random.normal(loc=0.0, scale=noise_std, size=(n_agents, n_steps, 2))
        self.ani = animation.FuncAnimation(self.fig, self.sim_and_animate, frames=n_steps, interval=interval, repeat=False)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/load agent and do MPC.')
    parser.add_argument('-load_agent_path', type=str,
                        help='path/file to load old agent from')
    parser.add_argument('-save_agent_path', type=str,
                        help='path/file to save newly-trained agent to')
    parser.add_argument('-new_agent', '-n', action='store_true',
                        help='flag to train new agent')
    parser.add_argument('-hidden_dim', type=int, default=512,
                        help='hidden layers dimension')
    parser.add_argument('-train_iters', type=int, default=10000,
                        help='number of training iterations for new agent')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='batch size for training new agent')
    parser.add_argument('-learning_rate', type=float, default=7e-4,
                        help='batch size for training new agent')
    parser.add_argument('-seed', type=int, default=1,
                        help='random seed for numpy and pytorch')
    parser.add_argument('-correction', action='store_true',
                        help='flag to retrain on mistakes during training')
    parser.add_argument('-correction_weight', type=int, default=4,
                        help='number of times to retrain on mistaken data')
    parser.add_argument('-discrete', action='store_true',
                        help='whether or not to train discrete model (on discrete data)')
    parser.add_argument('-stochastic', action='store_true',
                        help='whether or not to use stochastic transition data')
    parser.add_argument('-distribution', action='store_true',
                        help='whether to have the model output a distribution or a direct prediction')
    parser.add_argument('-swarm_mode', default='cluster',
                        help="specify swarm mode: 'follow' (the leader) or 'cluster'")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent_path = 'agents/discrete' if args.discrete else 'agents/continuous'

    agent_path += f"_train{args.train_iters}"
    agent_path += f"_dim{args.hidden_dim}"
    agent_path += f"_batch{args.batch_size}"
    agent_path += f"_lr{args.learning_rate}"
    if args.distribution:
        agent_path += "_distribution"
    if args.stochastic:
        agent_path += "_stochastic"
    if args.correction:
        agent_path += f"_correction{args.correction_weight}"
    agent_path += ".pkl"
    
    fig = plt.figure()
    n_agents = 3
    swarm_weight = 0.3
    tolerance = 2.
    sim = MPCSim(agent_path, fig, args.swarm_mode, swarm_weight=swarm_weight,
                 tol=tolerance, n_agents=n_agents)
    # starts = np.random.rand(n_agents, 2) * 100
    starts = np.array([[10., 10], [10, 40], [40, 10]])
    goal = np.array([60., 80])
    # goal = np.random.rand(2) * 100
    n_steps = 200
    mpc_steps = 1
    mpc_samples = 10000
    noise_std = 0.5
    interval = 75
    sim.run(starts, goal, n_agents, n_steps, mpc_steps, mpc_samples, noise_std, interval)
