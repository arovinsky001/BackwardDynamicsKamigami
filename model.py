import numpy as np
from pyparsing import Forward

import torch
from torch import nn
from tqdm import trange


class DynamicsNetwork(nn.Module):
    def __init__(self, state_range, action_range, forward=True):
        super(DynamicsNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(len(state_range) + len(action_range), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, len(state_range)),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = self.flatten(state_action)
        next_or_prev_state_dist = self.model(state_action)
        return next_or_prev_state_dist

    def update(self, state, action, next_state):
        if self.forward:
            # TODO this is for if we end up configuring the network to output a distribution
            # dist = self(state, action)
            # loss = -dist.log_prob(next_state).sum()
            pred_next_state = self(state, action)
            loss = self.loss_fn(pred_next_state, next_state)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            pred_state = self(next_state, action)
            loss = self.loss_fn(pred_state, state)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class ForwardBackwardModel:
    def __init__(self, state_range, action_range):
        self.forward_model = DynamicsNetwork(state_range, action_range, forward=True)
        self.backward_model = DynamicsNetwork(state_range, action_range, forward=False)
        self.state_range = state_range
        self.action_range = action_range
        self.action_dim = len(action_range)
        self.loss_fn = nn.MSELoss(reduction='none')

    def mpc_action(self, state, goal, n_steps=10, n_samples=1000):
        scale = self.action_range[0] - self.action_range[1]
        minimum = self.action_range[1]
        all_actions = np.random.rand(n_steps, n_samples, self.action_dim) * scale + minimum
        states = np.tile(state, (n_samples, 1))
        goals = np.tile(goal , (n_samples, 1))
        all_losses = []

        for i in range(n_steps):
            actions = all_actions[i]
            states = self.forward_model(states, actions)
            all_losses.append(self.loss_fn(states, goals))
        
        best_idx = np.array(all_losses).sum(axis=0).argmin()
        return all_actions[0, best_idx]
        

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    data = np.load(...)
    states, actions, next_states = data

    model = ForwardBackwardModel(states)
