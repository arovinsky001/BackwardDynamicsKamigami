import argparse
from pdb import set_trace

import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

# gridworld actions
LEFT = -1
RIGHT = 1
UP = -10
DOWN = 10
possible_actions = np.array([LEFT, RIGHT, UP, DOWN])

def to_tensor(*args):
    rets = []
    for arg in args:
        if type(arg) == np.ndarray:
            rets.append(torch.tensor(arg.astype('float'), requires_grad=True))
        else:
            rets.append(arg)
    return rets

class DynamicsNetwork(nn.Module):
    def __init__(self, state_range, action_range, hidden_dim=512, forward_model=True):
        super(DynamicsNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(state_range.shape[1] + action_range.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_range.shape[1]),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.loss_fn = nn.MSELoss()
        self.forward_model = forward_model

    def forward(self, state, action):
        state, action = to_tensor(state, action)
        if len(state.shape) == 1:
            state = state[:, None]
        if len(action.shape) == 1:
            action = action[:, None]
        state_action = torch.cat([state, action], dim=-1).float()
        # state_action = self.flatten(state_action)
        state_delta = self.model(state_action)
        return state_delta

    def update(self, state, action, next_state):
        state, action, next_state = to_tensor(state, action, next_state)

        input_state = state if self.forward_model else next_state
        target_state = next_state if self.forward_model else state

        state_delta = self(input_state, action)
        loss = self.loss_fn(input_state + state_delta, target_state)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


class ForwardBackwardAgent:
    def __init__(self, state_range, action_range, hidden_dim=512):
        self.forward_model = DynamicsNetwork(state_range, action_range, hidden_dim=hidden_dim, forward_model=True)
        self.backward_model = DynamicsNetwork(state_range, action_range, hidden_dim=hidden_dim, forward_model=False)
        self.state_range = state_range
        self.action_range = action_range
        self.action_dim = action_range.shape[1]
        self.loss_fn = nn.MSELoss(reduction='none')

    def mpc_action(self, state, goal, n_steps=10, n_samples=1000, discrete=False):
        if discrete:
            all_actions = np.eye(self.action_dim)[np.random.choice(self.action_dim, size=(n_steps, n_samples))]
        else:
            scale = self.action_range[1] - self.action_range[0]
            minimum = self.action_range[0]
            all_actions = np.random.rand(n_steps, n_samples, self.action_dim) * scale + minimum
        states = np.tile(state, (n_samples, 1))
        goals = np.tile(goal , (n_samples, 1))
        states, all_actions, goals = to_tensor(states, all_actions, goals)
        all_losses = []

        for i in range(n_steps):
            actions = all_actions[i]
            with torch.no_grad():
                if discrete:
                    states += np.rint(self.forward_model(states, actions))
                else:
                    states += self.forward_model(states, actions)
            states = states % 100
            all_losses.append(self.loss_fn(states, goals).detach().numpy())
        
        best_idx = np.array(all_losses).sum(axis=0).argmin()
        return all_actions[0, best_idx]

    def train(self, states, actions, next_states, model='both'):
        losses = []
        if model == 'forward':
            loss1 = self.forward_model.update(states, actions, next_states).detach().numpy()
            loss2 = None
        elif model == 'backward':
            loss1 = None
            loss2 = self.backward_model.update(states, actions, next_states).detach().numpy()
        else:
            loss1 = self.forward_model.update(states, actions, next_states).detach().numpy()
            loss2 = self.backward_model.update(states, actions, next_states).detach().numpy()
        losses.append(np.array([loss1, loss2]))
        return losses
    
def optimal_grid_policy(state, goal):
    next_state = state + possible_actions % 100
    dist = np.minimum((next_state - goal) % 100, (goal - next_state) % 100)
    return possible_actions[dist.argmin()]
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/load grid agent and do MPC.')
    parser.add_argument('--load_agent_path', type=str, default='agents/grid_agent_1e5_steps_best.pkl',
                        help='path/file to load old agent from')
    parser.add_argument('--save_agent_path', type=str,
                        help='path/file to save newly-trained agent to')
    parser.add_argument('--train_new_agent', action='store_true',
                        help='flag to train new agent')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='hidden layers dimension')
    parser.add_argument('--train_iters', type=int, default=10000,
                        help='number of training iterations for new agent')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for training new agent')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for numpy and pytorch')

    args = parser.parse_args()
    TRAIN_NEW_AGENT = args.train_new_agent
    AGENT_PKL = args.save_agent_path
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if TRAIN_NEW_AGENT:
        states = np.load('grid_data/states.npz', allow_pickle=True)
        actions = np.load('grid_data/actions_one_hot.npz', allow_pickle=True)
        next_states = np.load('grid_data/next_states.npz', allow_pickle=True)

        states_min = 0
        states_max = 99
        state_range = np.block([[states_min], [states_max]])

        actions_min = np.zeros(4)
        actions_max = np.ones(4) * 1.1
        action_range = np.block([[actions_min], [actions_max]])

        agent = ForwardBackwardAgent(state_range, action_range, hidden_dim=args.hidden_dim)

        losses = []
        for _ in trange(args.train_iters):
            data_idx = np.random.choice(len(states), size=args.batch_size, replace=True)
            train_states, train_actions, train_next_states = states[data_idx], actions[data_idx], next_states[data_idx]
            loss = agent.train(train_states[:, None], train_actions, train_next_states[:, None], model='forward')
            losses.append(loss)

        losses = np.array(losses).squeeze()
        plt.plot(np.arange(len(losses)), losses[:, 0])
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
        
        if args.save_agent_path:
            with open(args.save_agent_path, "wb") as f:
                pkl.dump(agent, f)
        else:
            save_string = f"agents/grid_agent"
            save_string += f"_dim{args.hidden_dim}"
            save_string += f"_train{args.train_iters}"
            save_string += f"_batch{args.batch_size}"
            save_string += ".pkl"
            with open(save_string, "wb") as f:
                pkl.dump(agent, f)
    else:
        with open(args.load_agent_path, "rb") as f:
            agent = pkl.load(f)

    # MPC parameters
    n_steps = 4         # sample trajectory length
    n_samples = 100     # number of trajectories to sample

    # repeatedly run trials testing the MPC policy against the optimal policy
    n_trials = 500
    max_steps = 100
    while True:
        optimal_lengths = []
        actual_lengths = []
        states = []
        goals = []
        optimal = 0
        for trial in trange(n_trials):
            init_state = np.random.randint(0, 100)
            states.append(init_state)
            goal = np.random.randint(0, 100)
            goals.append(goal)

            state = init_state
            i = 0
            while state != goal:
                state += optimal_grid_policy(state, goal)
                state = state % 100
                i += 1
            optimal_lengths.append(i)
            
            state = init_state
            j = 0
            while state != goal:
                if j == max_steps:
                    break
                action = np.where(agent.mpc_action(
                    state, goal, n_steps=n_steps, n_samples=n_samples, discrete=True).detach().numpy())[0]
                state += possible_actions[int(action)]
                state = state % 100
                j += 1
            actual_lengths.append(j)

            if j == i:
                optimal += 1
        
        print("\noptimal mean:", np.mean(optimal_lengths))
        print("optimal std:", np.std(optimal_lengths), "\n")
        print("actual mean:", np.mean(actual_lengths))
        print("actual std:", np.std(actual_lengths), "\n")
        print("optimality rate:", optimal / float(n_trials))
        print("timeout rate:", (np.array(actual_lengths) == max_steps).sum() / float(n_trials), "\n")
        set_trace()
