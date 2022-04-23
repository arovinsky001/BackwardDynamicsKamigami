import argparse
import pickle as pkl
from pdb import set_trace

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
from tqdm import trange

# discrete gridworld actions
LEFT = -1
RIGHT = 1
UP = -10
DOWN = 10
possible_actions = np.array([LEFT, RIGHT, UP, DOWN])

def to_tensor(*args):
    ret = []
    for arg in args:
        if type(arg) == np.ndarray:
            ret.append(tensor(arg.astype('float32'), requires_grad=True))
        else:
            ret.append(arg)
    return ret

class DynamicsNetwork(nn.Module):
    def __init__(self, state_range, action_range, hidden_dim=512, lr=7e-4, delta=False):
        super(DynamicsNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(state_range.shape[1] + action_range.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_range.shape[1]),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.delta = delta
        self.trained = False
        self.input_scaler = None
        self.output_scaler = None

    def forward(self, state, action):
        state, action = to_tensor(state, action)
        if len(state.shape) == 1:
            state = state[:, None]
        if len(action.shape) == 1:
            action = action[:, None]
        state_action = torch.cat([state, action], dim=-1).float()
        # state_action = self.flatten(state_action)
        pred = self.model(state_action)
        return pred

    def update(self, state, action, next_state, retain_graph=False):
        state, action, next_state = to_tensor(state, action, next_state)

        if self.delta:
            state_delta = self(state, action)
            losses = self.loss_fn(state + state_delta, next_state)
        else:
            pred_next_state = self(state, action)
            losses = self.loss_fn(pred_next_state, next_state)
        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        return losses.detach()
    
    def set_scalers(self, states, actions, next_states):
        with torch.no_grad():
            self.input_scaler = StandardScaler().fit(np.append(states, actions, axis=-1))
            self.output_scaler = StandardScaler().fit(next_states)
    
    def get_scaled(self, *args):
        if len(args) == 2:
            states, actions = args
            states_actions = np.append(states, actions, axis=-1)
            states_actions_scaled = self.input_scaler.transform(states_actions)
            states_scaled = states_actions_scaled[:, :states.shape[-1]]
            actions_scaled = states_actions_scaled[:, states.shape[-1]:]
            return states_scaled, actions_scaled
        else:
            next_states = args[0]
            next_states_scaled = self.output_scaler.transform(next_states)
            return next_states_scaled

class MPCAgent:
    def __init__(self, state_range, action_range, discrete=False, delta=False, hidden_dim=512, lr=7e-4):
        self.model = DynamicsNetwork(state_range, action_range, hidden_dim=hidden_dim, lr=lr, delta=delta)
        self.state_range = state_range.astype("int32")
        self.action_range = action_range.astype("int32")
        self.action_dim = action_range.shape[1]
        self.loss_fn = nn.MSELoss(reduction='none')
        self.discrete = discrete
        self.delta = delta

    def mpc_action(self, state, goal, n_steps=10, n_samples=1000):
        if self.discrete:
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
                if self.delta:
                    if self.discrete:
                        states = F.one_hot(states.long(), num_classes=100).squeeze()
                        states += np.where(np.rint(self.model(states, actions)))[1]
                        states = tensor(states.reshape(n_samples, 1))
                    else:
                        states_scaled, actions_scaled = self.model.get_scaled(states, actions)
                        states_delta_scaled = self.model(states_scaled, actions_scaled)
                        states_delta = self.model.output_scaler.inverse_transform(states_delta_scaled)
                        states += states_delta
                else:
                    if self.discrete:
                        states = F.one_hot(states.long(), num_classes=100).squeeze()
                        states = np.where(np.rint(self.model(states, actions)))[1]
                        states = tensor(states.reshape(n_samples, 1))
                    else:
                        states_scaled, actions_scaled = self.model.get_scaled(states, actions)
                        states_scaled = self.model(states_scaled, actions_scaled)
                        states = tensor(self.model.output_scaler.inverse_transform(states_scaled))
            if self.discrete:
                states = states % self.state_range[1]
            else:
                states = np.clip(states, *self.state_range)
            all_losses.append(self.loss_fn(states, goals).detach().numpy().mean(axis=-1))
        
        best_idx = np.array(all_losses).sum(axis=0).argmin()
        return all_actions[0, best_idx]

    def train(self, states, actions, next_states, train_iters=10000, batch_size=256, correction=False, error_weight=4):
        states, actions, next_states = to_tensor(states, actions, next_states)            

        losses = []

        for i in trange(train_iters):
            data_idx = np.random.choice(len(states), size=batch_size, replace=True)
            train_states, train_actions, train_next_states = states[data_idx], actions[data_idx], next_states[data_idx]
            
            loss = self.model.update(train_states, train_actions, train_next_states)
            
            if type(loss) != float:
                while len(loss.shape) > 1:
                    loss = loss.sum(axis=-1)

            if correction:
                loss = self.correct(states, actions, next_states, data_idx, loss,
                                    batch_size=batch_size, error_weight=error_weight)

            loss_mean = loss.mean()
            if i % 100 == 0:
                print(f"mean loss: {loss_mean}")
            losses.append(loss_mean)

        self.model.trained = True
        return losses

    def correct(self, states, actions, next_states, data_idx, loss, batch_size=256, error_weight=4):
        worst_idx = torch.topk(loss.squeeze(), int(batch_size / 10))[1].detach().numpy()
        train_idx = np.append(data_idx, np.tile(data_idx[worst_idx], (1, error_weight)))
        train_states, train_actions, train_next_states = states[train_idx], actions[train_idx], next_states[train_idx]
        loss = self.model.update(train_states, train_actions, train_next_states).detach().numpy()
        
        if type(loss) != float:
            while len(loss.shape) > 1:
                loss = loss.sum(axis=-1)

        return loss
    
def optimal_grid_policy(state, goal):
    next_state = state + possible_actions % 100
    dist = np.minimum((next_state - goal) % 100, (goal - next_state) % 100)
    return possible_actions[dist.argmin()]
    
def optimal_continuous_policy(state, goal):
    vec = goal - state
    return np.clip(vec, -10, 10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/load grid agent and do MPC.')
    parser.add_argument('-load_agent_path', type=str,
                        help='path/file to load old agent from')
    parser.add_argument('-save_agent_path', type=str,
                        help='path/file to save newly-trained agent to')
    parser.add_argument('-new_agent', action='store_true',
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

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.discrete:
        states = np.load('sim/data/discrete/states_one_hot.npz', allow_pickle=True)
        actions = np.load('sim/data/discrete/actions_one_hot.npz', allow_pickle=True)
        next_states = np.load('sim/data/discrete/next_states_one_hot.npz', allow_pickle=True)
        agent_path = 'agents/discrete'
    else:
        states = np.load('sim/data/continuous/states_continuous.npz', allow_pickle=True)
        actions = np.load('sim/data/continuous/actions_continuous.npz', allow_pickle=True)
        next_states = np.load('sim/data/continuous/next_states_continuous.npz', allow_pickle=True)
        agent_path = 'agents/continuous'

    print('\nDATA LOADED\n')

    agent_path += f"_train{args.train_iters}"
    agent_path += f"_dim{args.hidden_dim}"
    agent_path += f"_batch{args.batch_size}"
    agent_path += f"_lr{args.learning_rate}"
    if args.correction:
        agent_path += f"_correction{args.correction_weight}"
    agent_path += ".pkl"

    if args.new_agent:
        if args.discrete:
            states_min = np.zeros(100)
            states_max = np.ones(100) * 1.0
            state_range = np.block([[states_min], [states_max]])

            actions_min = np.zeros(4)
            actions_max = np.ones(4) * 1.0
            action_range = np.block([[actions_min], [actions_max]])
        else:
            states_min = np.zeros(2)
            states_max = np.ones(2) * 100.0
            state_range = np.block([[states_min], [states_max]])

            actions_min = np.ones(2) * -10.0
            actions_max = np.ones(2) * 10.0
            action_range = np.block([[actions_min], [actions_max]])

        agent = MPCAgent(state_range, action_range, discrete=False,
                         delta=False, hidden_dim=args.hidden_dim, lr=args.learning_rate)

        agent.model.set_scalers(states, actions, next_states)
        states_scaled, actions_scaled = agent.model.get_scaled(states, actions)
        next_states_scaled = agent.model.get_scaled(next_states)

        losses = agent.train(states_scaled, actions_scaled, next_states_scaled,
                                train_iters=args.train_iters, batch_size=args.batch_size,
                                correction=args.correction, error_weight=args.correction_weight)

        losses = np.array(losses).squeeze()
        plt.plot(np.arange(len(losses)), losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('(Forward) Dynamics Model Training Loss')
        plt.show()
        
        agent_path = args.save_agent_path if args.save_agent_path else agent_path
        with open(agent_path, "wb") as f:
            pkl.dump(agent, f)
    else:
        agent_path = args.load_agent_path if args.load_agent_path else agent_path
        with open(agent_path, "rb") as f:
            agent = pkl.load(f)

    # MPC parameters
    n_steps = 1         # sample trajectory length
    n_samples = 8000     # number of trajectories to sample

    # repeatedly run trials testing the MPC policy against the optimal policy
    n_trials = 100
    max_steps = 100
    while True:
        optimal_lengths = []
        actual_lengths = []
        optimal = 0
        all_states = []
        all_actions = []
        all_goals = []
        for trial in trange(n_trials):
            init_state = np.random.rand(2) * 100
            goal = np.random.rand(2) * 100

            state = init_state.copy()
            i = 0
            while not np.linalg.norm(goal - state) < 1.0:
                state += optimal_continuous_policy(state, goal)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                state = np.clip(state, 0, 100)
                i += 1
            optimal_lengths.append(i)
            
            state = init_state.copy()
            j = 0
            states, actions = [], []
            while not np.linalg.norm(goal - state) < 1.0:
                states.append(state)
                if j == max_steps:
                    break
                action = agent.mpc_action(
                    state, goal, n_steps=n_steps, n_samples=n_samples).detach().numpy()
                state += action
                state = np.clip(state, 0, 100)
                j += 1
                actions.append(action)

            all_states.append(states)
            all_actions.append(actions)
            all_goals.append(goal)
            actual_lengths.append(j)

            if j == i:
                optimal += 1

        print("\noptimal mean:", np.mean(optimal_lengths))
        print("optimal std:", np.std(optimal_lengths), "\n")
        print("actual mean:", np.mean(actual_lengths))
        print("actual std:", np.std(actual_lengths), "\n")
        print("mean error:", np.abs(np.mean(optimal_lengths) - np.mean(actual_lengths)) / np.mean(optimal_lengths))
        print("optimality rate:", optimal / float(n_trials))
        print("timeout rate:", (np.array(actual_lengths) == max_steps).sum() / float(n_trials), "\n")
        set_trace()
