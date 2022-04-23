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
            rets.append(torch.tensor(arg.astype('float32'), requires_grad=True))
        else:
            rets.append(arg)
    return rets

class DynamicsNetwork(nn.Module):
    def __init__(self, state_range, action_range, hidden_dim=512, delta=False, forward_model=True):
        super(DynamicsNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(state_range.shape[1] + action_range.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_range.shape[1]),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0007)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.forward_model = forward_model
        self.delta = delta

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

        input_state = state if self.forward_model else next_state
        target_state = next_state if self.forward_model else state

        if self.delta:
            state_delta = self(input_state, action)
            losses = self.loss_fn(input_state + state_delta, target_state)
        else:
            pred_next_state = self(input_state, action)
            losses = self.loss_fn(pred_next_state, target_state)
        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        return losses


class ForwardBackwardAgent:
    def __init__(self, state_range, action_range, delta=False, hidden_dim=512):
        self.forward_model = DynamicsNetwork(state_range, action_range, hidden_dim=hidden_dim, delta=delta, forward_model=True)
        self.backward_model = DynamicsNetwork(state_range, action_range, hidden_dim=hidden_dim, delta=delta, forward_model=False)
        self.state_range = state_range.astype("int32")
        self.action_range = action_range.astype("int32")
        self.action_dim = action_range.shape[1]
        self.loss_fn = nn.MSELoss(reduction='none')
        self.delta = delta

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
                if self.delta:
                    if discrete:
                        states = F.one_hot(states.long(), num_classes=100).squeeze()
                        states += np.where(np.rint(self.forward_model(states, actions)))[1]
                        states = torch.tensor(states.reshape(n_samples, 1))
                    else:
                        states += self.forward_model(states, actions)
                else:
                    if discrete:
                        states = F.one_hot(states.long(), num_classes=100).squeeze()
                        states = np.where(np.rint(self.forward_model(states, actions)))[1]
                        states = torch.tensor(states.reshape(n_samples, 1))
                    else:
                        states = self.forward_model(states, actions)
            if discrete:
                states = states % 100
            all_losses.append(self.loss_fn(states, goals).detach().numpy().mean(axis=-1))
        
        best_idx = np.array(all_losses).sum(axis=0).argmin()
        return all_actions[0, best_idx]

    def mpc_backward_action(self, state, goal, n_steps=10, n_samples=1000, discrete=False, backward_weight=0.3):
        if discrete:
            all_actions = np.eye(self.action_dim)[np.random.choice(self.action_dim, size=(n_steps, n_samples))]
        else:
            scale = self.action_range[1] - self.action_range[0]
            minimum = self.action_range[0]
            all_actions = np.random.rand(n_steps, n_samples, self.action_dim) * scale + minimum
        states = torch.tile(torch.tensor(state), (n_samples, 1)).int()
        init_states = states.clone()
        goals = np.tile(goal , (n_samples, 1))
        states, all_actions, goals = to_tensor(states, all_actions, goals)
        all_losses = []

        for i in range(n_steps):
            actions = all_actions[i]
            with torch.no_grad():
                if discrete:
                    states += self.forward_model(init_states, actions).round().int()
                    states = states % self.state_range[1]
                else:
                    states += self.forward_model(states, actions)
            standard_loss = self.loss_fn(states, goals).detach().numpy()
            all_losses.append(standard_loss)
        
        for i in range(n_steps):
            actions = all_actions[n_steps - 1 - i]
            with torch.no_grad():
                if discrete:
                    states += self.backward_model(states, actions).round().int()
                    states = states % self.state_range[1]
                else:
                    states += self.backward_model(states, actions)

        confidence_losses = np.array(self.loss_fn(states.float(), init_states.float()))
        standard_losses = np.array(all_losses).sum(axis=0)
        best_idx = np.argmin(standard_losses + confidence_losses * backward_weight)
        return all_actions[0, best_idx]

    def train(self, states, actions, next_states, train_iters=10000, batch_size=256, correction=False, error_weight=4, model='both'):
        states, actions, next_states = to_tensor(states, actions, next_states)
        losses = []

        for i in trange(train_iters):
            data_idx = np.random.choice(len(states), size=batch_size, replace=True)
            train_states, train_actions, train_next_states = states[data_idx], actions[data_idx], next_states[data_idx]
            
            if model == 'forward':
                loss1 = self.forward_model.update(train_states, train_actions, train_next_states)
                loss2 = 0.0
            elif model == 'backward':
                loss1 = 0.0
                loss2 = self.backward_model.update(train_states, train_actions, train_next_states)
            else:
                data_idx_back = np.random.choice(len(states), size=batch_size, replace=True)
                train_states_back = states[data_idx_back]
                train_actions_back = actions[data_idx_back]
                train_next_states_back = next_states[data_idx_back]
                loss1 = self.forward_model.update(train_states, train_actions, train_next_states)
                loss2 = self.backward_model.update(train_states_back, train_actions_back, train_next_states_back)
            
            if type(loss1) != float:
                while len(loss1.shape) > 1:
                    loss1 = loss1.sum(axis=-1)
            
            if type(loss2) != float:
                while len(loss2.shape) > 1:
                    loss2 = loss1.sum(axis=-1)

            if correction:
                if model == 'forward':
                    worst_idx = torch.topk(loss1.squeeze(), int(batch_size / 10))[1].detach().numpy()
                    train_idx = np.append(data_idx, np.tile(data_idx[worst_idx], (1, error_weight)))
                    train_states, train_actions, train_next_states = states[train_idx], actions[train_idx], next_states[train_idx]
                    loss1 = self.forward_model.update(train_states, train_actions, train_next_states).detach().numpy()
                elif model == 'backward':
                    worst_idx = torch.topk(loss2.squeeze(), int(batch_size / 10))[1].detach().numpy()
                    train_idx = np.append(data_idx, np.tile(data_idx[worst_idx], (1, error_weight)))
                    train_states, train_actions, train_next_states = states[train_idx], actions[train_idx], next_states[train_idx]
                    loss2 = self.backward_model.update(train_states, train_actions, train_next_states).detach().numpy()
                else:
                    worst_idx1 = torch.topk(loss1.squeeze(), int(batch_size / 10))[1].detach().numpy()
                    worst_idx2 = torch.topk(loss2.squeeze(), int(batch_size / 10))[1].detach().numpy()
                    train_idx1 = np.append(data_idx, np.tile(data_idx[worst_idx1], (1, error_weight)))
                    train_idx2 = np.append(data_idx, np.tile(data_idx[worst_idx2], (1, error_weight)))
                    train_states1, train_actions1, train_next_states1 = states[train_idx1], actions[train_idx1], next_states[train_idx1]
                    train_states2, train_actions2, train_next_states2 = states[train_idx2], actions[train_idx2], next_states[train_idx2]
                    loss1 = self.forward_model.update(train_states1, train_actions1, train_next_states1).detach().numpy()
                    loss2 = self.backward_model.update(train_states2, train_actions2, train_next_states2).detach().numpy()

                if type(loss1) != float:
                    while len(loss1.shape) > 1:
                        loss1 = loss1.sum(axis=-1)
                
                if type(loss2) != float:
                    while len(loss2.shape) > 1:
                        loss2 = loss1.sum(axis=-1)

            loss_mean1, loss_mean2 = np.mean(loss1), np.mean(loss2)
            if i % 100 == 0:
                print(f"mean forward loss: {loss_mean1} || mean backward loss: {loss_mean2}")
            losses.append(np.array([loss_mean1, loss_mean2]))

        return losses

    def train_disagree(self, states, actions, next_states, train_iters=1000, batch_size=256):
        states, actions, next_states = to_tensor(states, actions, next_states)
        losses = []

        for _ in trange(train_iters):
            data_idx = np.random.choice(len(states), size=batch_size, replace=True)
            cur_states, cur_actions = states[data_idx], actions[data_idx]
            pred_next_states = cur_states + self.forward_model(cur_states, cur_actions).round().int() % 100
            pred_cur_states = pred_next_states + self.backward_model(pred_next_states, cur_actions).round().int() % 100
            loss = self.loss_fn(pred_cur_states.float(), cur_states.float())
            worst_idx = torch.topk(loss.squeeze(), 16)[1].detach().numpy()
            train_idx = np.append(data_idx, np.tile(data_idx[worst_idx], (1, 4)))
            train_states, train_actions, train_next_states = states[train_idx], actions[train_idx], next_states[train_idx]
            loss1 = self.forward_model.update(train_states, train_actions, train_next_states, retain_graph=True).detach().numpy()
            loss2 = self.backward_model.update(train_states, train_actions, train_next_states).detach().numpy()
            losses.append(np.array([np.mean(loss1), np.mean(loss2)]))

        return losses
    
def optimal_grid_policy(state, goal):
    next_state = state + possible_actions % 100
    dist = np.minimum((next_state - goal) % 100, (goal - next_state) % 100)
    return possible_actions[dist.argmin()]
    
def optimal_continuous_policy(state, goal):
    vec = goal - state
    norm = np.linalg.norm(vec)
    if norm <= 10:
        return vec
    vec /= norm
    vec *= 10
    return vec


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
    parser.add_argument('--backward', action='store_true',
                        help='flag to train backward model')
    parser.add_argument('--correction', action='store_true',
                        help='flag to retrain on mistakes during training')
    parser.add_argument('--correction_weight', type=int, default=4,
                        help='number of times to retrain on mistaken data')

    args = parser.parse_args()
    TRAIN_NEW_AGENT = args.train_new_agent
    AGENT_PKL = args.save_agent_path
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # states = np.load('grid_data/states_one_hot.npz', allow_pickle=True)
    # actions = np.load('grid_data/actions_one_hot.npz', allow_pickle=True)
    # next_states = np.load('grid_data/next_states_one_hot.npz', allow_pickle=True)

    states = np.load('grid_data/states_continuous.npz', allow_pickle=True)
    actions = np.load('grid_data/actions_continuous.npz', allow_pickle=True)
    next_states = np.load('grid_data/next_states_continuous.npz', allow_pickle=True)

    print('Data Loaded')

    if TRAIN_NEW_AGENT:
        # states_min = np.zeros(100)
        # states_max = np.ones(100) * 1.1
        # state_range = np.block([[states_min], [states_max]])

        # actions_min = np.zeros(4)
        # actions_max = np.ones(4) * 1.1
        # action_range = np.block([[actions_min], [actions_max]])

        states_min = np.zeros(2)
        states_max = np.ones(2) * 100.0
        state_range = np.block([[states_min], [states_max]])

        actions_min = np.ones(2) * -10.0
        actions_max = np.ones(2) * 10.0
        action_range = np.block([[actions_min], [actions_max]])

        agent = ForwardBackwardAgent(state_range, action_range, delta=False, hidden_dim=args.hidden_dim)
        if args.backward:
            losses = agent.train(states, actions, next_states,
                                 train_iters=args.train_iters, batch_size=args.batch_size,
                                 correction=args.correction, error_weight=args.correction_weight,
                                 model='both')
        else:
            losses = agent.train(states, actions, next_states,
                                 train_iters=args.train_iters, batch_size=args.batch_size,
                                 correction=args.correction, error_weight=args.correction_weight,
                                 model='forward')

        losses = np.array(losses).squeeze()
        plt.plot(np.arange(len(losses)), losses[:, 0], label='Forward Model')
        plt.plot(np.arange(len(losses)), losses[:, 1], label='Backward Model')
        plt.legend()
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
    n_steps = 2         # sample trajectory length
    n_samples = 100     # number of trajectories to sample
    backward_weight = 0.0

    # repeatedly run trials testing the MPC policy against the optimal policy
    n_trials = 100
    max_steps = 100
    while True:
        optimal_lengths = []
        actual_lengths = []
        optimal = 0
        for trial in trange(n_trials):
            init_state = np.random.rand(2) * 100
            goal = np.random.rand(2) * 100

            state = init_state
            i = 0
            while not np.linalg.norm(goal - state) < 1.0:
                state += optimal_continuous_policy(state, goal)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                state = state % 100
                i += 1
            optimal_lengths.append(i)
            
            state = init_state
            j = 0
            while not np.linalg.norm(goal - state) < 1.0:
                if j == max_steps:
                    break
                if args.backward:
                    action = agent.mpc_backward_action(
                        state, goal, n_steps=n_steps, n_samples=n_samples, discrete=False, backward_weight=backward_weight).detach().numpy()
                else:
                    action = agent.mpc_action(
                        state, goal, n_steps=n_steps, n_samples=n_samples, discrete=False).detach().numpy()
                state += action
                state = state % 100
                j += 1
            actual_lengths.append(j)

            if j == i:
                optimal += 1
        
        # losses = agent.train_disagree(states[:, None], actions, next_states[:, None], train_iters=args.train_iters, batch_size=args.batch_size)
        # avg_disagree_loss = np.mean(losses, axis=0)
        
        print("\noptimal mean:", np.mean(optimal_lengths))
        print("optimal std:", np.std(optimal_lengths), "\n")
        print("actual mean:", np.mean(actual_lengths))
        print("actual std:", np.std(actual_lengths), "\n")
        print("mean error:", np.abs(np.mean(optimal_lengths) - np.mean(actual_lengths)) / np.mean(optimal_lengths))
        print("optimality rate:", optimal / float(n_trials))
        print("timeout rate:", (np.array(actual_lengths) == max_steps).sum() / float(n_trials), "\n")
        # print("correction training mean loss:", avg_disagree_loss)
        set_trace()
