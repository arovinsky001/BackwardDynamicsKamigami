Run gridworld agent using this command:
    python model.py --load_agent_path="agents/grid_agent_dim512_train100000_batch256.pkl" --train_iters=100000 --batch_size=256 --hidden_dim=512 --seed=1

Run this to check model predictions:
    agent.forward_model(np.array([[22]]), np.array([[1, 0, 0, 0]]))

    should have [1, 0, 0, 0] -> -1
                [0, 1, 0, 0] -> 1
                [0, 0, 1, 0] -> -10
                [0, 0, 0, 1] -> 10

current performance (# of steps from start to goal, based on 10000 trials):
    optimal mean: 4.9615
    optimal std: 2.0757210193087126 

    actual mean: 6.3936
    actual std: 7.706859220201184 

    optimality rate: 0.6956
    timeout rate: 0.0053
