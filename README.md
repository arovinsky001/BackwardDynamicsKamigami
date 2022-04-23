Run pretrained continuous agent with forward dynamics using this command:

    python forward_mpc_agent.py -train_iters=10000 -batch_size=512 -hidden_dim=128 -learning_rate=0.0007 -seed=1

current performance (# of steps from start to goal, based on 10000 trials):

    for n_samples=1000:

        optimal mean: 5.0732
        optimal std: 2.2316903369419334 

        actual mean: 5.1667
        actual std: 2.2619706253618768 

        mean error: 0.018430182133564555
        optimality rate: 0.9066
        timeout rate: 0.0

    
    for n_samples=10000:

        optimal mean: 5.0754
        optimal std: 2.2402934718469365 

        actual mean: 5.1065
        actual std: 2.2506349659596068 

        mean error: 0.006127595854513824
        optimality rate: 0.9689
        timeout rate: 0.0
