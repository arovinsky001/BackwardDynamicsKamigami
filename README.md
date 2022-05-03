Run pretrained continuous agent with forward dynamics using this command:

    python forward_mpc_agent.py -train_iters=3000 -batch_size=512 -hidden_dim=128 -learning_rate=0.0007 -seed=1 -distribution -stochastic

Refer to forward_mpc_agent.py for commandline arguments (check argparse section)

---

Generate simulated training data using this command:

    python -m sim.scripts.continuous_sim

Change simulation parameters in sim/scripts/generate_data.py

---

Run animated simulator in cluster mode using this command:

    python agent_simulator.py -train_iters=3000 -batch_size=512 -hidden_dim=128 -learning_rate=0.0007 -distribution -stochastic -seed=206 -swarm_mode=cluster

Can also use -swarm_mode=follow for follow-the-leader mode

Change simulation parameters in agent_simulator.py
