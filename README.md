Run pretrained continuous agent with forward dynamics using this command:

    python forward_mpc_agent.py -train_iters=3000 -batch_size=512 -hidden_dim=128 -learning_rate=0.0007 -seed=1 -distribution -stochastic

Refer to forward_mpc_agent.py for commandline arguments (check argparse section)

---

Generate simulated training data using this command:

    python -m sim.scripts.continuous_sim

Change simulation parameters in sim/params/*

---

Run animated simulator in cluster mode using this command:

    python agent_simulator.py -train_iters=3000 -batch_size=512 -hidden_dim=128 -learning_rate=0.0007 -distribution -stochastic -seed=206 -swarm_mode=cluster

Can also use -swarm_mode=follow for follow-the-leader mode

Change simulation parameters in agent_simulator.py

## Kamigami-ROS interface
### technical notes
In the CMakeLists.txt, make the following changes
- uncomment / edit this section to look like
<code>
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)
</code>
- uncomment / edit this section to look like
<code>
add_message_files(
   FILES
   RobotCmd.msg
)
</code>
- uncomment / edit this section to look like
<code>
add_service_files(
   FILES
   CommandAction.srv
)
</code>
- uncomment / edit this section to look like
<code>
generate_messages(
   DEPENDENCIES
   std_msgs
)
</code>
