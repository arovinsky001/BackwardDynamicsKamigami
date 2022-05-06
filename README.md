# Kamigami MPC Dynamics
## Model Training and Simulation
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

## ROS interface
Note that this is still a work in progress and things still need to be stitched together!
### Camera Calibration
The <code>camera_info</code> directory contains <code>head_camera.yaml</code>, which is a file containing camera calibration details for the particular camera we are using (this is necessary for vision-based pose estimation). Refer to this link for a tutorial on how to calibrate the camera https://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration

For us, our square side length was 2.6 cm -> 0.026 m, 6x8 lines for calibration. Try to have good lighting when calibrating, and provide the camera a diverse view of the calibration pattern.

### ROS file system setup
This could be tedious, but it is necessary to make ROS and the packages we're using work together (however it is possible to still run into bugs on some machines).

1. Choose a directory to be your workspace. You can name it anything you want and can put it anywhere you want. This is going to be your workspace. Call this <code>ws</code>.

2. Go inside <code>ws</code> and create a <code>src</code> directory (so that the location is <code>ws/src</code>)

3. Go inside <code>ws/src</code> and install the following packages with the following commands
- git clone https://github.com/ros-drivers/usb_cam.git
- git clone https://github.com/srv/srv_tools.git
    - cd srv_tools/pointcloud_tools 
    - open CMakeLists.txt, remove 'signals' from line 8 (in find_package)
- git clone https://github.com/ros-perception/ar_track_alvar.git
    - cd ar_track_alvar
    - git checkout noetic-devel
        - note: we chose to use noetic on the laptop client, since melodic is not supported on Ubuntu 20.04
- copy the <code>ros_stuff</code> directory into the same directory you installed your other packages (<code>ws/src</code>)

4. Go back to <code>ws</code> and run <code>catkin_make</code>.

5. Run <code>. devel/setup.bash</code>

6. All is set up for operation! Use by running <code>roslaunch ros_stuff webcam_track.launch</code>

### Dev notes
- The rviz config file is located in ros_stuff/rviz/default.rviz; <code>webcam_track.launch</code> currently looks the hard coded location (but looks for the ros_stuff package location dynamically)
- run <code>rostopic echo ar_pose_marker</code> to see ar tag information (e.g. id, pose)
