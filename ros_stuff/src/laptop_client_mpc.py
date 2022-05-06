#!/usr/bin/env python
import numpy as np
import pickle as pkl

import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from ros_stuff.srv import CommandAction
from ros_stuff.msg import RobotCmd
from tf.transformations import euler_from_quaternion



class RealMPC:
    def __init__(self, kami_ids, agent_path, goal, mpc_steps, mpc_samples):
        self.kami_ids = np.array(kami_ids)
        self.base_id = 1
        self.base_state = np.zeros(3)
        self.agents = []
        self.goal = goal
        self.tol = 0.05
        self.dones = [False] * len(kami_ids)
        self.states = np.zeros((len(kami_ids), 3))
        self.state_range = np.array([[-np.inf], [np.inf]])
        self.action_range = np.array([[-0.7], [0.7]])
        self.mpc_steps = mpc_steps
        self.mpc_samples = mpc_samples
        for _ in self.kami_ids:
            with open(agent_path, "rb") as f:
                self.agents.append(pkl.load(f))
        rospy.init_node('laptop_client')

        # Get info on positioning from camera & AR tags
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.callback)
        rospy.wait_for_service('/kami1/server')
        self.command_action = rospy.ServiceProxy('/kami1/server', CommandAction)

        rospy.spin()

    def callback(self, msg):
        try:
            for marker in msg.markers:
                if marker.id == self.base_id:
                    state = self.base_state
                elif marker.id in self.kami_ids:
                    if self.dones[self.kami_ids == marker.id]:
                        continue
                    state = self.states[self.kami_ids == marker.id]
                else:
                    print("\nSHOULD NOT HAVE BEEN REACHED\n")
                    raise ValueError
                
                state[0] = marker.pose.pose.position.x
                state[1] = marker.pose.pose.position.y

                o = marker.pose.pose.orientation
                o_list = [o.x, o.y, o.z, o.w]
                _, _, z = euler_from_quaternion(o_list)
                state[2] = z % (2 * np.pi) * 180 / np.pi
        except:
            print(f"could not update states, id: {marker.id}")
            import pdb;pdb.set_trace()

        diff = self.goal - self.states
        dists_xy = np.linalg.norm(diff[:, :-1], axis=-1)
        dists_theta = np.abs(diff[:, -1])
        distances = dists_xy + 0.01 * dists_theta
        self.dones[distances < self.tol] = True
        
        if np.all(self.dones):
            rospy.signal_shutdown("Finished! All robots reached goal.")

        for i, agent in enumerate(self.agents):
            if not self.dones[i]:
                action = agent.mpc_action(self.states[i], self.goal, self.state_range,
                                        self.action_range, n_steps=self.mpc_steps,
                                        n_samples=self.mpc_samples, swarm=False, swarm_weight=0.3).detach().numpy()

                action_req = RobotCmd()
                action_req.left_pwm = action[0]
                action_req.right_pwm = action[1]
                self.command_action(action_req, f'kami{self.kami_ids[i]}')


if __name__ == '__main__':
    kami_ids = [0, 1]
    agent_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/agents/real.pkl"
    goal = np.array([0.15, 0.15, 0.0])
    mpc_steps = 2
    mpc_samples = 1000
    r = RealMPC(kami_ids, agent_path, goal, mpc_steps, mpc_samples)
