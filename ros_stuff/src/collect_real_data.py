#!/usr/bin/python3

import numpy as np
from ar_track_alvar_msgs.msg import AlvarMarkers
from ros_stuff.srv import CommandAction
from ros_stuff.msg import RobotCmd
from tf.transformations import euler_from_quaternion

import rospy

AVG_STEPS = 5

class DataCollector:
    def __init__(self):
        print("started")
        self.min_action = 0.3
        self.max_action = 0.7
        self.current_state = np.zeros(3)        # (x, y, theta)
        self.base_state = np.zeros(3)
        self.robot_id = 0
        self.base_id = 1
        self.stamped_states = []
        self.stamped_actions = []
        rospy.init_node("data_collector")
        rospy.wait_for_service('/kami1/server')
        self.command_action = rospy.ServiceProxy('/kami1/server', CommandAction)
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.collect_data, queue_size=10)
        rospy.spin()

    def collect_data(self, msg):
        try:
            for marker in msg.markers:
                if marker.id == self.robot_id:
                    state = self.current_state
                elif marker.id == self.base_id:
                    state = self.base_state
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
            print("could not update states")
            import pdb;pdb.set_trace()
        
        current_state = self.current_state - self.base_state
        current_state[2] %= 2 * np.pi
        time_state = np.append(rospy.get_rostime().to_sec(), self.current_state)
        self.stamped_states.append(time_state)
        action = np.random.uniform(low=self.min_action, high=self.max_action, size=2)
        rand_neg = (np.random.rand(2) - 0.5) > 0
        if len(rand_neg > 0):
            action[rand_neg] *= -1
        print(action)
        action_req = RobotCmd()
        action_req.left_pwm = action[0]
        action_req.right_pwm = action[1]
        timestamp = self.command_action(action_req, 'kami1')
        print("sent action")
        time_action = np.append(timestamp.time.to_sec(), action)
        # time_action = np.append(1000, action)
        self.stamped_actions.append(time_action)
    
    def process_raw_data(self):
        state_times = self.stamped_states[:, 0]
        action_times = self.stamped_actions[:, 0]

        states = self.stamped_states[:, 1:]
        actions = self.stamped_actions[:, 1:]

        all_states = []
        all_actions = []
        all_next_states = []

        for i, action in enumerate(actions):
            t_action = action_times[i] 
            idx = np.where(state_times < t_action)[0].max()
            if idx < AVG_STEPS - 1 or len(states) - idx < AVG_STEPS + 1:
                continue
            current_state = states[idx-AVG_STEPS+1:idx+1].mean()
            next_state = states[idx+1:idx+1+AVG_STEPS].mean()
            all_states.append(current_state)
            all_actions.append(action)
            print("CUR STATE: ", current_state)
            all_next_states.append(next_state)
        
        np.savez_compressed("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/real_data.npz",
                            states=all_states, actions=all_actions, next_states=all_next_states)

            
if __name__ == '__main__':
    dc = DataCollector()
    
    try:
        print("started collecting")
        dc.process_raw_data()
    except:
        print("could not process raw data")
        import pdb;pdb.set_trace()
