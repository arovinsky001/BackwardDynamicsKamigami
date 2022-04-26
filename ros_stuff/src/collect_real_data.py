import numpy as np
from visualization_msgs.msg import Marker
from robot_server.srv import CommandAction
import rospy

MAGNITUDE = 20.0
SQRT_N_THREADS = 100
N_ITERS = 1000
DATA_PATH = "/Users/Obsidian/Desktop/eecs106b/projects/BackwardDynamicsKamigami/sim/data/continuous/"
TAG_ID = 0
AVG_STEPS = 5

class DataCollector:
    def __init__(self):
        self.current_state = np.zeros(3)        # (x, y, theta)
        rospy.Subscriber("visualization_marker", Marker, self.update_state)
        rospy.wait_for_service("robot_server")
        self.command_action = rospy.ServiceProxy("robot_server", CommandAction)
        self.min_action = np.zeros(2)
        self.max_action = np.ones(2) * 100

    def update_state(self, msg):
        assert msg.id == TAG_ID
        self.current_state[0] = msg.pose.position.x
        self.current_state[1] = msg.pose.position.y
        self.current_state[2] = 2 * np.arcsin(msg.pose.orientation.z)

    def collect_data(self):
        done = False
        stamped_states = []
        stamped_actions = []

        rospy.wait_for_message("visualization_marker", Marker)

        while not done:
            current_state = self.current_state
            time_state = np.append(rospy.get_rostime(), self.current_state)
            stamped_states.append(time_state)
            action = np.random.rand(2) * (self.max_action - self.min_action) + self.min_action
            timestamp = self.command_action(action)
            time_action = np.append(timestamp, action)
            stamped_actions.append(time_action)
            while current_state == self.current_state:
                print("waiting for state to update")
                rospy.sleep(0.001)
            time_state = np.append(rospy.get_rostime(), self.current_state)
            stamped_states.append(time_state)
        
        np.array(stamped_states).dump("stamped_states.npy")
        np.array(stamped_actions).dump("stamped_actions.npy")
    
    def process_raw_data(self, stamped_states, stamped_actions):
        state_times = stamped_states[:, 0]
        action_times = stamped_actions[:, 0]

        states = stamped_states[:, 1:]
        actions = stamped_actions[:, 1:]

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
            all_next_states.append(next_state)

        return all_states, all_actions, all_next_states

            
if __name__ == '__main__':
    stamped_states = np.load("stamped_states.npy")
    stamped_actions = np.load("stamped_actions.npy")

    