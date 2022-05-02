#from geometry_msgs.msg import Twist
from ros_stuff.msg import RobotCmd
import numpy as np
import rospy
#from std_srvs.srv import Empty
from ros_stuff.srv import CommandAction  # Service type
from ros_stuff.srv import RobotStateInfo
#from turtlesim.srv import TeleportAbsolute
import sys


def laptop_callback(request):
    #Getting pwm values from Kamigami
    #x, y, theta, name = request.x, request.y, request.theta, request.name
    left_pwm, right_pwm, name = request.left, request.right, request.name

    #Some service stuff for listening to robot
    rospy.wait_for_service('RobotStateInfo')
    curr_state = rospy.ServiceProxy('RobotStateInfo', some_func)
    try:
        resp1 = some_func(x, y)
    except rospy.ServiceException as exc:
        print("Service did not process request: " + str(exc))

    # rospy.wait_for_service('clear')
    # rospy.wait_for_service('/{}/teleport_absolute'.format(name))
    # clear_proxy = rospy.ServiceProxy('clear', Empty)
    # teleport_proxy = rospy.ServiceProxy(
    #     '/{}/teleport_absolute'.format(name),
    #     TeleportAbsolute
    # )
    # vel = request.vel  # Linear velocity
    # omega = request.omega  # Angular velocity

    # #send movement commands
    # pub = rospy.Publisher(
    #     '/{}/cmd_vel'.format(name), RobotCmd, queue_size=50)
    # cmd = Twist()
    # cmd.linear.x = vel
    # cmd.angular.z = omega

    # Publish to cmd_vel at 5 Hz
    rate = rospy.Rate(5)

    # Clear historical path traces
    clear_proxy()
    while not rospy.is_shutdown():
        pub.publish(cmd)  # Publish to cmd_vel
        rate.sleep()  # Sleep until 
    return cmd  # This line will never be reached

def laptop_server(name):
    # Initialize the server node for kamigami1
    rospy.init_node('{}_laptop_server'.format(name))
    # Register service
    rospy.Service(
        '/{}/laptop'.format(name),  # Service name
        CommandAction,  # Service type
        laptop_callback  # Service callback
    )
    rospy.loginfo('Running laptop server...')
    rospy.spin() # Spin the node until Ctrl-C


if __name__ == '__main__':
    laptop_server(sys.argv[1])