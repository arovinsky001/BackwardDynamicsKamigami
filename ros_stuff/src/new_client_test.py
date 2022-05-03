#!/usr/bin/env python
import numpy as np
import rospy
from ros_stuff.srv import CommandAction  # Import service type


def laptop_client():
    # Initialize the client node
    rospy.init_node('laptop_client')

    serv_name = '/kami1/server'
    rospy.wait_for_service(serv_name)
    try:
        # Acquire service proxy
        kami_proxy = rospy.ServiceProxy(
            serv_name, CommandAction)
        rospy.loginfo('Command kami1')
        # Call cmd service via the proxy
        kami_proxy(0.5, 0.5)
    except rospy.ServiceException as e:
        rospy.loginfo(e)


if __name__ == '__main__':
    laptop_client()