#!/usr/bin/env python
import numpy as np
import rospy
from ros_stuff.srv import CommandAction  # Import service type


def laptop_client():
    # Initialize the client node
    rospy.init_node('laptop_client')

    # Get info on positioning from camera & AR tags
    rospy.Subscriber("visualization_marker", Marker, self.update_cube_pose)
    rospy.Subscriber("usb_cam/image_raw", Image, self.write_to_bag)
    # Return dictionary with kaminame keys and values corresponding to pwm commands
    kami_dict = ...
    # Need to actually write this part later

    for kami in kami_dict.keys():
        # Wait until kami service is ready
        serv_name = '/{}/server'.format(kami)
        rospy.wait_for_service(serv_name)
        try:
            # Acquire service proxy
            kami_proxy = rospy.ServiceProxy(
                serv_name, CommandAction)
            rospy.loginfo('Command {} to pwm values: '.format(kami) + str(kami_dict[kami]))
            # Call patrol service via the proxy
            kami_proxy(kami_dict[kami][0], kami_dict[kami][1], kami)
        except rospy.ServiceException as e:
            rospy.loginfo(e)


if __name__ == '__main__':
    laptop_client()