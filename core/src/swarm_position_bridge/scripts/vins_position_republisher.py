#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry


class VinsPositionRepublisher:
    def __init__(self):
        input_topic = rospy.get_param("~input_topic", "vins_fusion/imu_propagate")
        output_topic = rospy.get_param("~output_topic", "vins_position")

        self.pub = rospy.Publisher(output_topic, PointStamped, queue_size=10)
        self.sub = rospy.Subscriber(
            input_topic, Odometry, self.odom_callback, queue_size=10
        )

        rospy.loginfo("VINS position republisher started")
        rospy.loginfo("Input topic: %s", rospy.resolve_name(input_topic))
        rospy.loginfo("Output topic: %s", rospy.resolve_name(output_topic))

    def odom_callback(self, msg):
        out = PointStamped()
        out.header = msg.header
        out.point = msg.pose.pose.position
        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("vins_position_republisher")
    VinsPositionRepublisher()
    rospy.spin()
