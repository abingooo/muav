#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class PlanPointLoopNode:
    """连续发送航点并等待 waypoint_done 回执的闭环测试."""

    def __init__(self):
        self.topic = rospy.get_param("~topic", "/single_plan_point")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.feedback_topic = rospy.get_param(
            "~feedback_topic", "planning/waypoint_done"
        )
        self.max_cycles = rospy.get_param("~max_cycles", 20)
        self.wait_timeout = rospy.get_param("~wait_timeout", 1.0)

        self.publisher = rospy.Publisher(self.topic, PoseStamped, queue_size=10)
        self.subscriber = rospy.Subscriber(
            self.feedback_topic, String, self._feedback_cb, queue_size=10
        )

        self.sent_count = 0
        self.confirmed_count = 0
        self.waiting_for_feedback = False

        rospy.loginfo(
            "PlanPointLoopNode ready, looping %d times between %s and %s",
            self.max_cycles,
            self.topic,
            self.feedback_topic,
        )

        rospy.Timer(rospy.Duration(0.1), self._initial_publish, oneshot=True)

    def _initial_publish(self, _event):
        if self.sent_count == 0:
            self._publish_plan_point()

    def _publish_plan_point(self):
        if self.sent_count >= self.max_cycles:
            rospy.loginfo("All plan points already sent")
            return

        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while (
            self.publisher.get_num_connections() == 0
            and not rospy.is_shutdown()
            and (rospy.Time.now() - start).to_sec() < self.wait_timeout
        ):
            rate.sleep()

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = 1.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0

        self.publisher.publish(pose)
        self.sent_count += 1
        self.waiting_for_feedback = True
        rospy.loginfo(
            "Published plan point #%d: (%.2f, %.2f, %.2f)",
            self.sent_count,
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )

    def _feedback_cb(self, msg):
        if not self.waiting_for_feedback:
            rospy.logwarn("Received unexpected feedback: %s", msg.data)
            return

        self.confirmed_count += 1
        rospy.loginfo(
            "Received waypoint_done #%d with payload: %s",
            self.confirmed_count,
            msg.data,
        )

        if self.confirmed_count >= self.max_cycles:
            rospy.loginfo("Completed %d send/ack cycles, shutting down", self.max_cycles)
            rospy.signal_shutdown("loop complete")
            return

        self.waiting_for_feedback = False
        self._publish_plan_point()


def main():
    rospy.init_node("test_plan_point_loop_py")
    PlanPointLoopNode()
    rospy.spin()


if __name__ == "__main__":
    main()
