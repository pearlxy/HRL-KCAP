#!/usr/bin/env python

import rospy
import cv_bridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from derived_object_msgs.msg import ObjectArray
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
import numpy as np
import message_filters

class Visualization:
    def __init__(self, ambush_site=None, vis_region=None, res=None):
        self.ambush_site = ambush_site
        self.vis_region = vis_region
        self.res = res
        self.height = res[0]
        self.width = res[1]
        self.current_gear = "D"
        self._global_step = 0
        self._skip_frame = 5

        # ROS Subscribers
        odom = message_filters.Subscriber("/carla/ego_vehicle/odometry", Odometry)
        self.gear_sub = rospy.Subscriber("/gear", String, self.gear_callback)

        # Synchronize topics
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer([odom], queue_size=1, slop=0.1)
        self.time_synchronizer.registerCallback(self._callback)

        # ROS Publishers
        self.speed_pub = rospy.Publisher("/vehicle/speed", Float32, queue_size=10)
        self.gear_pub = rospy.Publisher("/vehicle/gear", String, queue_size=10)
        self.steering_pub = rospy.Publisher("/vehicle/steering", Float32, queue_size=10)
        self.throttle_pub = rospy.Publisher("/vehicle/throttle", Float32, queue_size=10)
        self.asg_pub = rospy.Publisher("/asg_visualization", Image, queue_size=1)

        self.cv_bridge = cv_bridge.CvBridge()

    def gear_callback(self, msg):
        self.current_gear = msg.data  # Update current gear

    def _callback(self, odom_msg: Odometry) -> None:
        self._global_step += 1
        if self._global_step % self._skip_frame != 0:
            return  # Skip frames to reduce processing load

        # Get current ego vehicle velocity, steering, and throttle (for demo purposes, assuming throttle as linear.x)
        current_speed = odom_msg.twist.twist.linear.x
        steering_angle = odom_msg.twist.twist.angular.z
        throttle = odom_msg.twist.twist.linear.x  # Assuming throttle is linear velocity

        # Publish speed, gear, steering, and throttle data
        self.speed_pub.publish(current_speed)
        self.gear_pub.publish(self.current_gear)
        self.steering_pub.publish(steering_angle)
        self.throttle_pub.publish(throttle)

        # For visualization and logging
        rospy.loginfo(f"Speed: {current_speed}, Gear: {self.current_gear}, Steering: {steering_angle}, Throttle: {throttle}")

        # Visualization logic goes here (as in your original code)

if __name__ == "__main__":
    rospy.init_node("vehicle_monitor", anonymous=True)
    visualize_region = [-10, -20, 30, 20]  # Define the region for visualization
    visualization_resolution = [1000, 1000]  # Resolution of the canvas
    visualization = Visualization(ambush_site=ambush_site, vis_region=visualize_region, res=visualization_resolution)
    rospy.spin()