import cv2
import json
import rospy
import carla
import cv_bridge
import tf.transformations as tf_trans
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from derived_object_msgs.msg import ObjectArray
from std_msgs.msg import String
from utils.bev_utils import BEVObject
from geometry_msgs.msg import Twist

# Town05 Parking Ambush Site Example
ambush_site = [
    -1.6003426313400269,
    12.600106239318848,
    0.16843931376934052,
    0.0004334496071379324,
    0.0002518811991329416,
    0.7071069246386728,
    0.6884453847276413
]

class Visualization:
    def __init__(self, ambush_site=None, vis_region=None, res=None):
        self.ambush_site = ambush_site  # Ambush site.
        self.vis_region = vis_region  # Visualization region.
        self.res = res  # Resolution.
        self.height = res[0]  # Image height (1200).
        self.width = res[1]  # Image width (900).
        self.current_gear = "D"  # Default gear (forward).
        self._ego_history = []  # List to store ego vehicle trajectory and gear.
        self._adagent_history = {}  # List to store adversarial agent info.
        self._global_step = 0  # Frame counter.
        self._skip_frame = 5  # Skip every 5 frames.

        # Set voxel size for converting map coordinates to pixel coordinates.
        x_range = self.vis_region[2] - self.vis_region[0]
        y_range = self.vis_region[3] - self.vis_region[1]
        self.voxel_size = [
            (self.vis_region[2] - self.vis_region[0]) / self.height,
            (self.vis_region[3] - self.vis_region[1]) / self.width,
        ]
        self.origin_pixel = [
            int(self.res[1] * self.vis_region[3] / y_range - 1),
            int(self.res[0] * self.vis_region[2] / x_range - 1),
        ]

        # ROS Subscribers
        odom = message_filters.Subscriber("/carla/ego_vehicle/odometry", Odometry)
        objs = message_filters.Subscriber("/carla/ego_vehicle/objects", ObjectArray)
        self.gear_sub = rospy.Subscriber("/gear", String, self.gear_callback)
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer([odom, objs], queue_size=1, slop=0.1)
        self.time_synchronizer.registerCallback(self._callback)

        # ROS Publisher
        self.asg_pub = rospy.Publisher("/asg_visualization", Image, queue_size=1)
        self.cv_bridge = cv_bridge.CvBridge()

        # Ambush Site Loc, Yaw
        self.ambush_site_loc = np.array(ambush_site[:3])  # (x, y, z)
        _, _, self.ambush_site_yaw = tf_trans.euler_from_quaternion(ambush_site[3:])

    def gear_callback(self, msg):
        self.current_gear = msg.data  # Update current gear from message.

    def transform_map_to_local(self, map_points, ego_self_pos: np.ndarray, ego_self_yaw):
        """Transform map frame points to Ego-Vehicle frame."""
        transform_matrix = np.array([[np.cos(ego_self_yaw), np.sin(ego_self_yaw)],
                                     [-np.sin(ego_self_yaw), np.cos(ego_self_yaw)]], dtype=np.float32)
        map_points = map_points.T  # Transpose (2, N).
        map_points[0, :] -= ego_self_pos[0]
        map_points[1, :] -= ego_self_pos[1]
        local_points = transform_matrix @ map_points
        return local_points.T

    def transform_coords_to_pixels(self, local_coords):
        """Transform local coordinates to pixel format for image drawing."""
        temp = local_coords.copy()
        temp[:, 0] = -local_coords[:, 1] / self.voxel_size[0]
        temp[:, 1] = -local_coords[:, 0] / self.voxel_size[1]
        temp[:, 0] += self.origin_pixel[0]
        temp[:, 1] += self.origin_pixel[1]
        return temp.astype(np.int32)

    def _callback(self, odom_msg: Odometry, objs_msg: ObjectArray) -> None:
        self._global_step += 1
        if self._global_step % self._skip_frame != 0:
            return  # Skip frames to reduce processing load.

        canvas = np.ones((self.height, self.width, 3), np.uint8) * 255  # White canvas.

        # Get current ego vehicle pose and velocity.
        curr_ego_pose = odom_msg.pose
        curr_ego_velocity = odom_msg.twist.twist.linear.x

        # Append the current ego vehicle trajectory along with the current gear.
        self._ego_history.append((curr_ego_pose, curr_ego_velocity, self.current_gear))

        # Draw ego vehicle history. Each segment uses the corresponding gear's color.
        for ego_item in self._ego_history:
            ego_pose, ego_velocity, ego_gear = ego_item
            pose = ego_pose.pose
            twist = Twist()
            length = 10  # Ego vehicle length.
            width = 7    # Ego vehicle width.
            cls_id = 6

            ego_obj = BEVObject(pose, twist, length, width, cls_id)
            ego_vertexes_map = ego_obj.get_four_vertexes_in_map()
            ego_local_vertexes = self.transform_map_to_local(ego_vertexes_map, self.ambush_site_loc[:2], self.ambush_site_yaw)
            vertexes_pixels = self.transform_coords_to_pixels(ego_local_vertexes)

            # Choose color based on the gear at the time of the recorded trajectory segment.
            if ego_gear == "D":  # Forward gear.
                color = (255, 125, 15)  # Light blue.
            elif ego_gear == "R":  # Reverse gear.
                color = (122, 160, 255)  # Light brown.
            else:
                color = (156, 146, 130)  # Default gray.

            cv2.drawContours(canvas, [vertexes_pixels], -1, color, thickness=4)  # Draw ego vehicle.

        # Draw other objects (e.g., adversarial agents).
        for adagent_obj in objs_msg.objects:
            obj_cls_id = adagent_obj.classification
            obj_pose = adagent_obj.pose
            obj_twist = adagent_obj.twist
            length = 10
            width = 5

            bev_obj = BEVObject(obj_pose, obj_twist, length, width, obj_cls_id)
            bev_vertexes_map = bev_obj.get_four_vertexes_in_map()
            local_vertexes = self.transform_map_to_local(bev_vertexes_map, self.ambush_site_loc[:2], self.ambush_site_yaw)
            vertexes_pixels = self.transform_coords_to_pixels(local_vertexes)
            obj_id = adagent_obj.id
            if obj_id not in self._adagent_history:
                self._adagent_history[obj_id] = []
            self._adagent_history[obj_id].append(vertexes_pixels)
            for history_pixels in self._adagent_history[obj_id]:
                cv2.drawContours(canvas, [history_pixels], -1, (0, 0, 0), thickness=4)  # Black contour for obstacles.

        # Publish visualization image.
        asg_vis_msg = self.cv_bridge.cv2_to_imgmsg(canvas, "bgr8")
        self.asg_pub.publish(asg_vis_msg)
        rospy.loginfo(f"Publishing ASG results...")

if __name__ == "__main__":
    rospy.init_node("asg_validation", anonymous=True)
    visualize_region = [-10, -20, 30, 20]  # Define the region for visualization.
    visualization_resolution = [1000, 1000]  # Resolution of the canvas.
    visualization = Visualization(ambush_site=ambush_site, vis_region=visualize_region, res=visualization_resolution)
    rospy.spin()