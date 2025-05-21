import cv2
import json
import rospy
import carla
import cv_bridge
import tf.transformations as tf_trans
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from nav_msgs.msg import Path, Odometry
from derived_object_msgs.msg import ObjectArray, Object
from geometry_msgs.msg import PoseStamped, Twist
from utils.bev_utils import BEVObject

# Town02 Crossing Example 1
# crossing1
ambush_site = [
    99.71880340576172,
    -240.90411376953125,
    0.0,
    -3.71966280188038e-20,
    -1.2246467426580385e-16,
    0.00030373351765342906,
    -0.999999953872974
]
# crossing2
# ambush_site = [
#     136.13204956054688,
#     -217.18202209472656,
#     0.0,
#     0.0,
#     0.0,
#     0.7067482818884985,
#     0.7074650988194793
#
# ]

# merge1
# ambush_site = [
#     109.71880340576172,
#     -240.91018676757812,
#     0.0,
#     -3.71966280188038e-20,
#     -1.2246467426580385e-16,
#     0.00030373351765342906,
#     -0.999999953872974
# ]
# merge2
# ambush_site = [
#     164.0990753173828,
#     -191.62745666503906,
#     0.0,
#     0.0,
#     -0.0,
#     -0.00022918490548084604,
#     0.9999999737371392
# ]

# Town03 Stop
# ambush_site = [
#     -107.72811889648438,
#     -0.32876384258270264,
#     0.0,
#     -3.148582698845666e-19,
#     -1.2246427516173223e-16,
#     0.002571012883908921,
#     -0.9999966949409137
# ]

# Town03 Reverse:
# ambush_site = [
#     64.45357513427734,
#     -7.483488082885742,
#     0.05704117566347122,
#     -1.0127891888932733e-05,
#     -0.0013560433908338001,
#     -0.007468492661885249,
#     0.999971190915572
# ]


# ambush_site = [
#     5.017314434051514,
#     80.90469360351562,
#     0.0,
#     0.0,
#     0.0,
#     0.6983306920493616,
#     0.7157752751680234
# ]

# Reverse
# ambush_site = [
#     5.017314434051514,
#     80.90469360351562,
#     0.0,
#     0.0,
#     0.0,
#     0.6983306920493616,
#     0.7157752751680234
# ]
# ambush_site = [
#     70.70928192138672,
#     202.72946166992188,
#     0.0,
#     0.0,
#     0.0,
#     0.999921092453533,
#     0.01256219990818032
# ]


# From min to max the Green color in Morandy Style
ego_morandy = [(213, 228, 213), (183, 204, 183), (153, 179, 153),
               (123, 155, 123), (93, 130, 93), (63, 106, 63)]

adagent_morandy = [(238, 228, 213),  # 浅蓝色
    (214, 204, 183),  # 稍深的浅蓝色
    (191, 179, 153),  # 中浅蓝色
    (167, 155, 123),  # 中等蓝色
    (144, 130, 93),   # 稍深的中等蓝色
    (120, 106, 63)]


class Visualization:
    def __init__(self, ambush_site = None, vis_region = None, res = None):
        self.ambush_site = ambush_site # ambush site .
        self.vis_region = vis_region # visualization region (1200, 900) corresponding to (x range, y range).
        self.res = res # resolution .

        self.height = res[0]  # 1200 - x
        self.width = res[1]   # 900  - y

        x_range = self.vis_region[2] - self.vis_region[0]
        y_range = self.vis_region[3] - self.vis_region[1]
        # voxel size : x, y in 3D frame.
        self.voxel_size = [(self.vis_region[2]-self.vis_region[0])/self.height, (self.vis_region[3] - self.vis_region[1])/self.width]
        print(self.voxel_size)

        self.origin_pixel = [int(self.res[1] * self.vis_region[3] / y_range - 1),
                             int(self.res[0] * self.vis_region[2] / x_range - 1)]

        # path_sub = rospy.Subscriber("/carla/ego_vehicle/waypoints", Path, self._path_callback)
        global_path_msg = rospy.wait_for_message("/carla/ego_vehicle/waypoints", Path, timeout=5.0)
        self.global_path_poses = global_path_msg.poses # we collect all global path waypoints .
        rospy.loginfo(f"Global path of the AV acquired !")

        odom = message_filters.Subscriber("/carla/ego_vehicle/odometry", Odometry)
        objs = message_filters.Subscriber("/carla/ego_vehicle/objects", ObjectArray)
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer([odom, objs], queue_size=1, slop=0.1)
        self.time_synchronizer.registerCallback(self._callback)

        self.cv_bridge = cv_bridge.CvBridge()

        # publishers
        self.asg_pub = rospy.Publisher("/asg_visualization", Image, queue_size=1)

        # Ambush Site Loc, Yaw
        self.ambush_site_loc = np.array(ambush_site[:3]) # (x, y, z)
        _, _,  self.ambush_site_yaw = tf_trans.euler_from_quaternion(ambush_site[3:])


        #
        self._ego_history = []
        self._adagent_history = []


        # frame to skip
        self._global_step = 0
        self._skip_frame = 5


        # Loading the Town Segment Map .
        with open("Town02_segment_lines.json", "r") as f:
            town_segments = json.load(f)

        assert len(town_segments["segment_left_lines"]) == len(town_segments["segment_right_lines"])
        segment_left_lines, segment_right_lines = town_segments["segment_left_lines"], town_segments["segment_right_lines"]
        self.segment_left_lines = [np.array(item) for item in segment_left_lines]
        self.segment_right_lines = [np.array(item) for item in segment_right_lines]
        rospy.loginfo(f"Town Map Checked !")

        print(f"site: {self.ambush_site_loc}, yaw: {self.ambush_site_yaw}")


    def transform_map_to_local(self, map_points, ego_self_pos: np.ndarray, ego_self_yaw):
        """
        This function transform all points from map frame to Ego-Vehicle's frame.
        :param map_points: np.ndarray, (N, 2), points in map frame.
        :param ego_self_pos: np.ndarray, (1, 2), ego vehicle's center position in map frame.
        :param ego_self_yaw: the ego vehicle's current yaw angle
        :return: np.ndarray, (N, 2), shape the same as map_points
        """
        transform_matrix = np.array([[ np.cos(ego_self_yaw), np.sin(ego_self_yaw)],
                                     [-np.sin(ego_self_yaw), np.cos(ego_self_yaw)]], dtype= np.float32)
        map_points = map_points.T # (2, N)
        map_points[0, :] -= ego_self_pos[0]
        map_points[1, :] -= ego_self_pos[1]
        local_points = transform_matrix @ map_points
        return local_points.T


    def transform_coords_to_pixels(self, local_coords):
        """
        This function transform the local points into image space, pixel format.
        :param local_coords: np.ndarray: (N, 2)
        :return: local points' pixels in image, np.ndarray, (N, 2)
        """

        temp = local_coords.copy()
        temp[:, 0] = -local_coords[:, 1] / self.voxel_size[0]
        temp[:, 1] = -local_coords[:, 0] / self.voxel_size[1]

        temp[:, 0] += self.origin_pixel[0]
        temp[:, 1] += self.origin_pixel[1]
        temp = temp.astype(np.int32)
        return temp


    def _callback(self, odom_msg: Odometry, objs_msg: ObjectArray) -> None:
        global color_index
        self._global_step += 1
        # rospy.loginfo(f"Messages receiving ...")
        canvas = np.ones((self.height, self.width, 3), np.uint8)
        canvas = canvas * 255
        rospy.loginfo(self.origin_pixel)

        """
        Remember this, when you draw anything through cv2 on a picture.
        (x, y) corresponding to the image coord-system.
        """
        # cv2.circle(canvas, self.origin_pixel, 10, (0, 0, 0), -1)

        if self._global_step % self._skip_frame != 0:
            rospy.loginfo(f"Not painting ...")
            return

        else:
        # First, we put all global path waypoints into the Region.
        # route_waypoints_map = []
        # for pose in self.global_path_poses:
        #     wpt_x = pose.pose.position.x
        #     wpt_y = pose.pose.position.y
        #     local_x = (wpt_x - self.ambush_site_loc[0]) * np.cos(self.ambush_site_yaw) + (wpt_y - self.ambush_site_loc[1]) * np.sin(self.ambush_site_yaw)
        #     local_y = -(wpt_x - self.ambush_site_loc[0]) * np.sin(self.ambush_site_yaw) + (
        #                 wpt_y - self.ambush_site_loc[1]) * np.cos(self.ambush_site_yaw)
        #     img_coord_x = -local_y / self.voxel_size[1] + self.origin_pixel[0]
        #     img_coord_y = -local_x / self.voxel_size[0] + self.origin_pixel[1]
        #     route_waypoints_map.append([img_coord_x, img_coord_y])
        #
        # route_local = np.array(route_waypoints_map, dtype= np.int32)
        # cv2.polylines(canvas, [route_local], False, (128, 0, 128), 40)


            # Second, we draw ego_vehicle's history pose into the Region.
            curr_ego_pose = odom_msg.pose # PoseWithCovariance .
            curr_ego_velocity = odom_msg.twist.twist.linear.x
            self._ego_history.append((curr_ego_pose, curr_ego_velocity))

            if self._adagent_history != []:
                # _adagent_history_traj = []
                for adagent_obj in self._adagent_history:
                    # wpt_x = adagent_pose[0]
                    # wpt_y = adagent_pose[1]
                    # local_x = (wpt_x - self.ambush_site_loc[0]) * np.cos(self.ambush_site_yaw) + (wpt_y - self.ambush_site_loc[1]) * np.sin(self.ambush_site_yaw)
                    # local_y = -(wpt_x - self.ambush_site_loc[0]) * np.sin(self.ambush_site_yaw) + (
                    #             wpt_y - self.ambush_site_loc[1]) * np.cos(self.ambush_site_yaw)
                    # img_coord_x = int(-local_y / self.voxel_size[1] + self.origin_pixel[0])
                    # img_coord_y = int(-local_x / self.voxel_size[0] + self.origin_pixel[1])
                    # cv2.circle(canvas, (img_coord_x, img_coord_y), 5, (255, 0, 0), -1)
                    obj_cls_id = adagent_obj.classification
                    obj_pose = adagent_obj.pose
                    obj_twist = adagent_obj.twist
                    length, width, height = adagent_obj.shape.dimensions
                    bev_obj = BEVObject(obj_pose, obj_twist, length, width, obj_cls_id)
                    bev_vertexes_map = bev_obj.get_four_vertexes_in_map()
                    local_vertexes = self.transform_map_to_local(bev_vertexes_map, self.ambush_site_loc[:2],
                                                                 self.ambush_site_yaw)
                    vertexes_pixels = self.transform_coords_to_pixels(local_vertexes)
                    vertexes_pixels = vertexes_pixels.astype(np.int32)

                    color_index_obj = int(obj_twist.linear.x / 0.3)
                    cv2.drawContours(canvas, [vertexes_pixels], -1, (0, 0, 0), thickness=4)
                    cv2.drawContours(canvas, [vertexes_pixels], -1, adagent_morandy[-1], -1)


            if self._ego_history != []:
                _ego_history_traj = []
                for ego_item in self._ego_history:
                    ego_pose, ego_velocity = ego_item
                    pose = ego_pose.pose
                    twist = Twist()
                    length = 2.52
                    width = 1.47
                    cls_id = 6
                    ego_obj = BEVObject(pose, twist, length, width, cls_id)
                    ego_vertexes_map = ego_obj.get_four_vertexes_in_map()
                    ego_local_vertexes = self.transform_map_to_local(ego_vertexes_map, self.ambush_site_loc[:2], self.ambush_site_yaw)
                    vertexes_pixels = self.transform_coords_to_pixels(ego_local_vertexes)
                    vertexes_pixels = vertexes_pixels.astype(np.int32)

                    print(f"vertexes pixels shape: {vertexes_pixels.shape}")

                    # color_index = int(ego_velocity/0.4)
                    color_index = 0
                    if ego_velocity > 0.8:
                        color_index = 5
                    if ego_velocity > 0.5 and ego_velocity < 0.8:
                        color_index = 2

                    if ego_velocity <0.3:
                        color_index = 0

                    cv2.drawContours(canvas, [vertexes_pixels], -1, (0, 0, 0), thickness=4)
                    cv2.drawContours(canvas, [vertexes_pixels], -1, ego_morandy[color_index], -1)

                    # wpt_x = ego_pose[0]
                    # wpt_y = ego_pose[1]
                    # local_x = (wpt_x - self.ambush_site_loc[0]) * np.cos(self.ambush_site_yaw) + (wpt_y - self.ambush_site_loc[1]) * np.sin(self.ambush_site_yaw)
                    # local_y = -(wpt_x - self.ambush_site_loc[0]) * np.sin(self.ambush_site_yaw) + (
                    #             wpt_y - self.ambush_site_loc[1]) * np.cos(self.ambush_site_yaw)
                    # img_coord_x = int(-local_y / self.voxel_size[1] + self.origin_pixel[0])
                    # img_coord_y = int(-local_x / self.voxel_size[0] + self.origin_pixel[1])
                    # # _ego_history_traj.append([img_coord_x, img_coord_y])
                    # cv2.circle(canvas, (img_coord_x, img_coord_y), 5, (0, 255, 0), -1)

            rospy.loginfo(f"There are total {len(objs_msg.objects)} objects !")

            if len(objs_msg.objects) > 0:
                for obj in objs_msg.objects:
                    self._adagent_history.append(obj)
                    rospy.loginfo(f"Check the adversarial agent !")



            cos_yaw, sin_yaw = np.cos(self.ambush_site_yaw), np.sin(self.ambush_site_yaw)
            rot_matrix = np.array([[cos_yaw, -sin_yaw],
                                   [sin_yaw, cos_yaw]])
            # Draw the traffic lines.
            for id in range(len(self.segment_left_lines)):
                translated_left_points = self.segment_left_lines[id] - [self.ambush_site_loc[0], self.ambush_site_loc[1]]
                translated_right_points = self.segment_right_lines[id] - [self.ambush_site_loc[0], self.ambush_site_loc[1]]

                transformed_left_points = np.dot(translated_left_points, rot_matrix)
                transformed_right_points = np.dot(translated_right_points, rot_matrix)

                img_coords_left_final = np.zeros(transformed_left_points.shape)
                img_coords_right_final = np.zeros(transformed_right_points.shape)

                img_coords_left = -transformed_left_points / self.voxel_size[1]
                img_coords_right = -transformed_right_points / self.voxel_size[0]

                img_coords_left_final[:, 0] = img_coords_left[:, 1]
                img_coords_left_final[:, 1] = img_coords_left[:, 0]

                img_coords_right_final[:, 0] = img_coords_right[:, 1]
                img_coords_right_final[:, 1] = img_coords_right[:, 0]

                final_left = img_coords_left_final + [self.origin_pixel[0], self.origin_pixel[1]]
                final_right = img_coords_right_final + [self.origin_pixel[0], self.origin_pixel[1]]

                left_line = np.array([final_left], dtype=np.int32)
                right_line = np.array([final_right], dtype=np.int32)

                cv2.polylines(img=canvas, pts=left_line, isClosed=False, color=1, thickness=2)
                cv2.polylines(img=canvas, pts=right_line, isClosed=False, color=1, thickness=2)

        asg_vis_msg = self.cv_bridge.cv2_to_imgmsg(canvas, "bgr8")
        self.asg_pub.publish(asg_vis_msg)
        rospy.loginfo(f"Publishing ASG results ...")


if __name__ == "__main__":
    rospy.init_node("asg_validation", anonymous=True)
    """
            ^y
            |
            |
            |
    <-------O (ambush site)
    x
    """
    visualize_region = [-20, -20, 100, 100] # x: -30m to +10m; y: -15m to +15m .
    visualization_resolution = [1000, 1000]
    visualization = Visualization(ambush_site= ambush_site,
                                  vis_region = visualize_region,
                                  res = visualization_resolution)
    rospy.spin()
