#!/usr/bin/env python3.8
import csv
import numpy as np
import subprocess
import torch
import os
import random
import math
import carla
import time
import cv2
import rospy
from std_msgs.msg import String
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl, CarlaCollisionEvent, CarlaEgoVehicleInfo
from tf import transformations
from skimage.transform import resize
from std_msgs.msg import UInt32
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Point32, PolygonStamped, Pose, PoseStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan, Image, Imu
from derived_object_msgs.msg import ObjectArray
from jsk_recognition_msgs.msg import PolygonArray
import message_filters
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header

# Configuration constants
_HOST_ = '127.0.0.1'
_PORT_ = 2000
_SLEEP_TIME_ = 0.5

TRAINING_INDICATOR = 2
SELECTED_MODEL = 'only_throttle'
TRAINING_NAME = 'training'
SELECTED_SPAWNING_METHOD = 0

ACTIONS_SIZE = 1
STATE_SIZE = 16

MAX_COLLISION_IMPULSE = 50
MAX_DISTANCE = 15.7
MAX_REWARD = 20.0
SIGMA = 2.0

# Training parameters
MEMORY_FRACTION = 0.3333
TOTAL_EPISODES = 1000
STEPS_PER_EPISODE = 100
AVERAGE_EPISODES_COUNT = 40
CORRECT_POSITION_NON_MOVING_STEPS = 5
OFF_POSITION_NON_MOVING_STEPS = 100
REPLAY_BUFFER_CAPACITY = 100000
BATCH_SIZE = 64
CRITIC_LR = 0.002
ACTOR_LR = 0.001
GAMMA = 0.99
TAU = 0.005
epsilon = 1
EXPLORE = 100000.0
MIN_EPSILON = 0.000001

class ParkingEnv():
    def __init__(self):
        rospy.init_node("park")
        # Publisher for gear data
        self.gear_pub = rospy.Publisher("/gear", String, queue_size=10)
        self.predefined_locations = [
            carla.Transform(carla.Location(x=-1.6, y=-12.6, z=0.15), carla.Rotation(pitch=0, yaw=-90, roll=0))
        ]
        self.parking_spots = {
            "east_park": {"location": carla.Location(x=17, y=-20, z=0.05), "rotation": carla.Rotation(pitch=0, yaw=268, roll=0), "check_stop_condition": self.check_stop_condition_east, "calculate_reward": self.calculate_reward_east},
            "west_park": {"location": carla.Location(x=-1.6, y=-12.6, z=0.15), "rotation": carla.Rotation(pitch=0, yaw=90, roll=0), "check_stop_condition": self.check_stop_condition_west, "calculate_reward": self.calculate_reward_west}
        }
        self.parking_areas = {
            'east_park': {
                'x_range': (2, 5),
                'y_range': (2, 5),
                'theta_range': (1.221, 1.7453),
                'center': ((2 + 5)/2, (2 + 5)/2)
            },
            'west_park': {
                'x_range': (3, 5.1),
                'y_range': (-3.5, -1.7),
                'theta_range': (-1.9, -1.221),
                'center': ((3 + 5.1)/2, (-3.5 + -1.7)/2)
            }
        }
        self.car_start_transform = None
        self.park_list = {}
        # Initialize CARLA client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.parking_point_checked = False
        self.vehicle_exits = False
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.audi_actor = None
        self.obstacle_active = False
        self.flag1 = 1
        self.model_3_heigth = 1.443
        self.width = 2.089
        self.height = 4.69
        self.index = 0
        self.ego_location = None
        self.ego_heading = None
        self.ego_speed = None
        self.surrounding_objs = None
        self.ego_odom = None
        self.ego_vehicle_status = None
        self.cad_ranges = None
        self.bev_range = [-22, -22, 24, 22]
        self.bev_img_size = [512, 512]
        self.next_wpt = None
        self.last_diff_y = float('inf')
        self.last_diff_x = float('inf')
        self.diff_y = None
        self.diff_x = None
        self.kp = 30
        self.last_distance_to_goal = 0
        self.current_distance_to_goal = 0
        self.past_distance = []
        self.angle_history = []
        self.num_distance = 5
        self.obstacle_vehicle = False
        self.obstacle_moving = False
        self.reverse = False
        self.last_angle = 0
        self.current_angle = 0
        ego_vehicle_info = rospy.wait_for_message("/carla/ego_vehicle/vehicle_info", CarlaEgoVehicleInfo)
        self.ego_id = ego_vehicle_info.id
        self.ego_vehicle = self.world.get_actor(ego_vehicle_info.id)
        self.current_select_area = None
        self.car_transform = self.ego_vehicle.get_transform()
        gen_park_flag = self.gen_park_list()
        if not gen_park_flag:
            return False
        self.target_park = self.park_list["park_15"]
        self.target_x = self.park_list["park_15"]["location"].x
        self.target_y = self.park_list["park_15"]["location"].y
        self.target_width = self.park_list["park_15"]["width"]
        self.target_height = self.park_list["park_15"]["height"]
        self.park_yaw = self.park_list["park_15"]["rotation"].yaw
        debug = self.world.debug
        debug.draw_box(carla.BoundingBox(self.park_list["park_15"]["location"], carla.Vector3D(self.target_height/2, self.target_width/2, 0.1)), self.park_list["park_15"]["rotation"], 0.05, carla.Color(255,0,0,0), 0)
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(self.target_park["location"] + carla.Location(z=20), carla.Rotation(pitch=-90)))
        self.bev_obs_size = [128, 128]
        x_range = self.bev_range[2] - self.bev_range[0]
        y_range = self.bev_range[3] - self.bev_range[1]
        ego_vehicle_dimensions = [2.52, 1.47]
        target_velocity = 10
        self.ego_anchor_pixel = [int(self.bev_img_size[1] * self.bev_range[3] / y_range - 1), int(self.bev_img_size[0] * self.bev_range[2] / x_range - 1)]
        self.ego_vehicle_dimensions = ego_vehicle_dimensions
        self.run_off_dis_threshold = 2
        self.angle_diff_max = 7
        self.target_velocity = target_velocity
        self.min_velocity = 2
        self.max_velocity = 10
        self.voxel_size = [(self.bev_range[2]-self.bev_range[0])/self.bev_img_size[0], (self.bev_range[3]-self.bev_range[1])/self.bev_img_size[1]]
        self.invalid_stop_frames = 0
        self.collision_with_actor = None
        self.crossed_lane_markings = None
        self.hm = 2.875
        self.hp = 6
        self.h = 3
        self.Rmin = 3.6
        self.bridge = CvBridge()
        self.zero_control_time = 0
        self.time_step = 0.05
        # Subscribers
        self.odom_sub = rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, self.odometry_callback)
        odom_sub = message_filters.Subscriber("/carla/ego_vehicle/imu", Imu)
        vehicle_status_sub = message_filters.Subscriber("/carla/ego_vehicle/vehicle_status", CarlaEgoVehicleStatus)
        objs_sub = message_filters.Subscriber("/carla/ego_vehicle/objects", ObjectArray)
        cad_sub = message_filters.Subscriber("/cad_carla_gt", LaserScan)
        self.sys_synchronizer = message_filters.ApproximateTimeSynchronizer([odom_sub, vehicle_status_sub, objs_sub, cad_sub], queue_size=10, slop=0.1)
        self.sys_synchronizer.registerCallback(self.sys_callback_api)
        collision_sub = rospy.Subscriber("/carla/ego_vehicle/collision", CarlaCollisionEvent, self.collision_callback)
        self.safety_polygon_pub = rospy.Publisher("/safety_area", PolygonArray, queue_size=2)
        # Publishers
        self.global_goal_pub = rospy.Publisher("/carla/ego_vehicle/goal", PoseStamped, latch=True, queue_size=5)
        self.control_pub = rospy.Publisher("/carla/ego_vehicle/vehicle_control_cmd", CarlaEgoVehicleControl, latch=True, queue_size=10)
        self.cad_render_pub = rospy.Publisher("/bev_perception", Image, queue_size=10)
        self.park_pub = rospy.Publisher("/map/parklist", MarkerArray, queue_size=10)
        self.global_path = rospy.Publisher("/vehicle_path", Path, queue_size=10)
        self.vehicle_path = Path()
        self.vehicle_path.header.frame_id = "map"
        self.learning_steps_threshold = 100

    def odometry_callback(self, odom_msg):
        """Callback to handle odometry data and publish vehicle path."""
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position.x = position.x
        pose_stamped.pose.position.y = position.y
        pose_stamped.pose.position.z = position.z
        pose_stamped.pose.orientation = orientation
        self.vehicle_path.poses.append(pose_stamped)
        self.vehicle_path.header.stamp = rospy.Time.now()
        self.global_path.publish(self.vehicle_path)

    def publish_gear(self):
        """Publish gear state (R for reverse, D for drive)."""
        gear_msg = "R" if self.reverse else "D"
        self.gear_pub.publish(gear_msg)

    def clear_vehicles(self):
        """Destroy all vehicle actors in the simulation."""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        vehicle_actors = self.world.get_actors().filter("vehicle.*")
        for vehicle in vehicle_actors:
            vehicle.destroy()
            rospy.loginfo(f"Vehicle {vehicle.id} destroyed")
            self.world.tick()
            time.sleep(0.1)
            try:
                self.world.tick()
            except Exception as e:
                pass
            time.sleep(0.5)

    def sys_callback_api(self, imu_msg: Imu, vehicle_status_msg: CarlaEgoVehicleStatus, objs_msg: ObjectArray, cad_msg: LaserScan):
        """Callback to process system messages and update ego vehicle state."""
        ego_loc = self.ego_vehicle.get_location()
        ego_pos = np.array([ego_loc.x, -ego_loc.y])
        surrounding_objs = [obj for obj in objs_msg.objects if np.linalg.norm(ego_pos - np.array([obj.pose.position.x, obj.pose.position.y])) < 30]
        self.surrounding_objs = surrounding_objs
        self.ego_odom = Odometry()
        self.ego_odom.pose.pose.position.x = ego_loc.x
        self.ego_odom.pose.pose.position.y = -ego_loc.y
        self.ego_odom.pose.pose.position.z = ego_loc.z
        self.ego_odom.pose.pose.orientation.x = vehicle_status_msg.orientation.x
        self.ego_odom.pose.pose.orientation.y = vehicle_status_msg.orientation.y
        self.ego_odom.pose.pose.orientation.z = vehicle_status_msg.orientation.z
        self.ego_odom.pose.pose.orientation.w = vehicle_status_msg.orientation.w
        self.ego_odom.twist.twist.linear.x = vehicle_status_msg.velocity
        self.ego_odom.twist.twist.angular.z = imu_msg.angular_velocity.z
        self.ego_vehicle_status = vehicle_status_msg
        self.cad_ranges = cad_msg.ranges
        self.ego_location = np.array([self.ego_odom.pose.pose.position.x, self.ego_odom.pose.pose.position.y, self.ego_odom.pose.pose.position.z])
        self.ego_heading = np.array([self.ego_odom.pose.pose.orientation.x, self.ego_odom.pose.pose.orientation.y, self.ego_odom.pose.pose.orientation.z, self.ego_odom.pose.pose.orientation.w])
        self.ego_speed = [self.ego_odom.twist.twist.linear.x, self.ego_odom.twist.twist.angular.z]

    def collision_callback(self, collision_msg: CarlaCollisionEvent):
        """Callback to handle collision events."""
        self.collision_with_actor = collision_msg.other_actor_id

    def transform_angle(self, angle):
        """Convert angle from -180 to 180 degrees to 0 to 360 degrees."""
        return 360 + angle if angle < 0 else angle

    def get_state(self):
        """Retrieve the current state of the ego vehicle."""
        current_vehicle_transform = self.ego_vehicle.get_transform()
        current_vehicle_location = current_vehicle_transform.location
        current_vehicle_x = current_vehicle_location.x
        current_vehicle_y = current_vehicle_location.y
        angle = self.transform_angle(current_vehicle_transform.rotation.yaw)
        current_vehicle_linear_velocity = self.ego_vehicle.get_velocity().x
        current_vehicle_angular_velocity = self.ego_vehicle.get_angular_velocity().z
        x = current_vehicle_x
        y = current_vehicle_y
        x_rel = self.target_park["location"].x - current_vehicle_x
        y_rel = self.target_park["location"].y - current_vehicle_y
        angle = self.transform_angle(angle)
        vx = current_vehicle_linear_velocity
        wz = current_vehicle_angular_velocity
        distance_to_goal = current_vehicle_location.distance(self.target_park['location'])
        sensor_values_dict = {
            'x': x,
            'y': y,
            'x_rel': x_rel,
            'y_rel': y_rel,
            'angle': angle,
            'vx': vx,
            'wz': wz,
            'distance_to_goal': distance_to_goal,
        }
        return sensor_values_dict

    def get_cad_bev(self):
        """Convert CAD perception into a BEV image."""
        if self.cad_ranges is None:
            rospy.loginfo("No CAD perception!")
            return None, None, None
        adjusted_height = self.bev_img_size[0]
        bev_img = np.zeros((adjusted_height, self.bev_img_size[1], 3), dtype=np.uint8)
        cad_points = []
        closest_north_distance = float('inf')
        closest_north_angle = None
        north_area = self.target_park['rotation'].yaw
        range_points = 200
        angle_step = 2 * np.pi / 384
        current_idx = int((north_area + np.pi) / angle_step)
        left_idx = (current_idx - range_points) % 384
        right_idx = (current_idx + range_points) % 384
        for idx in range(len(self.cad_ranges)):
            r = self.cad_ranges[idx] / self.voxel_size[0]
            r1 = self.cad_ranges[idx]
            theta = (0.5 + idx) * angle_step - np.pi
            x = -r * np.sin(theta)
            y = -r * np.cos(theta)
            cad_points.append([x, y])
            if (left_idx <= idx <= right_idx) or (left_idx > right_idx and (idx >= left_idx or idx <= right_idx)):
                if r1 < closest_north_distance:
                    closest_north_distance = r1
                    closest_north_angle = theta
        cad_points = np.array(cad_points, dtype=np.int32)
        cad_points[:, 0] = cad_points[:, 0] + self.ego_anchor_pixel[0]
        cad_points[:, 1] = cad_points[:, 1] + self.ego_anchor_pixel[1]
        cv2.drawContours(bev_img, [cad_points], -1, (255, 255, 255), -1)
        resized_bev_img = cv2.resize(bev_img, (512, 512), interpolation=cv2.INTER_AREA)
        cad_bev_img_msg = self.bridge.cv2_to_imgmsg(resized_bev_img, encoding="bgr8")
        self.cad_render_pub.publish(cad_bev_img_msg)
        return bev_img, closest_north_distance, closest_north_angle

    def get_bev_img_obs(self):
        """Generate BEV image observation for RL training."""
        cad_bev_img, closest_north_distance, closest_north_angle = self.get_cad_bev()
        if cad_bev_img is None or cad_bev_img.size == 0:
            return None, None, None
        cad_bev_img_s = cv2.cvtColor(cad_bev_img, cv2.COLOR_RGB2GRAY)
        assert cad_bev_img_s.shape[:2] == (512, 512)
        stacked_img = np.zeros((512, 512, 2), dtype=np.float32)
        stacked_img[:, :, 0] = cad_bev_img_s
        bev_resized = cv2.resize(stacked_img, (self.bev_obs_size[0], self.bev_obs_size[1]))
        bev_resized = bev_resized / 255.0
        bev = np.transpose(bev_resized, (2, 0, 1))
        return bev, closest_north_distance, closest_north_angle

    def get_propriceptive_obs(self):
        """Retrieve proprioceptive observations of the ego vehicle."""
        if self.ego_vehicle_status is not None and self.ego_vehicle_status.control is not None:
            real_time_throttle = self.ego_vehicle_status.control.throttle
        real_time_steer = self.ego_vehicle_status.control.steer
        real_time_linear_vel = self.ego_odom.twist.twist.linear.x
        real_time_angular_vel = self.ego_odom.twist.twist.angular.z
        current_vehicle_transform = self.ego_vehicle.get_transform()
        _, _, ego_yaw = transformations.euler_from_quaternion(self.ego_heading)
        wpt_yaw = self.target_park["rotation"].yaw
        real_time_orientation_diff = (wpt_yaw - ego_yaw) * 180 / np.pi
        distance_to_goal = current_vehicle_transform.location.distance(self.target_park['location'])
        proprioceptive_obs = np.array([real_time_throttle, real_time_steer, real_time_linear_vel, real_time_angular_vel, distance_to_goal, real_time_orientation_diff])
        return self.ego_location, self.ego_heading, self.ego_speed, proprioceptive_obs

    def find_nearest_parking_area(self, diff_x, diff_y):
        """Determine the nearest parking area based on vehicle's position."""
        if self.current_select_area is not None:
            return self.current_select_area
        max_distance = -1
        for area_name, area_props in self.parking_areas.items():
            dx = diff_x - area_props['center'][0]
            dy = diff_y - area_props['center'][1]
            distance = math.sqrt(dx**2 + dy**2)
            if distance > max_distance:
                max_distance = distance
                nearest_park_area = area_name
        return nearest_park_area

    def get_localtion_info(self):
        """Get the current location of the vehicle."""
        current_vehicle_location = self.ego_vehicle.get_transform().location
        return current_vehicle_location.x, current_vehicle_location.y

    def reset_car(self):
        """Reset the vehicle's position until it is stationary."""
        vehicle_position_stable = False
        while not vehicle_position_stable:
            if self.ego_vehicle:
                self.ego_vehicle.destroy()
                self.ego_vehicle = None
            time.sleep(1)
            if self.vehicle_is_stationary():
                vehicle_position_stable = True
            else:
                pass

    def vehicle_is_stationary(self):
        """Check if the vehicle is stationary based on velocity and acceleration."""
        velocity = self.ego_vehicle.get_velocity()
        acceleration = self.ego_vehicle.get_acceleration()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        accel = math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
        return speed < 0.2 and accel < 0.2

    def reset(self):
        """Reset the environment to initial conditions."""
        self.ego_vehicle.apply_control(carla.VehicleControl(brake=True))
        rospy.sleep(0.3)
        self.ego_vehicle.apply_control(carla.VehicleControl(brake=False))
        self.ego_vehicle.apply_control(carla.VehicleControl(reverse=False))
        self.car_start_transform = random.choice(self.predefined_locations)
        self.ego_vehicle.set_transform(self.car_start_transform)
        self.reverse = False
        self.flag1 = 1
        self.zero_control_time = 0
        self.parking_point_checked = False
        self.collision_with_actor = None
        current_state_dict = self.get_state()
        distance = current_state_dict['distance_to_goal']
        angle = current_state_dict['angle']
        self.distance_to_goal = None
        self.last_distance_to_goal = self.distance_to_goal
        self.distance_to_goal = distance
        self.angle = None
        self.last_angle = self.angle
        self.angle = angle
        self.non_moving_steps_cnt = 0
        current_vehicle_location = self.car_start_transform.location
        current_vehicle_rotation = self.car_start_transform.rotation
        current_x = current_vehicle_location.x
        current_y = -current_vehicle_location.y
        target_park_rotation = self.target_park['rotation']
        self.target_location = self.target_park['location']
        vehicle_yaw = -current_vehicle_rotation.yaw
        park_yaw = target_park_rotation.yaw
        theta = vehicle_yaw - park_yaw
        R, steer, diff_x, diff_y, xr, yr = self.calculate_reverse_steering1(current_x, current_y, theta)
        self.current_select_area = None
        nearest_area = self.find_nearest_parking_area(diff_x, diff_y)
        self.current_select_area = nearest_area
        self.get_weight_path_by_area()

    def gen_park_list(self):
        """Generate a list of parking spots based on road lines."""
        roadline_list = self.world.get_environment_objects(carla.CityObjectLabel.RoadLines)
        self.all_park_roadline = {item.name: item for item in roadline_list if item.name.find("Plane") > -1}
        park_line_name = [f"Plane{line_index}_SM_0" for line_index in range(9, 81)]
        park_raodline = {item.name: item for item in roadline_list if item.name in park_line_name}
        self.park_list = {}
        for park_index in range(9, 80):
            left_park_name = f"Plane{park_index}_SM_0"
            right_park_name = f"Plane{park_index + 1}_SM_0"
            if left_park_name not in park_raodline or right_park_name not in park_raodline:
                continue
            temp = {}
            temp_park_line_left = park_raodline[left_park_name]
            temp_park_line_right = park_raodline[right_park_name]
            center_location = carla.Location()
            center_location.x = (temp_park_line_left.bounding_box.location.x + temp_park_line_right.bounding_box.location.x) / 2
            center_location.y = (temp_park_line_left.bounding_box.location.y + temp_park_line_right.bounding_box.location.y) / 2
            center_location.z = (temp_park_line_left.bounding_box.location.z + temp_park_line_right.bounding_box.location.z) / 2
            center_rotation = carla.Rotation(yaw=0)
            temp_width = ((temp_park_line_left.bounding_box.location.x - temp_park_line_right.bounding_box.location.x) ** 2 + 
                         (temp_park_line_left.bounding_box.location.y - temp_park_line_right.bounding_box.location.y) ** 2) ** 0.5
            temp_height = 5
            temp["location"] = center_location
            temp["rotation"] = center_rotation
            temp["width"] = temp_width
            temp["height"] = temp_height
            park_name = f"park_{park_index - 8}"
            self.park_list[park_name] = temp
        return True

    def get_car_box(self, car_transform):
        """Get the bounding box of the vehicle."""
        rect = ((car_transform.location.x, car_transform.location.y), (self.height, self.width), car_transform.rotation.yaw)
        box = cv2.boxPoints(rect)
        return box

    def check_non_movement(self, current_distance_to_goal, last_distance_to_goal):
        """Check if the vehicle has stopped moving."""
        correct_position_non_movement_indicator = False
        off_position_non_movement_indicator = False
        if abs(last_distance_to_goal - current_distance_to_goal) <= 0.05:
            self.non_moving_steps_cnt += 1
            goal_angle = self.transform_angle(self.target_park['rotation'].yaw)
            vehicle_angle = self.angle
            if self.check_if_parked(goal_angle, vehicle_angle) and (self.non_moving_steps_cnt >= CORRECT_POSITION_NON_MOVING_STEPS):
                correct_position_non_movement_indicator = True
            elif (not self.check_if_parked(goal_angle, vehicle_angle)) and (self.non_moving_steps_cnt >= OFF_POSITION_NON_MOVING_STEPS):
                off_position_non_movement_indicator = True
        else:
            self.non_moving_steps_cnt = 0
        return correct_position_non_movement_indicator, off_position_non_movement_indicator

    def check_if_parked(self, goal_angle, vehicle_angle):
        """Check if the vehicle is parked correctly."""
        vehicle_parked = False
        if (self.current_distance_to_goal <= 0.8) and ((abs(goal_angle - vehicle_angle) <= 20) or (abs(goal_angle - vehicle_angle) >= 160)):
            vehicle_parked = True
        return vehicle_parked

    def calculate_reverse_steering(self, target_x, target_y, theta):
        """Calculate steering parameters for reverse parking (left-front)."""
        xr = target_x - (self.hm * math.cos(theta) / 2)
        yr = target_y - (self.hm * math.sin(theta) / 2)
        diff_x = xr - self.target_park['location'].x
        diff_y = yr - (-self.target_park['location'].y)
        R = (diff_x - self.h) / (math.sin(theta))
        phi = math.atan(self.hm / R)
        if phi > math.pi / 2:
            phi = math.pi - phi
        phi_max = math.atan(self.hm / self.Rmin)
        steer = phi / phi_max
        steer = max(0, min(1, steer))
        return R, steer, diff_x, diff_y

    def calculate_reverse_steering1(self, target_x, target_y, theta):
        """Calculate steering parameters for reverse parking (right-front)."""
        xr = target_x - (self.hm * math.cos(theta) / 2)
        yr = target_y - (self.hm * (-math.sin(theta)) / 2)
        diff_x = xr - self.target_park['location'].x
        diff_y = yr - (-self.target_park['location'].y)
        R = (diff_x - self.h) / (-math.sin(theta) + 0.01)
        phi = math.atan(self.hm / R)
        if phi > math.pi / 2:
            phi = math.pi - phi
        phi_max = math.atan(self.hm / self.Rmin)
        steer = phi / phi_max
        steer = max(-1, min(1, steer))
        return R, steer, diff_x, diff_y, xr, yr

    def check_stop_condition_east(self, diff_x, diff_y, theta):
        """Check stopping condition for east parking area."""
        theta_radian = math.radians(theta)
        x_condition = 4.5 < diff_x < 6.5
        y_condition = 3.5 <= diff_y < 5.0
        theta_condition = 1.221 <= theta_radian <= 1.7453
        return x_condition and y_condition and theta_condition

    def get_current_parking_status(self):
        """Return the current parking area for training."""
        return self.current_select_area

    def get_weight_path_by_area(self):
        """Retrieve the policy path based on the current parking area."""
        policy_path = "/home/vsisauto/xxy/ppo_0116/checkpoints_Town03east1/"
        current_area_name = self.get_current_parking_status()
        if current_area_name == "east_park":
            policy_path = "/home/vsisauto/xxy/ppo_0116/checkpoints_Town03east1/"
        elif current_area_name == "west_park":
            policy_path = "/home/vsisauto/xxy/ppo_0116/checkpoints_Town02west/"
        return policy_path

    def calculate_reward_east(self, diff_y, vehicle_yaw, park_yaw, last_diff_y, diff_x, last_diff_x, at_parking_point, vehicle_transform, target_location, d_val_1=6.5, d_val_2=11.5):
        """Calculate reward for east parking area."""
        current_distance_to_goal = self.current_distance_to_goal
        last_distance_to_goal = self.last_distance_to_goal
        if len(self.past_distance) < self.num_distance:
            return 0
        average_past_distance = sum(self.past_distance) / len(self.past_distance)
        vehicle_forward_vector = np.array([math.cos(vehicle_transform.rotation.yaw), math.sin(vehicle_transform.rotation.yaw)])
        to_target_vector = np.array([target_location.x - vehicle_transform.location.x, (-target_location.y) - (-vehicle_transform.location.y)])
        dot_product = np.dot(vehicle_forward_vector, to_target_vector)
        theta = vehicle_yaw - park_yaw
        theta = math.radians(theta)
        angle_reward = 0
        angle_weight = 0.5
        distance_reward = 0
        distance_weight = 0
        optimal_direction = np.pi / 2
        theta_diff = theta - optimal_direction
        if -0.15 <= theta_diff < 0.01:
            angle_reward = 5
        elif 0.01 <= theta_diff < 0.05:
            angle_reward = 10
        elif 0.05 <= theta_diff <= 0.39:
            angle_reward = 15
        elif 0.35 <= theta_diff < 0.50:
            angle_reward = -5
        else:
            angle_reward = -10
        if 5.5 <= current_distance_to_goal <= d_val_1:
            if 5 <= diff_x <= 7 or 3.5 <= diff_y <= 6.0:
                distance_reward = 10
                if 5 <= diff_x <= 7 and 4.3 <= diff_y <= 6:
                    distance_reward = 15
            else:
                distance_reward = -5
            distance_weight = 1
            angle_weight = 1.0
        elif d_val_1 < current_distance_to_goal <= d_val_2:
            distance_reward = -2
            if current_distance_to_goal < average_past_distance:
                distance_reward = 5
            else:
                distance_reward = -3
            distance_weight = 1
            angle_weight = 0.5
        else:
            if current_distance_to_goal < average_past_distance or dot_product > 0:
                distance_reward = 2
            else:
                distance_reward = -5
            angle_weight = 1
            distance_weight = 0.5
        reward = distance_reward * distance_weight + angle_reward * angle_weight
        return reward

    def check_stop_condition_west(self, diff_x, diff_y, theta):
        """Check stopping condition for west parking area."""
        theta_radian = math.radians(theta)
        x_condition = -7 <= diff_x <= -4.0
        y_condition = 5.0 <= diff_y <= 7.5
        theta_condition = 1.45 <= theta_radian <= 1.71
        return x_condition and y_condition and theta_condition

    def calculate_reward_west(self, velocity, diff_y, vehicle_yaw, park_yaw, last_diff_y, diff_x, last_diff_x, at_parking_point, vehicle_transform, target_location, d_val_1=8, d_val_2=15.3):
        """Calculate reward for west parking area."""
        current_distance_to_goal = self.current_distance_to_goal
        last_distance_to_goal = self.last_distance_to_goal
        if len(self.past_distance) < self.num_distance:
            return 0
        average_past_distance = sum(self.past_distance) / len(self.past_distance)
        vehicle_forward_vector = np.array([math.cos(vehicle_transform.rotation.yaw), math.sin(vehicle_transform.rotation.yaw)])
        to_target_vector = np.array([target_location.x - vehicle_transform.location.x, (-target_location.y) - (-vehicle_transform.location.y)])
        dot_product = np.dot(vehicle_forward_vector, to_target_vector)
        theta = vehicle_yaw - park_yaw
        theta = math.radians(theta)
        angle_reward = 0
        angle_weight = 0.5
        distance_reward = 0
        distance_weight = 0
        optimal_direction = np.pi / 2
        theta_diff = theta - optimal_direction
        if 0.05 <= theta_diff < 0.12:
            angle_reward = 5
        elif -0.05 <= theta_diff < 0.05:
            angle_reward = 10
        elif -0.16 <= theta_diff < -0.05:
            angle_reward = 5
        elif -0.5 <= theta_diff < -0.16:
            angle_reward = -10
        else:
            angle_reward = -15
        if 5.5 <= current_distance_to_goal <= d_val_1:
            if -7.1 <= diff_x <= -4 or 6 <= diff_y <= 8.5:
                distance_reward = 15
                if -7.1 <= diff_x <= -4 and 6 <= diff_y <= 8.5:
                    distance_reward = 20
            else:
                distance_reward = -5
            distance_weight = 1
            angle_weight = 1.0
        elif 4 <= current_distance_to_goal <= 5.5:
            distance_reward = 10
            distance_weight = 1
            angle_weight = 1.0
        elif d_val_1 < current_distance_to_goal <= d_val_2:
            distance_reward = 5
            if current_distance_to_goal <= average_past_distance:
                distance_reward = 10
            else:
                distance_reward = -3
            distance_weight = 1
            angle_weight = 1.0
        else:
            if current_distance_to_goal <= average_past_distance or abs(current_distance_to_goal - average_past_distance) <= 0.1:
                distance_reward = 5
            else:
                distance_reward = -5
            angle_weight = 0.5
            distance_weight = 1.0
        reward = distance_reward * distance_weight + angle_reward * angle_weight
        if velocity <= 0.5:
            reward = reward * -0.5
        elif velocity >= 6:
            reward = reward - 5
        return reward


    def check_alignment(self, diff_x, diff_y):
        """Check if the vehicle is aligned with the parking spot."""
        x_at_correct_position = abs(diff_x) <= 0.5
        y_at_correct_position = abs(diff_y) <= 0.5
        return x_at_correct_position, y_at_correct_position

    def update_diff(self):
        """Update the difference between vehicle and target position."""
        vehicle_transform = self.ego_vehicle.get_transform()
        target_location = self.target_park['location']
        vehicle_location = vehicle_transform.location
        current_diff_x = vehicle_location.x - target_location.x
        current_diff_y = abs(vehicle_location.y) - abs(target_location.y)
        return current_diff_x, current_diff_y

    def if_parked(self, current_location_x, current_location_y, target_location_x, target_location_y):
        """Check if the vehicle is parked within tolerance."""
        x_parked = math.isclose(current_location_x, target_location_x, abs_tol=0.8)
        y_parked = math.isclose(current_location_y, abs(target_location_y), abs_tol=0.5)
        return x_parked, y_parked

    def destroy_car(self):
        """Destroy obstacle vehicle if it exists."""
        if self.vehicle_exits:
            self.audi_actor.destroy()
            time.sleep(0.5)
            self.vehicle_exits = False
    #   PF module _boundary
    def get_sector_boundary_points(self, x_p, y_p, R_max, R_min, w):
        """Define boundary points for the safety sector."""
        upper_boundary_part1 = []
        lower_boundary_part1 = []
        y_vals_1 = np.linspace(y_p - w, y_p + w, num=30)
        for y_r in y_vals_1:
            x_max = x_p + np.sqrt(np.abs((R_max + w)**2 - (y_r - y_p + R_max)**2))
            x_min = x_p
            upper_boundary_part1.append(Point32(-x_max, -y_r, 0))
            lower_boundary_part1.append(Point32(-x_min, -y_r, 0))
        upper_boundary_part2 = []
        lower_boundary_part2 = []
        y_vals_2 = np.linspace(y_p - R_min, y_p - w, num=30)
        for y_r in y_vals_2:
            x_max = x_p + np.sqrt(np.abs((R_max + w)**2 - (y_r - y_p + R_max)**2))
            x_min = x_p + np.sqrt(np.abs((R_min - w)**2 - (y_r - y_p + R_min)**2))
            upper_boundary_part2.append(Point32(-x_max, -y_r, 0))
            lower_boundary_part2.append(Point32(-x_min, -y_r, 0))
        upper_boundary_part3 = []
        lower_boundary_part3 = []
        y_vals_3 = np.linspace(y_p - R_max, y_p - R_min, num=30)
        x_max_3 = x_p + np.sqrt(np.abs((R_max + w)**2 - (y_vals_3[-1] - y_p + R_max)**2))
        x_min_3 = x_p + R_min
        for y_r in y_vals_3:
            x_max = x_p + np.sqrt(np.abs((R_max + w)**2 - (y_r - y_p + R_max)**2))
            upper_boundary_part3.append(Point32(-x_max, -y_r, 0))
            lower_boundary_part3.append(Point32(-x_min_3, -y_r, 0))
        lower_boundary_part3[0] = lower_boundary_part2[0]
        return (upper_boundary_part1, lower_boundary_part1), (upper_boundary_part2, lower_boundary_part2), (upper_boundary_part3, lower_boundary_part3)
    #publish PF module 
    def publish_polygon(self):
        """Publish the safety polygon area."""
        polygon_msg = PolygonArray()
        polygon_msg.header.frame_id = "map"
        polygon_msg.header.stamp = rospy.Time.now()
        R_max, R_min, w = 3.8, 3.6, 0.2
        part1, part2, part3 = self.get_sector_boundary_points(-self.target_x, self.target_y, self.Rma, R_min, w)
        for idx, (upper_boundary, lower_boundary) in enumerate([part1, part2, part3]):
            safety_area = PolygonStamped()
            safety_area.header.frame_id = "map"
            safety_area.header.stamp = rospy.Time.now()
            full_boundary = upper_boundary + lower_boundary[::-1]
            if idx == 2:
                full_boundary = upper_boundary + [lower_boundary[0]]
            for point in full_boundary:
                point_msg = Point32()
                point_msg.x = point.x
                point_msg.y = point.y
                point_msg.z = point.z
                safety_area.polygon.points.append(point_msg)
            polygon_msg.polygons.append(safety_area)
            polygon_msg.labels.append(idx)
        self.safety_polygon_pub.publish(polygon_msg)
 
    def is_within_sector(self, x_r, y_r, x_p, y_p, R_max, R_min, w):
        """Check if a point is within the safety sector."""
        if y_p - w < y_r < y_p + w:
            x_min = x_p
            x_max = x_p + np.sqrt((R_max + w)**2 - (y_r - y_p + R_max)**2)
            if x_min <= x_r <= x_max:
                return True
        elif y_p - R_min < y_r < y_p - w:
            x_min = x_p + np.sqrt((R_min - w)**2 - (y_r - y_p + R_min)**2)
            x_max = x_p + np.sqrt((R_max + w)**2 - (y_r - y_p + R_max)**2)
            if x_min <= x_r <= x_max:
                return True
        elif y_p - R_max <= y_r <= y_p - R_min:
            x_min = x_p + R_min - w
            x_max = x_p + np.sqrt((R_max + w)**2 - (y_r - y_p + R_max)**2)
            if x_min <= x_r <= x_max:
                return True
        return False

    def calculate_reverse(self, current_x, current_y, theta_degree, x_r, y_r):
        """Calculate reward for reverse parking."""
        start_time = rospy.get_time()
        theta_radian = math.radians(theta_degree)
        xt_diff = abs(abs(current_x) - (abs(self.target_location.x)))
        yt_diff = abs((abs(current_y) - abs(self.target_location.y)))
        distance_diff = math.sqrt(xt_diff ** 2 + yt_diff ** 2)
        x_parked, y_parked = self.if_parked(current_x, current_y, self.target_location.x, self.target_location.y)
        d_reward = 0
        self.angle_history.append(theta_degree)
        if len(self.angle_history) > 25:
            self.angle_history.pop(0)
        angle_penalty = 0
        if len(self.angle_history) == 25:
            all_angle_valid = [abs(self.angle_history[i] - self.angle_history[i - 1]) for i in range(1, len(self.angle_history))]
            if all(change <= 5 for change in all_angle_valid) and not (110 <= abs(theta_degree) <= 180):
                angle_penalty = -20
        if -90 <= theta_degree <= 90:
            angle_reward = -math.cos(math.radians(theta_degree)) * 20
        else:
            angle_reward = abs(math.cos(math.radians(theta_degree))) * 20
        x_reward = 15 * (1 / (xt_diff + 0.001))
        y_reward = 1 / (yt_diff - 0.2 + 0.001)
        distance_reward = math.sqrt(x_reward ** 2 + y_reward ** 2)
        if 5 <= distance_diff < 8.5:
            ex_reward = (distance_reward * 0.1) * (angle_reward * 0.9)
        elif 0 <= distance_diff < 5:
            ex_reward = (distance_reward * 0.4) * (angle_reward * 0.6)
        else:
            ex_reward = (distance_reward * 0.5) * (angle_reward * 0.5)
        if yt_diff <= 0.01:
            d_reward = -20
        reward_suceess = 50 if x_parked and y_parked else 0
        sector_reward = 20 if self.is_within_sector(x_r, y_r, -self.target_location.x, abs(self.target_location.y), self.Rma, self.Rmin, 0.05) else -10
        reward = ex_reward + reward_suceess + d_reward + angle_penalty + sector_reward
        return reward

    def step(self, action):
        """Execute one step in the environment."""
        self.publish_gear()
        done = False
        reward = 0
        reward_stage1 = 0
        reward_stage2 = 0
        if hasattr(self, 'current_distance_to_goal'):
            self.last_distance_to_goal = self.current_distance_to_goal
        else:
            self.last_distance_to_goal = None
        current_vehicle_transform = self.ego_vehicle.get_transform()
        current_vehicle_location = current_vehicle_transform.location
        current_x = current_vehicle_location.x
        current_y = -current_vehicle_location.y
        target_park_rotation = self.target_park['rotation']
        self.target_location = self.target_park['location']
        vehicle_yaw = -current_vehicle_transform.rotation.yaw
        park_yaw = target_park_rotation.yaw
        theta = vehicle_yaw - park_yaw
        self.current_angle = theta
        if self.last_angle is None:
            self.last_angle = self.current_angle
        _, _, diff_x, diff_y, xr, yr = self.calculate_reverse_steering1(current_x, current_y, theta)
        dx = abs(current_x - self.target_location.x)
        _, north_close_distance, _ = self.get_bev_img_obs()
        north_close_distance = min(north_close_distance, 5.2) if north_close_distance is not None else 5.2
        self.Rma = north_close_distance + dx
        self.current_distance_to_goal = math.sqrt(diff_x ** 2 + diff_y ** 2)
        self.past_distance.append(self.current_distance_to_goal)
        augular_velocity = self.ego_vehicle.get_angular_velocity()
        linear_velocity = self.ego_vehicle.get_velocity()
        vehicle_angular_velocity = math.sqrt((augular_velocity.x) ** 2 + (augular_velocity.y) ** 2 + (augular_velocity.z) ** 2)
        vehicle_linear_velocity = math.sqrt(linear_velocity.x ** 2 + linear_velocity.y ** 2 + linear_velocity.z ** 2)
        if len(self.past_distance) > self.num_distance:
            self.past_distance.pop(0)
        nearest_area = self.find_nearest_parking_area(diff_x, diff_y)
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        if self.reverse:
            carla_control_msg = CarlaEgoVehicleControl()
            carla_control_msg.steer = action[0]
            carla_control_msg.reverse = True
            if action[1] >= 0:
                carla_control_msg.throttle = action[1] * 0.3
            else:
                carla_control_msg.throttle = 0.0
                carla_control_msg.brake = 0.0
            self.control_pub.publish(carla_control_msg)
            self.publish_polygon()
            start_time = rospy.get_time()
            correct_position_stagnating, off_position_stagnating = self.check_non_movement(self.current_distance_to_goal, self.last_distance_to_goal)
            if rospy.get_time() - start_time > 35:
                done = True
            elif self.collision_with_actor is not None:
                done = True
            elif off_position_stagnating:
                reward_stage1 = -30
                done = True
            else:
                reward_stage1 = self.calculate_reverse(current_x, current_y, theta, xr, yr)
                x_park, y_park = self.if_parked(current_x, current_y, self.target_location.x, self.target_location.y)
                if x_park and y_park:
                    reward_stage1 = 50
                    done = True
                else:
                    done = False
            time.sleep(0.1)
        else:
            at_parking_point = self.check_stop_condition_west(diff_x, diff_y, theta)
            if at_parking_point:
                carla_control_msg = CarlaEgoVehicleControl()
                carla_control_msg.reverse = True
                self.control_pub.publish(carla_control_msg)
                self.reverse = True
            carla_control_msg = CarlaEgoVehicleControl()
            carla_control_msg.steer = action[0]
            carla_control_msg.reverse = False
            if action[1] >= 0:
                carla_control_msg.throttle = action[1]
                carla_control_msg.brake = 0.0
            else:
                carla_control_msg.throttle = 0.0
                carla_control_msg.brake = abs(action[1])
            self.control_pub.publish(carla_control_msg)
            self.last_diff_x = diff_x
            self.last_diff_y = diff_y
            time.sleep(0.15)
            current_state_dict = self.get_state()
            distance = current_state_dict['distance_to_goal']
            correct_position_stagnating, off_position_stagnating = self.check_non_movement(self.current_distance_to_goal, self.last_distance_to_goal)
            done = False
            vehicle_transform = self.ego_vehicle.get_transform()
            if self.collision_with_actor is not None:
                done = True
            elif off_position_stagnating:
                reward_stage2 = -30
                done = True
            else:
                reward_stage2 = self.calculate_reward_west(vehicle_linear_velocity * 3.6, diff_y=diff_y, vehicle_yaw=vehicle_yaw, park_yaw=park_yaw, last_diff_y=self.last_diff_y, diff_x=diff_x, last_diff_x=self.last_diff_x, at_parking_point=at_parking_point, vehicle_transform=vehicle_transform, target_location=self.target_location, d_val_1=8.5, d_val_2=10.5)
                if reward_stage2 <= -20:
                    done = True
                else:
                    done = False
                self.last_diff_x = diff_x
                self.last_diff_y = diff_y
                self.car_transform = self.ego_vehicle.get_transform().location
                self.vehicle_yaw = vehicle_yaw
                self.park_yaw = park_yaw
        if done:
            self.reset()
        reward = reward_stage1 + reward_stage2
        return reward, done, {}

    def run(self):
        """Run the parking environment simulation."""
        running = True
        episode = 0
        index = 0
        while not rospy.is_shutdown():
            self.get_state()
            index += 1
            action = np.array([(1-(-1)) * np.random.random() + (-1), (1-(-1)) * np.random.random() + (-1)])
            next_state, reward, done = self.step(action)
            state = next_state
            if done:
                episode += 1
                self.reset()

if __name__ == "__main__":
    env = ParkingEnv()
    env.run()#!/usr/bin/env python3.8
import csv
import numpy as np
import subprocess
import torch
import os
import random
import math
import carla
import time
import cv2
import rospy
from std_msgs.msg import String
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl, CarlaCollisionEvent, CarlaEgoVehicleInfo
from tf import transformations
from skimage.transform import resize
from std_msgs.msg import UInt32
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Point32, PolygonStamped, Pose, PoseStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan, Image, Imu
from derived_object_msgs.msg import ObjectArray
from jsk_recognition_msgs.msg import PolygonArray
import message_filters
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header

# Configuration constants
_HOST_ = '127.0.0.1'
_PORT_ = 2000
_SLEEP_TIME_ = 0.5

TRAINING_INDICATOR = 2
SELECTED_MODEL = 'only_throttle'
TRAINING_NAME = 'training'
SELECTED_SPAWNING_METHOD = 0

ACTIONS_SIZE = 1
STATE_SIZE = 16

MAX_COLLISION_IMPULSE = 50
MAX_DISTANCE = 15.7
MAX_REWARD = 20.0
SIGMA = 2.0

# Training parameters
MEMORY_FRACTION = 0.3333
TOTAL_EPISODES = 1000
STEPS_PER_EPISODE = 100
AVERAGE_EPISODES_COUNT = 40
CORRECT_POSITION_NON_MOVING_STEPS = 5
OFF_POSITION_NON_MOVING_STEPS = 100
REPLAY_BUFFER_CAPACITY = 100000
BATCH_SIZE = 64
CRITIC_LR = 0.002
ACTOR_LR = 0.001
GAMMA = 0.99
TAU = 0.005
epsilon = 1
EXPLORE = 100000.0
MIN_EPSILON = 0.000001

class ParkingEnv():
    def __init__(self):
        rospy.init_node("park")
        # Publisher for gear data
        self.gear_pub = rospy.Publisher("/gear", String, queue_size=10)
        self.predefined_locations = [
            carla.Transform(carla.Location(x=-1.6, y=-12.6, z=0.15), carla.Rotation(pitch=0, yaw=-90, roll=0))
        ]
        self.parking_spots = {
            "east_park": {"location": carla.Location(x=17, y=-20, z=0.05), "rotation": carla.Rotation(pitch=0, yaw=268, roll=0), "check_stop_condition": self.check_stop_condition_east, "calculate_reward": self.calculate_reward_east},
            "west_park": {"location": carla.Location(x=-1.6, y=-12.6, z=0.15), "rotation": carla.Rotation(pitch=0, yaw=90, roll=0), "check_stop_condition": self.check_stop_condition_west, "calculate_reward": self.calculate_reward_west}
        }
        self.parking_areas = {
            'east_park': {
                'x_range': (2, 5),
                'y_range': (2, 5),
                'theta_range': (1.221, 1.7453),
                'center': ((2 + 5)/2, (2 + 5)/2)
            },
            'west_park': {
                'x_range': (3, 5.1),
                'y_range': (-3.5, -1.7),
                'theta_range': (-1.9, -1.221),
                'center': ((3 + 5.1)/2, (-3.5 + -1.7)/2)
            }
        }
        self.car_start_transform = None
        self.park_list = {}
        # Initialize CARLA client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.parking_point_checked = False
        self.vehicle_exits = False
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.world.apply_settings(settings)
        self.audi_actor = None
        self.obstacle_active = False
        self.flag1 = 1
        self.model_3_heigth = 1.443
        self.width = 2.089
        self.height = 4.69
        self.index = 0
        self.ego_location = None
        self.ego_heading = None
        self.ego_speed = None
        self.surrounding_objs = None
        self.ego_odom = None
        self.ego_vehicle_status = None
        self.cad_ranges = None
        self.bev_range = [-22, -22, 24, 22]
        self.bev_img_size = [512, 512]
        self.next_wpt = None
        self.last_diff_y = float('inf')
        self.last_diff_x = float('inf')
        self.diff_y = None
        self.diff_x = None
        self.kp = 30
        self.last_distance_to_goal = 0
        self.current_distance_to_goal = 0
        self.past_distance = []
        self.angle_history = []
        self.num_distance = 5
        self.obstacle_vehicle = False
        self.obstacle_moving = False
        self.reverse = False
        self.last_angle = 0
        self.current_angle = 0
        ego_vehicle_info = rospy.wait_for_message("/carla/ego_vehicle/vehicle_info", CarlaEgoVehicleInfo)
        self.ego_id = ego_vehicle_info.id
        self.ego_vehicle = self.world.get_actor(ego_vehicle_info.id)
        self.current_select_area = None
        self.car_transform = self.ego_vehicle.get_transform()
        gen_park_flag = self.gen_park_list()
        if not gen_park_flag:
            print("Failed to generate park list")
            return False
        self.target_park = self.park_list["park_15"]
        self.target_x = self.park_list["park_15"]["location"].x
        self.target_y = self.park_list["park_15"]["location"].y
        self.target_width = self.park_list["park_15"]["width"]
        self.target_height = self.park_list["park_15"]["height"]
        self.park_yaw = self.park_list["park_15"]["rotation"].yaw
        debug = self.world.debug
        print(self.park_list["park_15"]["rotation"])
        debug.draw_box(carla.BoundingBox(self.park_list["park_15"]["location"], carla.Vector3D(self.target_height/2, self.target_width/2, 0.1)), self.park_list["park_15"]["rotation"], 0.05, carla.Color(255,0,0,0), 0)
        print("Final parking spot:", self.target_x, self.target_y)
        print("Final parking spot dimensions:", self.target_width, self.target_height)
        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(self.target_park["location"] + carla.Location(z=20), carla.Rotation(pitch=-90)))
        self.bev_obs_size = [128, 128]
        x_range = self.bev_range[2] - self.bev_range[0]
        y_range = self.bev_range[3] - self.bev_range[1]
        ego_vehicle_dimensions = [2.52, 1.47]
        target_velocity = 10
        self.ego_anchor_pixel = [int(self.bev_img_size[1] * self.bev_range[3] / y_range - 1), int(self.bev_img_size[0] * self.bev_range[2] / x_range - 1)]
        self.ego_vehicle_dimensions = ego_vehicle_dimensions
        self.run_off_dis_threshold = 2
        self.angle_diff_max = 7
        self.target_velocity = target_velocity
        self.min_velocity = 2
        self.max_velocity = 10
        self.voxel_size = [(self.bev_range[2]-self.bev_range[0])/self.bev_img_size[0], (self.bev_range[3]-self.bev_range[1])/self.bev_img_size[1]]
        self.invalid_stop_frames = 0
        self.collision_with_actor = None
        self.crossed_lane_markings = None
        self.hm = 2.875
        self.hp = 6
        self.h = 3
        self.Rmin = 3.6
        self.bridge = CvBridge()
        self.zero_control_time = 0
        self.time_step = 0.05
        # Subscribers
        self.odom_sub = rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, self.odometry_callback)
        odom_sub = message_filters.Subscriber("/carla/ego_vehicle/imu", Imu)
        vehicle_status_sub = message_filters.Subscriber("/carla/ego_vehicle/vehicle_status", CarlaEgoVehicleStatus)
        objs_sub = message_filters.Subscriber("/carla/ego_vehicle/objects", ObjectArray)
        cad_sub = message_filters.Subscriber("/cad_carla_gt", LaserScan)
        self.sys_synchronizer = message_filters.ApproximateTimeSynchronizer([odom_sub, vehicle_status_sub, objs_sub, cad_sub], queue_size=10, slop=0.1)
        self.sys_synchronizer.registerCallback(self.sys_callback_api)
        collision_sub = rospy.Subscriber("/carla/ego_vehicle/collision", CarlaCollisionEvent, self.collision_callback)
        self.safety_polygon_pub = rospy.Publisher("/safety_area", PolygonArray, queue_size=2)
        # Publishers
        self.global_goal_pub = rospy.Publisher("/carla/ego_vehicle/goal", PoseStamped, latch=True, queue_size=5)
        self.control_pub = rospy.Publisher("/carla/ego_vehicle/vehicle_control_cmd", CarlaEgoVehicleControl, latch=True, queue_size=10)
        self.cad_render_pub = rospy.Publisher("/bev_perception", Image, queue_size=10)
        self.park_pub = rospy.Publisher("/map/parklist", MarkerArray, queue_size=10)
        self.global_path = rospy.Publisher("/vehicle_path", Path, queue_size=10)
        self.vehicle_path = Path()
        self.vehicle_path.header.frame_id = "map"
        self.learning_steps_threshold = 100

    def odometry_callback(self, odom_msg):
        """Callback to handle odometry data and publish vehicle path."""
        position = odom_msg.pose.pose.position
        orientation = odom_msg.pose.pose.orientation
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position.x = position.x
        pose_stamped.pose.position.y = position.y
        pose_stamped.pose.position.z = position.z
        pose_stamped.pose.orientation = orientation
        self.vehicle_path.poses.append(pose_stamped)
        self.vehicle_path.header.stamp = rospy.Time.now()
        self.global_path.publish(self.vehicle_path)

    def publish_gear(self):
        """Publish gear state (R for reverse, D for drive)."""
        gear_msg = "R" if self.reverse else "D"
        self.gear_pub.publish(gear_msg)

    def clear_vehicles(self):
        """Destroy all vehicle actors in the simulation."""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        vehicle_actors = self.world.get_actors().filter("vehicle.*")
        for vehicle in vehicle_actors:
            vehicle.destroy()
            rospy.loginfo(f"Vehicle {vehicle.id} destroyed")
            self.world.tick()
            time.sleep(0.1)
            try:
                self.world.tick()
            except Exception as e:
                print("Error during destruction:", e)
            time.sleep(0.5)

    def sys_callback_api(self, imu_msg: Imu, vehicle_status_msg: CarlaEgoVehicleStatus, objs_msg: ObjectArray, cad_msg: LaserScan):
        """Callback to process system messages and update ego vehicle state."""
        ego_loc = self.ego_vehicle.get_location()
        ego_pos = np.array([ego_loc.x, -ego_loc.y])
        surrounding_objs = [obj for obj in objs_msg.objects if np.linalg.norm(ego_pos - np.array([obj.pose.position.x, obj.pose.position.y])) < 30]
        self.surrounding_objs = surrounding_objs
        self.ego_odom = Odometry()
        self.ego_odom.pose.pose.position.x = ego_loc.x
        self.ego_odom.pose.pose.position.y = -ego_loc.y
        self.ego_odom.pose.pose.position.z = ego_loc.z
        self.ego_odom.pose.pose.orientation.x = vehicle_status_msg.orientation.x
        self.ego_odom.pose.pose.orientation.y = vehicle_status_msg.orientation.y
        self.ego_odom.pose.pose.orientation.z = vehicle_status_msg.orientation.z
        self.ego_odom.pose.pose.orientation.w = vehicle_status_msg.orientation.w
        self.ego_odom.twist.twist.linear.x = vehicle_status_msg.velocity
        self.ego_odom.twist.twist.angular.z = imu_msg.angular_velocity.z
        self.ego_vehicle_status = vehicle_status_msg
        self.cad_ranges = cad_msg.ranges
        self.ego_location = np.array([self.ego_odom.pose.pose.position.x, self.ego_odom.pose.pose.position.y, self.ego_odom.pose.pose.position.z])
        self.ego_heading = np.array([self.ego_odom.pose.pose.orientation.x, self.ego_odom.pose.pose.orientation.y, self.ego_odom.pose.pose.orientation.z, self.ego_odom.pose.pose.orientation.w])
        self.ego_speed = [self.ego_odom.twist.twist.linear.x, self.ego_odom.twist.twist.angular.z]

    def collision_callback(self, collision_msg: CarlaCollisionEvent):
        """Callback to handle collision events."""
        self.collision_with_actor = collision_msg.other_actor_id

    def transform_angle(self, angle):
        """Convert angle from -180 to 180 degrees to 0 to 360 degrees."""
        return 360 + angle if angle < 0 else angle

    def get_state(self):
        """Retrieve the current state of the ego vehicle."""
        current_vehicle_transform = self.ego_vehicle.get_transform()
        current_vehicle_location = current_vehicle_transform.location
        current_vehicle_x = current_vehicle_location.x
        current_vehicle_y = current_vehicle_location.y
        angle = self.transform_angle(current_vehicle_transform.rotation.yaw)
        current_vehicle_linear_velocity = self.ego_vehicle.get_velocity().x
        current_vehicle_angular_velocity = self.ego_vehicle.get_angular_velocity().z
        x = current_vehicle_x
        y = current_vehicle_y
        x_rel = self.target_park["location"].x - current_vehicle_x
        y_rel = self.target_park["location"].y - current_vehicle_y
        angle = self.transform_angle(angle)
        vx = current_vehicle_linear_velocity
        wz = current_vehicle_angular_velocity
        distance_to_goal = current_vehicle_location.distance(self.target_park['location'])
        sensor_values_dict = {
            'x': x,
            'y': y,
            'x_rel': x_rel,
            'y_rel': y_rel,
            'angle': angle,
            'vx': vx,
            'wz': wz,
            'distance_to_goal': distance_to_goal,
        }
        return sensor_values_dict

    def get_cad_bev(self):
        """Convert CAD perception into a BEV image."""
        if self.cad_ranges is None:
            rospy.loginfo("No CAD perception!")
            return
        adjusted_height = self.bev_img_size[0]
        bev_img = np.zeros((adjusted_height, self.bev_img_size[1], 3), dtype=np.uint8)
        cad_points = []
        closest_north_distance = float('inf')
        closest_north_angle = None
        north_area = self.target_park['rotation'].yaw
        range_points = 200
        angle_step = 2 * np.pi / 384
        current_idx = int((north_area + np.pi) / angle_step)
        left_idx = (current_idx - range_points) % 384
        right_idx = (current_idx + range_points) % 384
        for idx in range(len(self.cad_ranges)):
            r = self.cad_ranges[idx] / self.voxel_size[0]
            r1 = self.cad_ranges[idx]
            theta = (0.5 + idx) * angle_step - np.pi
            x = -r * np.sin(theta)
            y = -r * np.cos(theta)
            cad_points.append([x, y])
            if (left_idx <= idx <= right_idx) or (left_idx > right_idx and (idx >= left_idx or idx <= right_idx)):
                if r1 < closest_north_distance:
                    closest_north_distance = r1
                    closest_north_angle = theta
        cad_points = np.array(cad_points, dtype=np.int32)
        cad_points[:, 0] = cad_points[:, 0] + self.ego_anchor_pixel[0]
        cad_points[:, 1] = cad_points[:, 1] + self.ego_anchor_pixel[1]
        cv2.drawContours(bev_img, [cad_points], -1, (255, 255, 255), -1)
        resized_bev_img = cv2.resize(bev_img, (512, 512), interpolation=cv2.INTER_AREA)
        print("=====cad_bev_img", resized_bev_img.shape)
        print("=====closest_north_distance bang", closest_north_distance)
        print("=====closest_north_angle", closest_north_angle)
        cad_bev_img_msg = self.bridge.cv2_to_imgmsg(resized_bev_img, encoding="bgr8")
        self.cad_render_pub.publish(cad_bev_img_msg)
        return bev_img, closest_north_distance, closest_north_angle

    def get_bev_img_obs(self):
        """Generate BEV image observation for RL training."""
        print("====generate bev image time", time.time())
        cad_bev_img, closest_north_distance, closest_north_angle = self.get_cad_bev()
        if cad_bev_img is None or cad_bev_img.size == 0:
            print("Error: CAD BEV is empty")
            return None, None, None
        cad_bev_img_s = cv2.cvtColor(cad_bev_img, cv2.COLOR_RGB2GRAY)
        assert cad_bev_img_s.shape[:2] == (512, 512)
        stacked_img = np.zeros((512, 512, 2), dtype=np.float32)
        stacked_img[:, :, 0] = cad_bev_img_s
        bev_resized = cv2.resize(stacked_img, (self.bev_obs_size[0], self.bev_obs_size[1]))
        bev_resized = bev_resized / 255.0
        bev = np.transpose(bev_resized, (2, 0, 1))
        print("====generate bev image time", time.time())
        return bev, closest_north_distance, closest_north_angle

    def get_propriceptive_obs(self):
        """Retrieve proprioceptive observations of the ego vehicle."""
        if self.ego_vehicle_status is not None and self.ego_vehicle_status.control is not None:
            real_time_throttle = self.ego_vehicle_status.control.throttle
        real_time_steer = self.ego_vehicle_status.control.steer
        real_time_linear_vel = self.ego_odom.twist.twist.linear.x
        real_time_angular_vel = self.ego_odom.twist.twist.angular.z
        current_vehicle_transform = self.ego_vehicle.get_transform()
        _, _, ego_yaw = transformations.euler_from_quaternion(self.ego_heading)
        wpt_yaw = self.target_park["rotation"].yaw
        real_time_orientation_diff = (wpt_yaw - ego_yaw) * 180 / np.pi
        distance_to_goal = current_vehicle_transform.location.distance(self.target_park['location'])
        print("-----------------------real_time_orientation_diff: %s" % real_time_orientation_diff)
        print("-----------------------distance_to_goal: %s" % distance_to_goal)
        proprioceptive_obs = np.array([real_time_throttle, real_time_steer, real_time_linear_vel, real_time_angular_vel, distance_to_goal, real_time_orientation_diff])
        return self.ego_location, self.ego_heading, self.ego_speed, proprioceptive_obs

    def find_nearest_parking_area(self, diff_x, diff_y):
        """Determine the nearest parking area based on vehicle's position."""
        if self.current_select_area is not None:
            print(f"Continue current area: {self.current_select_area}")
            return self.current_select_area
        max_distance = -1
        for area_name, area_props in self.parking_areas.items():
            dx = diff_x - area_props['center'][0]
            dy = diff_y - area_props['center'][1]
            distance = math.sqrt(dx**2 + dy**2)
            print(f"area: {area_name}, distance: {distance}")
            if distance > max_distance:
                max_distance = distance
                nearest_park_area = area_name
        print(f"Currently selected parking area: {nearest_park_area}")
        return nearest_park_area

    def get_localtion_info(self):
        """Get the current location of the vehicle."""
        current_vehicle_location = self.ego_vehicle.get_transform().location
        print("before vehicle_location", current_vehicle_location)
        return current_vehicle_location.x, current_vehicle_location.y

    def reset_car(self):
        """Reset the vehicle's position until it is stationary."""
        vehicle_position_stable = False
        while not vehicle_position_stable:
            if self.ego_vehicle:
                self.ego_vehicle.destroy()
                self.ego_vehicle = None
            time.sleep(1)
            if self.vehicle_is_stationary():
                vehicle_position_stable = True
                print("Vehicle is stable")
            else:
                print("Vehicle is not stable")

    def vehicle_is_stationary(self):
        """Check if the vehicle is stationary based on velocity and acceleration."""
        velocity = self.ego_vehicle.get_velocity()
        acceleration = self.ego_vehicle.get_acceleration()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        accel = math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
        return speed < 0.2 and accel < 0.2

    def reset(self):
        """Reset the environment to initial conditions."""
        print("start reset=======")
        self.ego_vehicle.apply_control(carla.VehicleControl(brake=True))
        rospy.sleep(0.3)
        self.ego_vehicle.apply_control(carla.VehicleControl(brake=False))
        self.ego_vehicle.apply_control(carla.VehicleControl(reverse=False))
        self.car_start_transform = random.choice(self.predefined_locations)
        self.ego_vehicle.set_transform(self.car_start_transform)
        self.reverse = False
        self.flag1 = 1
        self.zero_control_time = 0
        self.parking_point_checked = False
        print(f"Reset car to {self.car_start_transform}")
        print(f"location", self.car_start_transform.location)
        self.collision_with_actor = None
        current_state_dict = self.get_state()
        distance = current_state_dict['distance_to_goal']
        angle = current_state_dict['angle']
        self.distance_to_goal = None
        self.last_distance_to_goal = self.distance_to_goal
        self.distance_to_goal = distance
        self.angle = None
        self.last_angle = self.angle
        self.angle = angle
        self.non_moving_steps_cnt = 0
        current_vehicle_location = self.car_start_transform.location
        current_vehicle_rotation = self.car_start_transform.rotation
        current_x = current_vehicle_location.x
        current_y = -current_vehicle_location.y
        print("Current vehicle position", current_x, current_y)
        target_park_rotation = self.target_park['rotation']
        self.target_location = self.target_park['location']
        vehicle_yaw = -current_vehicle_rotation.yaw
        park_yaw = target_park_rotation.yaw
        theta = vehicle_yaw - park_yaw
        R, steer, diff_x, diff_y, xr, yr = self.calculate_reverse_steering1(current_x, current_y, theta)
        self.current_select_area = None
        nearest_area = self.find_nearest_parking_area(diff_x, diff_y)
        self.current_select_area = nearest_area
        print("Initialize current_select_area information")
        self.get_weight_path_by_area()
        print("!!current_select_area", self.current_select_area)

    def gen_park_list(self):
        """Generate a list of parking spots based on road lines."""
        roadline_list = self.world.get_environment_objects(carla.CityObjectLabel.RoadLines)
        self.all_park_roadline = {item.name: item for item in roadline_list if item.name.find("Plane") > -1}
        park_line_name = [f"Plane{line_index}_SM_0" for line_index in range(9, 81)]
        print("Expected park line names:", park_line_name)
        park_raodline = {item.name: item for item in roadline_list if item.name in park_line_name}
        print("Actual park roadlines found:", park_raodline.keys())
        self.park_list = {}
        for park_index in range(9, 80):
            left_park_name = f"Plane{park_index}_SM_0"
            right_park_name = f"Plane{park_index + 1}_SM_0"
            if left_park_name not in park_raodline or right_park_name not in park_raodline:
                print(f"Skipping missing Plane at index: {park_index}")
                continue
            temp = {}
            temp_park_line_left = park_raodline[left_park_name]
            temp_park_line_right = park_raodline[right_park_name]
            print(park_index)
            print(temp_park_line_left.bounding_box.rotation)
            center_location = carla.Location()
            center_location.x = (temp_park_line_left.bounding_box.location.x + temp_park_line_right.bounding_box.location.x) / 2
            center_location.y = (temp_park_line_left.bounding_box.location.y + temp_park_line_right.bounding_box.location.y) / 2
            center_location.z = (temp_park_line_left.bounding_box.location.z + temp_park_line_right.bounding_box.location.z) / 2
            center_rotation = carla.Rotation(yaw=0)
            temp_width = ((temp_park_line_left.bounding_box.location.x - temp_park_line_right.bounding_box.location.x) ** 2 + 
                         (temp_park_line_left.bounding_box.location.y - temp_park_line_right.bounding_box.location.y) ** 2) ** 0.5
            temp_height = 5
            temp["location"] = center_location
            temp["rotation"] = center_rotation
            temp["width"] = temp_width
            temp["height"] = temp_height
            park_name = f"park_{park_index - 8}"
            self.park_list[park_name] = temp
            print(f"park_name: {park_name}, location: {center_location}, rotation: {center_rotation}, width: {temp_width}, height: {temp_height}")
        return True

    def get_car_box(self, car_transform):
        """Get the bounding box of the vehicle."""
        rect = ((car_transform.location.x, car_transform.location.y), (self.height, self.width), car_transform.rotation.yaw)
        box = cv2.boxPoints(rect)
        return box

    def check_non_movement(self, current_distance_to_goal, last_distance_to_goal):
        """Check if the vehicle has stopped moving."""
        correct_position_non_movement_indicator = False
        off_position_non_movement_indicator = False
        if abs(last_distance_to_goal - current_distance_to_goal) <= 0.05:
            self.non_moving_steps_cnt += 1
            goal_angle = self.transform_angle(self.target_park['rotation'].yaw)
            vehicle_angle = self.angle
            if self.check_if_parked(goal_angle, vehicle_angle) and (self.non_moving_steps_cnt >= CORRECT_POSITION_NON_MOVING_STEPS):
                correct_position_non_movement_indicator = True
            elif (not self.check_if_parked(goal_angle, vehicle_angle)) and (self.non_moving_steps_cnt >= OFF_POSITION_NON_MOVING_STEPS):
                off_position_non_movement_indicator = True
        else:
            self.non_moving_steps_cnt = 0
        return correct_position_non_movement_indicator, off_position_non_movement_indicator

    def check_if_parked(self, goal_angle, vehicle_angle):
        """Check if the vehicle is parked correctly."""
        vehicle_parked = False
        if (self.current_distance_to_goal <= 0.8) and ((abs(goal_angle - vehicle_angle) <= 20) or (abs(goal_angle - vehicle_angle) >= 160)):
            vehicle_parked = True
        return vehicle_parked

    def calculate_reverse_steering(self, target_x, target_y, theta):
        """Calculate steering parameters for reverse parking (left-front)."""
        xr = target_x - (self.hm * math.cos(theta) / 2)
        yr = target_y - (self.hm * math.sin(theta) / 2)
        diff_x = xr - self.target_park['location'].x
        diff_y = yr - (-self.target_park['location'].y)
        R = (diff_x - self.h) / (math.sin(theta))
        phi = math.atan(self.hm / R)
        if phi > math.pi / 2:
            phi = math.pi - phi
        phi_max = math.atan(self.hm / self.Rmin)
        steer = phi / phi_max
        steer = max(0, min(1, steer))
        return R, steer, diff_x, diff_y

    def calculate_reverse_steering1(self, target_x, target_y, theta):
        """Calculate steering parameters for reverse parking (right-front)."""
        xr = target_x - (self.hm * math.cos(theta) / 2)
        yr = target_y - (self.hm * (-math.sin(theta)) / 2)
        diff_x = xr - self.target_park['location'].x
        diff_y = yr - (-self.target_park['location'].y)
        R = (diff_x - self.h) / (-math.sin(theta) + 0.01)
        phi = math.atan(self.hm / R)
        if phi > math.pi / 2:
            phi = math.pi - phi
        phi_max = math.atan(self.hm / self.Rmin)
        steer = phi / phi_max
        steer = max(-1, min(1, steer))
        return R, steer, diff_x, diff_y, xr, yr

    def check_stop_condition_east(self, diff_x, diff_y, theta):
        """Check stopping condition for east parking area."""
        theta_radian = math.radians(theta)
        x_condition = 4.5 < diff_x < 6.5
        y_condition = 3.5 <= diff_y < 5.0
        theta_condition = 1.221 <= theta_radian <= 1.7453
        return x_condition and y_condition and theta_condition

    def get_current_parking_status(self):
        """Return the current parking area for training."""
        print("Selected area:", self.current_select_area)
        return self.current_select_area

    def get_weight_path_by_area(self):
        """Retrieve the policy path based on the current parking area."""
        policy_path = "/home/vsisauto/xxy/ppo_0116/checkpoints_Town03east1/"
        current_area_name = self.get_current_parking_status()
        if current_area_name == "east_park":
            print("########checkpoints_Town03east")
            policy_path = "/home/vsisauto/xxy/ppo_0116/checkpoints_Town03east1/"
        elif current_area_name == "west_park":
            print("########checkpoints_Town02west")
            policy_path = "/home/vsisauto/xxy/ppo_0116/checkpoints_Town02west/"
        return policy_path

    def calculate_reward_east(self, diff_y, vehicle_yaw, park_yaw, last_diff_y, diff_x, last_diff_x, at_parking_point, vehicle_transform, target_location, d_val_1=6.5, d_val_2=11.5):
        """Calculate reward for east parking area."""
        current_distance_to_goal = self.current_distance_to_goal
        last_distance_to_goal = self.last_distance_to_goal
        if len(self.past_distance) < self.num_distance:
            return 0
        average_past_distance = sum(self.past_distance) / len(self.past_distance)
        vehicle_forward_vector = np.array([math.cos(vehicle_transform.rotation.yaw), math.sin(vehicle_transform.rotation.yaw)])
        to_target_vector = np.array([target_location.x - vehicle_transform.location.x, (-target_location.y) - (-vehicle_transform.location.y)])
        dot_product = np.dot(vehicle_forward_vector, to_target_vector)
        theta = vehicle_yaw - park_yaw
        theta = math.radians(theta)
        angle_reward = 0
        angle_weight = 0.1
        distance_reward = 0.1
        distance_weight = 0
        optimal_direction = np.pi / 2
        theta_diff = theta - optimal_direction
        if -0.15 <= theta_diff < 0.01:
            angle_reward = 5
        elif 0.01 <= theta_diff < 0.05:
            angle_reward = 10
        elif 0.05 <= theta_diff <= 0.39:
            angle_reward = 10
            print("1111111111111111111111111")
        elif 0.35 <= theta_diff < 0.50:
            angle_reward = -20
            print("2222222222222222222222222")
        else:
            angle_reward = -50
            print("33333333333333333333333")
        print("theta", theta)
        print("angle_reward", angle_reward)
        print("theta_diff", theta_diff)
        print("diff_x", diff_x)
        print("diff_y", diff_y)
        print("average_past_distance", average_past_distance)
        print("dot_product", dot_product)
        if 5.5 <= current_distance_to_goal <= d_val_1:
            print("!!!!<7")
            if 5 <= diff_x <= 7 or 3.5 <= diff_y <= 6.0:
                distance_reward = 15
                if 5 <= diff_x <= 7 and 4.3 <= diff_y <= 6:
                    distance_reward = (math.exp(d_val_1 / (current_distance_to_goal + 0.01))) ** 4
            else:
                distance_reward = -20
            distance_weight = 2
            angle_weight = 2.0
        elif d_val_1 < current_distance_to_goal <= d_val_2:
            print("!!!!<10")
            distance_reward = -5
            if current_distance_to_goal < average_past_distance:
                distance_reward = (d_val_2 / (current_distance_to_goal))
            else:
                distance_reward = -((d_val_2 - current_distance_to_goal) ** 2)
            distance_weight = 2
            angle_weight = 1.0
        else:
            print("!!!!>10 or <2")
            if current_distance_to_goal < average_past_distance or dot_product > 0:
                distance_reward += 1.0 + (current_distance_to_goal / d_val_1)
            else:
                distance_reward -= min(20, math.exp(current_distance_to_goal - d_val_2) ** 3)
            angle_weight = 2
            distance_weight = 1
        reward = distance_reward * distance_weight + angle_reward * angle_weight
        print("distance_reward", distance_reward)
        return reward

    def check_stop_condition_west(self, diff_x, diff_y, theta):
        """Check stopping condition for west parking area."""
        theta_radian = math.radians(theta)
        x_condition = -7 <= diff_x <= -4.0
        y_condition = 5.0 <= diff_y <= 7.5
        theta_condition = 1.45 <= theta_radian <= 1.71
        print("theta_radian", theta_radian, "diff_x", diff_x, "diff_y", diff_y)
        print("x_condition, y_condition, theta_condition", x_condition, y_condition, theta_condition)
        return x_condition and y_condition and theta_condition

    def calculate_reward_west(self, velocity, diff_y, vehicle_yaw, park_yaw, last_diff_y, diff_x, last_diff_x, at_parking_point, vehicle_transform, target_location, d_val_1=8, d_val_2=15.3):
        """Calculate reward for west parking area."""
        current_distance_to_goal = self.current_distance_to_goal
        last_distance_to_goal = self.last_distance_to_goal
        if len(self.past_distance) < self.num_distance:
            return 0
        average_past_distance = sum(self.past_distance) / len(self.past_distance)
        vehicle_forward_vector = np.array([math.cos(vehicle_transform.rotation.yaw), math.sin(vehicle_transform.rotation.yaw)])
        to_target_vector = np.array([target_location.x - vehicle_transform.location.x, (-target_location.y) - (-vehicle_transform.location.y)])
        dot_product = np.dot(vehicle_forward_vector, to_target_vector)
        theta = vehicle_yaw - park_yaw
        theta = math.radians(theta)
        angle_reward = 0
        angle_weight = 0.1
        distance_reward = 0.1
        distance_weight = 0
        optimal_direction = np.pi / 2
        theta_diff = theta - optimal_direction
        if 0.05 <= theta_diff < 0.12:
            angle_reward = 20
        elif -0.05 <= theta_diff < 0.05:
            angle_reward = 40
        elif -0.16 <= theta_diff < -0.05:
            angle_reward = 5
        elif -0.5 <= theta_diff < -0.16:
            angle_reward = -15
        else:
            angle_reward = -20
        if 5.5 <= current_distance_to_goal <= d_val_1:
            print("!!!!<5")
            if -7.1 <= diff_x <= -4 or 6 <= diff_y <= 8.5:
                distance_reward = 5 + 20 / (current_distance_to_goal + 0.01)
                if -7.1 <= diff_x <= -4 and 6 <= diff_y <= 8.5:
                    distance_reward = 2 * (math.exp(d_val_1 / (current_distance_to_goal + 0.01))) ** 4
            else:
                distance_reward = -10
            distance_weight = 1
            angle_weight = 2.0
        elif 4 <= current_distance_to_goal <= 5.5:
            print("!!!!<5.5")
            distance_reward = 50 + 20 / (current_distance_to_goal + 0.01)
            distance_weight = 1
            angle_weight = 2.0
        elif d_val_1 < current_distance_to_goal <= d_val_2:
            print("!!!!<10.5")
            distance_reward = 5
            if current_distance_to_goal <= average_past_distance:
                distance_reward =  math.exp(d_val_2 / (current_distance_to_goal)) ** 2
            else:
                distance_reward = -((d_val_2 - current_distance_to_goal / d_val_1) ** 2)
            distance_weight = 1
            angle_weight = 2.0
        else:
            print("!!!!>15.5 or <8.5")
            if current_distance_to_goal <= average_past_distance or abs(current_distance_to_goal - average_past_distance) <= 0.1:
                distance_reward = -current_distance_to_goal 
            else:
                distance_reward = (-min(20, math.exp(current_distance_to_goal - d_val_1) ** 2))
            angle_weight = 1
            distance_weight = 1.5
        print("distance_reward:", distance_reward)
        reward = distance_reward * distance_weight + angle_reward * angle_weight
        if velocity <= 0.5:
            reward = reward * -0.1
        elif velocity >= 6:
            reward = reward - 10
        return reward

    

    def check_alignment(self, diff_x, diff_y):
        """Check if the vehicle is aligned with the parking spot."""
        x_at_correct_position = abs(diff_x) <= 0.5
        y_at_correct_position = abs(diff_y) <= 0.5
        return x_at_correct_position, y_at_correct_position

    def update_diff(self):
        """Update the difference between vehicle and target position."""
        vehicle_transform = self.ego_vehicle.get_transform()
        target_location = self.target_park['location']
        vehicle_location = vehicle_transform.location
        current_diff_x = vehicle_location.x - target_location.x
        current_diff_y = abs(vehicle_location.y) - abs(target_location.y)
        return current_diff_x, current_diff_y

    def if_parked(self, current_location_x, current_location_y, target_location_x, target_location_y):
        """Check if the vehicle is parked within tolerance."""
        x_parked = math.isclose(current_location_x, target_location_x, abs_tol=0.8)
        y_parked = math.isclose(current_location_y, abs(target_location_y), abs_tol=0.5)
        return x_parked, y_parked

    def destroy_car(self):
        """Destroy obstacle vehicle if it exists."""
        if self.vehicle_exits:
            print("Destroying vehicle")
            self.audi_actor.destroy()
            time.sleep(0.5)
            self.vehicle_exits = False
        else:
            print("No vehicle to destroy")

    def get_sector_boundary_points(self, x_p, y_p, R_max, R_min, w):
        """Define boundary points for the safety sector."""
        upper_boundary_part1 = []
        lower_boundary_part1 = []
        y_vals_1 = np.linspace(y_p - w, y_p + w, num=30)
        for y_r in y_vals_1:
            x_max = x_p + np.sqrt(np.abs((R_max + w)**2 - (y_r - y_p + R_max)**2))
            x_min = x_p
            upper_boundary_part1.append(Point32(-x_max, -y_r, 0))
            lower_boundary_part1.append(Point32(-x_min, -y_r, 0))
        upper_boundary_part2 = []
        lower_boundary_part2 = []
        y_vals_2 = np.linspace(y_p - R_min, y_p - w, num=30)
        for y_r in y_vals_2:
            x_max = x_p + np.sqrt(np.abs((R_max + w)**2 - (y_r - y_p + R_max)**2))
            x_min = x_p + np.sqrt(np.abs((R_min - w)**2 - (y_r - y_p + R_min)**2))
            upper_boundary_part2.append(Point32(-x_max, -y_r, 0))
            lower_boundary_part2.append(Point32(-x_min, -y_r, 0))
        upper_boundary_part3 = []
        lower_boundary_part3 = []
        y_vals_3 = np.linspace(y_p - R_max, y_p - R_min, num=30)
        x_max_3 = x_p + np.sqrt(np.abs((R_max + w)**2 - (y_vals_3[-1] - y_p + R_max)**2))
        x_min_3 = x_p + R_min
        for y_r in y_vals_3:
            x_max = x_p + np.sqrt(np.abs((R_max + w)**2 - (y_r - y_p + R_max)**2))
            upper_boundary_part3.append(Point32(-x_max, -y_r, 0))
            lower_boundary_part3.append(Point32(-x_min_3, -y_r, 0))
        lower_boundary_part3[0] = lower_boundary_part2[0]
        print("lower_boundry_part1", upper_boundary_part1)
        print("lower_boundry_part2", upper_boundary_part2)
        print("lower_boundry_part3", upper_boundary_part3)
        return (upper_boundary_part1, lower_boundary_part1), (upper_boundary_part2, lower_boundary_part2), (upper_boundary_part3, lower_boundary_part3)

    def publish_polygon(self):
        """Publish the safety polygon area."""
        polygon_msg = PolygonArray()
        polygon_msg.header.frame_id = "map"
        polygon_msg.header.stamp = rospy.Time.now()
        R_max, R_min, w = 3.8, 3.6, 0.2
        part1, part2, part3 = self.get_sector_boundary_points(-self.target_x, self.target_y, self.Rma, R_min, w)
        for idx, (upper_boundary, lower_boundary) in enumerate([part1, part2, part3]):
            safety_area = PolygonStamped()
            safety_area.header.frame_id = "map"
            safety_area.header.stamp = rospy.Time.now()
            full_boundary = upper_boundary + lower_boundary[::-1]
            if idx == 2:
                full_boundary = upper_boundary + [lower_boundary[0]]
            for point in full_boundary:
                point_msg = Point32()
                point_msg.x = point.x
                point_msg.y = point.y
                point_msg.z = point.z
                safety_area.polygon.points.append(point_msg)
            polygon_msg.polygons.append(safety_area)
            polygon_msg.labels.append(idx)
        self.safety_polygon_pub.publish(polygon_msg)

    def is_within_sector(self, x_r, y_r, x_p, y_p, R_max, R_min, w):
        """Check if a point is within the safety sector."""
        if y_p - w < y_r < y_p + w:
            x_min = x_p
            x_max = x_p + np.sqrt((R_max + w)**2 - (y_r - y_p + R_max)**2)
            if x_min <= x_r <= x_max:
                print("1111x_min", x_min, "x_max", x_max, "x_r", x_r)
                return True
            else:
                print("0111x_min", x_min, "x_max", x_max, "x_r", x_r)
        elif y_p - R_min < y_r < y_p - w:
            x_min = x_p + np.sqrt((R_min - w)**2 - (y_r - y_p + R_min)**2)
            x_max = x_p + np.sqrt((R_max + w)**2 - (y_r - y_p + R_max)**2)
            if x_min <= x_r <= x_max:
                print("2222x_min", x_min, "x_max", x_max, "x_r", x_r)
                return True
            else:
                print("0222x_min", x_min, "x_max", x_max, "x_r", x_r)
        elif y_p - R_max <= y_r <= y_p - R_min:
            x_min = x_p + R_min - w
            x_max = x_p + np.sqrt((R_max + w)**2 - (y_r - y_p + R_max)**2)
            if x_min <= x_r <= x_max:
                print("3333x_min", x_min, "x_max", x_max, "x_r", x_r)
                return True
            else:
                print("0333x_min", x_min, "x_max", x_max, "x_r", x_r)
        return False

    def calculate_reverse(self, current_x, current_y, theta_degree, x_r, y_r):
        """Calculate reward for reverse parking."""
        start_time = rospy.get_time()
        theta_radian = math.radians(theta_degree)
        xt_diff = abs(abs(current_x) - (abs(self.target_location.x)))
        yt_diff = abs((abs(current_y) - abs(self.target_location.y)))
        distance_diff = math.sqrt(xt_diff ** 2 + yt_diff ** 2)
        x_parked, y_parked = self.if_parked(current_x, current_y, self.target_location.x, self.target_location.y)
        scale_factor = -30
        offset = 200
        d_reward = 0
        self.angle_history.append(theta_degree)
        if len(self.angle_history) > 25:
            self.angle_history.pop(0)
        angle_penalty = 0
        if len(self.angle_history) == 25:
            all_angle_valid = [abs(self.angle_history[i] - self.angle_history[i - 1]) for i in range(1, len(self.angle_history))]
            if all(change <= 5 for change in all_angle_valid) and not (110 <= abs(theta_degree) <= 180):
                angle_penalty = -20
        if -90 <= theta_degree <= 90:
            angle_reward = -math.cos(math.radians(theta_degree)) 
        else:
            angle_reward = abs(math.cos(math.radians(theta_degree))) 
        x_reward = (1 / (xt_diff + 0.001))
        y_reward = 1 / (yt_diff - 0.2 + 0.001)
        distance_reward = math.sqrt(x_reward ** 2 + y_reward ** 2)
        print("distance_diff", distance_diff)
        print("angle_penalty", angle_penalty, "theta_degree", theta_degree)
        if 5 <= distance_diff < 8.5:
            ex_reward = (distance_reward * 0.1) * (angle_reward * 0.9)
        elif 0 <= distance_diff < 5:
            ex_reward = (distance_reward * 0.4) * (angle_reward * 0.6)
        else:
            ex_reward = (distance_reward * 0.5) * (angle_reward * 0.5)
        if yt_diff <= 0.01:
            d_reward = -20
        reward_suceess = 50 if x_parked and y_parked else 0
        sector_reward = 15 if self.is_within_sector(x_r, y_r, -self.target_location.x, abs(self.target_location.y), self.Rma, self.Rmin, 0.05) else -10
        reward = ex_reward + reward_suceess + d_reward + angle_penalty + sector_reward
        print("x_reward", x_reward, "y_reward", y_reward)
        print("Distance Reward:", distance_reward)
        print("Angle Reward:", angle_reward)
        print("ex_reward", ex_reward)
        print("Total Reward:", reward)
        return reward

    def step(self, action):
        """Execute one step in the environment."""
        print("Current gear:", self.reverse)
        self.publish_gear()
        done = False
        reward = 0
        reward_stage1 = 0
        reward_stage2 = 0
        if hasattr(self, 'current_distance_to_goal'):
            self.last_distance_to_goal = self.current_distance_to_goal
        else:
            self.last_distance_to_goal = None
        current_vehicle_transform = self.ego_vehicle.get_transform()
        current_vehicle_location = current_vehicle_transform.location
        current_x = current_vehicle_location.x
        current_y = -current_vehicle_location.y
        target_park_rotation = self.target_park['rotation']
        self.target_location = self.target_park['location']
        vehicle_yaw = -current_vehicle_transform.rotation.yaw
        park_yaw = target_park_rotation.yaw
        theta = vehicle_yaw - park_yaw
        self.current_angle = theta
        if self.last_angle is None:
            self.last_angle = self.current_angle
        _, _, diff_x, diff_y, xr, yr = self.calculate_reverse_steering1(current_x, current_y, theta)
        dx = abs(current_x - self.target_location.x)
        _, north_close_distance, _ = self.get_bev_img_obs()
        north_close_distance = min(north_close_distance, 5.2)
        self.Rma = north_close_distance + dx
        print("Current max turning radius===", self.Rma, "Obstacle distance====", north_close_distance, "diff_x===", dx, "diff_y===", diff_y, "theta", theta, "self.last_angle", self.last_angle)
        self.current_distance_to_goal = math.sqrt(diff_x ** 2 + diff_y ** 2)
        self.past_distance.append(self.current_distance_to_goal)
        augular_velocity = self.ego_vehicle.get_angular_velocity()
        linear_velocity = self.ego_vehicle.get_velocity()
        vehicle_angular_velocity = math.sqrt((augular_velocity.x) ** 2 + (augular_velocity.y) ** 2 + (augular_velocity.z) ** 2)
        vehicle_linear_velocity = math.sqrt(linear_velocity.x ** 2 + linear_velocity.y ** 2 + linear_velocity.z ** 2)
        if len(self.past_distance) > self.num_distance:
            self.past_distance.pop(0)
        nearest_area = self.find_nearest_parking_area(diff_x, diff_y)
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        if self.reverse:
            print("Entering reverse process")
            carla_control_msg = CarlaEgoVehicleControl()
            carla_control_msg.steer = action[0]
            carla_control_msg.reverse = True
            if action[1] >= 0:
                carla_control_msg.throttle = action[1] * 0.3
            else:
                carla_control_msg.throttle = 0.0
                carla_control_msg.brake = 0.0
            self.control_pub.publish(carla_control_msg)
            self.publish_polygon()
            start_time = rospy.get_time()
            correct_position_stagnating, off_position_stagnating = self.check_non_movement(self.current_distance_to_goal, self.last_distance_to_goal)
            if rospy.get_time() - start_time > 35:
                print("Reverse time too long, exiting")
                done = True
            elif self.collision_with_actor is not None:
                print("collided")
                done = True
            elif off_position_stagnating:
                done = True
                print('stayed long')
            else:
                reward_stage1 = self.calculate_reverse(current_x, current_y, theta, xr, yr)
                x_park, y_park = self.if_parked(current_x, current_y, self.target_location.x, self.target_location.y)
                if x_park and y_park:
                    reward_stage1 = 30
                    print("Parking successful")
                    done = True
                else:
                    print("still learning")
                    done = False
            print("throttle", carla_control_msg.throttle)
            print("steer", carla_control_msg.steer)
            print("reward_stage1", reward_stage1)
            print("current_x", current_x, "current_y", current_y)
            print("target_x", self.target_location.x, "target_y", self.target_location.y)
            print("Distance x:", current_x - self.target_location.x, "Distance y:", current_y - abs(self.target_location.y))
            time.sleep(0.1)
        else:
            at_parking_point = self.check_stop_condition_west(diff_x, diff_y, theta)
            if at_parking_point:
                print("Reached west parking area", "diff_x", diff_x, "diff_y", diff_y)
                carla_control_msg = CarlaEgoVehicleControl()
                carla_control_msg.reverse = True
                self.control_pub.publish(carla_control_msg)
                self.reverse = True
            carla_control_msg = CarlaEgoVehicleControl()
            carla_control_msg.steer = action[0]
            carla_control_msg.reverse = False
            if action[1] >= 0:
                carla_control_msg.throttle = action[1]
                carla_control_msg.brake = 0.0
            else:
                carla_control_msg.throttle = 0.0
                carla_control_msg.brake = abs(action[1])
            self.control_pub.publish(carla_control_msg)
            self.last_diff_x = diff_x
            self.last_diff_y = diff_y
            time.sleep(0.15)
            current_state_dict = self.get_state()
            distance = current_state_dict['distance_to_goal']
            correct_position_stagnating, off_position_stagnating = self.check_non_movement(self.current_distance_to_goal, self.last_distance_to_goal)
            done = False
            vehicle_transform = self.ego_vehicle.get_transform()
            if self.collision_with_actor is not None:
                print("collided")
                done = True
            elif off_position_stagnating:
                done = True
                print('stayed long')
            else:
                reward_stage2 = self.calculate_reward_west(vehicle_linear_velocity * 3.6, diff_y=diff_y, vehicle_yaw=vehicle_yaw, park_yaw=park_yaw, last_diff_y=self.last_diff_y, diff_x=diff_x, last_diff_x=self.last_diff_x, at_parking_point=at_parking_point, vehicle_transform=vehicle_transform, target_location=self.target_location, d_val_1=8.5, d_val_2=10.5)
                # if reward_stage2 <= -50:
                #     print('distance too far')
                #     done = True
                # else:
                #     done = False
                #     print("Agent still learning %s" % reward_stage2)
                self.last_diff_x = diff_x
                self.last_diff_y = diff_y
                self.car_transform = self.ego_vehicle.get_transform().location
                self.vehicle_yaw = vehicle_yaw
                self.park_yaw = park_yaw
            print("vehicle_yaw", vehicle_yaw)
            print("reward_stage2 %s" % reward_stage2)
            print(f"current_distance_to_goal: {self.current_distance_to_goal}")
            print(f"last_distance_to_goal {self.last_distance_to_goal}")
        if done:
            print("Resetting environment due to done = true")
            self.reset()
        reward = reward_stage1 + reward_stage2
        print("++++++reward_stage1", reward_stage1, "++++++reward_stage2", reward_stage2)
        print("+++++reward", reward)
        return reward, done, {}

    def run(self):
        """Run the parking environment simulation."""
        running = True
        episode = 0
        index = 0
        while not rospy.is_shutdown():
            self.get_state()
            index += 1
            action = np.array([(1-(-1)) * np.random.random() + (-1), (1-(-1)) * np.random.random() + (-1)])
            print(action)
            print("---action: %s" % action)
            next_state, reward, done = self.step(action)
            print(reward)
            state = next_state
            if done:
                episode += 1
                self.reset()

if __name__ == "__main__":
    env = ParkingEnv()
    env.run()