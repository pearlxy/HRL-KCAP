import math
from collections import deque
from geometry_msgs.msg import Pose
from nav_msgs.msg import Path

class Waypoint:
    def __init__(self, pose: Pose):
        self.x = pose.position.x
        self.y = pose.position.y
        self.z = pose.position.z
        self.qx = pose.orientation.x
        self.qy = pose.orientation.y
        self.qz = pose.orientation.z
        self.qw = pose.orientation.w


class GlobalPlanner:
    def __init__(self, global_path_msg: Path):
        self.global_path_msg = global_path_msg

        self.original_global_waypoints = None
        # self.global_original_path = None
        self.global_path_waypoints = None
        self.interval_dis = 0  # total local target waypoints
        self.initialize_path()
        # self.generate_global_path_waypoints()

    def initialize_path(self):
        """
        Generate Global Path in original distance.
        :return: None
        """
        self.original_global_waypoints = [Waypoint(wpt.pose) for wpt in self.global_path_msg.poses]
        print(f"Total length: {len(self.original_global_waypoints)}")


    def generate_global_path_waypoints(self):
        if self.original_global_waypoints is None:
            print(f"Empty Global Plan, Please Check !")
            return

        if len(self.original_global_waypoints) <= 2:
            print(f"Too short Global Plann !")
            return

        global_path_waypoints = []
        # step = math.ceil(len(self.original_global_waypoints) / self.interval_dis)
        # for i in range(self.interval_dis, len(self.original_global_waypoints), step):
        #     global_path_waypoints.append(self.original_global_waypoints[i])
        for i in range(len(self.original_global_waypoints)):
            global_path_waypoints.append(self.original_global_waypoints[i])
        self.global_path_waypoints = deque(global_path_waypoints)
        print(f"We totally choose {len(self.global_path_waypoints)} waypoints as our global path !")

        # for i in range(len(self.global_planner) - 1):
        #     dis = math.sqrt((self.global_planner[i+1].x - self.global_planner[i].x)**2 + (self.global_planner[i + 1].y - self.global_planner[i].y) ** 2)
        #     print(f"interval dis: {dis}")
