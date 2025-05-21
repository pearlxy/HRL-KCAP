import carla
import time
from nav_msgs.msg import Odometry

import math
import rospy
import signal
import sys

def signal_handler(sig, frame):
    print('signal received',sig)
    clean_up()
    sys.exit(0)

def clean_up():
    global vehicle_instance
    vehicle_instance.destroy_vehicle()

class ObstaclePark():
    def __init__(self):
        global vehicle_instance
        vehicle_instance = self
        rospy.init_node('obstacle_park')
        self.vehicle_odom_sub = rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, self.vehicle_odom_callback)
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(10.0)  # 设置超时为10秒
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        rospy.loginfo("Connected to Carla server")
        
        self.park_location = carla.Location(x=3.691190719604492, y= -27.149078369140625, z=0.5)
        self.vehicle_pose = [0, 0, 0, 0, 0, 0]
        
        self.vehicle_bp = self.blueprint_library.find('vehicle.audi.a2')
        self.spawn_point = carla.Transform(carla.Location(x=-6.5, y= -21.6, z=0.5))
        self.vehicle_tesla_bp = self.blueprint_library.find("vehicle.dodge.charger_police")
        self.transform1 = carla.Transform(carla.Location(x=-6.7, y= -30.0, z=0.5))
        self.spawn1 = self.blueprint_library.find('vehicle.jeep.wrangler_rubicon')
        self.transform2 = carla.Transform(carla.Location(x=-6.3, y= -18.8, z=0.5))
        self.spawn2 = self.blueprint_library.find('vehicle.citroen.c3')
        self.transform3 = carla.Transform(carla.Location(x=3.5, y= -18.7, z=0.5))
        self.spawn3 = self.blueprint_library.find('vehicle.jeep.wrangler_rubicon')
        self.transform4 = carla.Transform(carla.Location(x=3.5, y= -21.6, z=0.5))
        self.spawn4 = self.blueprint_library.find('vehicle.tesla.model3')
        self.transform5 = carla.Transform(carla.Location(x=3.9, y= -27.2, z=0.5))
        self.spawn5 = self.blueprint_library.find('vehicle.jeep.wrangler_rubicon')
        self.transform6 = carla.Transform(carla.Location(x=3.1, y= -32.7, z=0.5))
        self.spawn6 = self.blueprint_library.find('vehicle.audi.a2')
        self.transform7 = carla.Transform(carla.Location(x=3.7, y= -35.6, z=0.5))
        self.spawn7 = self.blueprint_library.find('vehicle.citroen.c3')
        self.transform8 = carla.Transform(carla.Location(x=-6.6, y= -35.6, z=0.5))
        self.spawn8 = self.blueprint_library.find('vehicle.jeep.wrangler_rubicon')
        self.transform9 = carla.Transform(carla.Location(x=3.7, y= -38.3, z=0.5))
        self.spawn9 = self.blueprint_library.find('vehicle.audi.a2')
        self.transform10 = carla.Transform(carla.Location(x=-6.6, y= -38.3, z=0.5))
        self.spawn10 = self.blueprint_library.find('vehicle.jeep.wrangler_rubicon')
        self.transform11 = carla.Transform(carla.Location(x=3.3, y= -41.1, z=0.5))
        self.spawn11 = self.blueprint_library.find('vehicle.citroen.c3')
        self.transform12 = carla.Transform(carla.Location(x=3.4, y= -43.9, z=0.5))
        self.spawn12 = self.blueprint_library.find('vehicle.audi.a2')
        self.transform13 = carla.Transform(carla.Location(x=9.9, y= -24.2, z=0.5))
        self.spawn13 = self.blueprint_library.find('vehicle.tesla.model3')
        self.transform14 = carla.Transform(carla.Location(x=3.8, y= -29.9, z=0.5))
        self.spawn14 = self.blueprint_library.find('vehicle.citroen.c3')
        self.transform15 = carla.Transform(carla.Location(x=9.9, y= -29.9, z=0.5))
        self.spawn15 = self.blueprint_library.find('vehicle.dodge.charger_police')
        self.transform16 = carla.Transform(carla.Location(x=9.9, y= -32.7, z=0.5))
        self.spawn16 = self.blueprint_library.find('vehicle.tesla.model3')
        self.transform17 = carla.Transform(carla.Location(x=9.9, y= -35.5, z=0.5))
        self.spawn17 = self.blueprint_library.find('vehicle.audi.a2')
        self.transform18 = carla.Transform(carla.Location(x=9.9, y= -38.3, z=0.5))
        self.spawn18 = self.blueprint_library.find('vehicle.toyota.prius')
        self.transform19 = carla.Transform(carla.Location(x=-13.5, y= -21.7, z=0.5))
        self.spawn19 = self.blueprint_library.find('vehicle.tesla.model3')
        self.transform20 = carla.Transform(carla.Location(x=-13.5, y= -24.5, z=0.5))
        self.spawn20 = self.blueprint_library.find('vehicle.audi.a2')
        self.transform21 = carla.Transform(carla.Location(x=-13.5, y= -27.3, z=0.5))
        self.spawn21 = self.blueprint_library.find('vehicle.dodge.charger_police')
        self.transform22 = carla.Transform(carla.Location(x=-13.5, y= -30.1, z=0.5))
        self.spawn22 = self.blueprint_library.find('vehicle.citroen.c3')
        self.transform23 = carla.Transform(carla.Location(x=-13.5, y= -33, z=0.5))
        self.spawn23 = self.blueprint_library.find('vehicle.toyota.prius')
        self.transform24 = carla.Transform(carla.Location(x=-13.5, y= -35.8, z=0.5))
        self.spawn24 = self.blueprint_library.find('vehicle.tesla.model3')
        self.transform25 = carla.Transform(carla.Location(x=-13.5, y= -38.6, z=0.5))
        self.spawn25 = self.blueprint_library.find('vehicle.audi.a2')
        self.transform26 = carla.Transform(carla.Location(x=-13.5, y= -41.4, z=0.5))
        self.spawn26 = self.blueprint_library.find('vehicle.dodge.charger_police')
        self.transform27 = carla.Transform(carla.Location(x=-13.5, y= -44.0, z=0.5))
        self.spawn27 = self.blueprint_library.find('vehicle.citroen.c3')
        self.transform28 = carla.Transform(carla.Location(x=-13.5, y= -47, z=0.5))
        #大巴车
      



        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, self.spawn_point)
        self.vehicle_tesla = self.world.try_spawn_actor(self.vehicle_tesla_bp, self.transform1)
        # self.spawn1 = self.world.try_spawn_actor(self.spawn1, self.transform2)
        # if self.spawn1:
        #     self.spawn1.set_simulate_physics(False)
        self.spawn2 = self.world.try_spawn_actor(self.spawn2, self.transform3)
        # if self.spawn2:
        #     self.spawn2.set_simulate_physics(False)
        # self.spawn3 = self.world.try_spawn_actor(self.spawn3, self.transform4)
        # if self.spawn3:
        #     self.spawn3.set_simulate_physics(False)
        self.spawn4 = self.world.try_spawn_actor(self.spawn4, self.transform5)
        # if self.spawn4:
        #     self.spawn4.set_simulate_physics(False)
        self.spawn5 = self.world.try_spawn_actor(self.spawn5, self.transform6)
        # if self.spawn5:
        #     self.spawn5.set_simulate_physics(False)
        self.spawn6 = self.world.try_spawn_actor(self.spawn6, self.transform7)
        # if self.spawn6:
        #    self.spawn6.set_simulate_physics(False)
        # self.spawn7 = self.world.try_spawn_actor(self.spawn7, self.transform8)
        # if self.spawn7:
        #     self.spawn7.set_simulate_physics(False)
        # self.spawn8 = self.world.try_spawn_actor(self.spawn8, self.transform9)
        # if self.spawn8:
        #     self.spawn8.set_simulate_physics(False)
        # self.spawn9 = self.world.try_spawn_actor(self.spawn9, self.transform10)
        # if self.spawn9:
        #     self.spawn9.set_simulate_physics(False)
        # self.spawn10 = self.world.try_spawn_actor(self.spawn10, self.transform11)
        # if self.spawn10:
        #     self.spawn10.set_simulate_physics(False)
        # self.spawn11 = self.world.try_spawn_actor(self.spawn11, self.transform12)
        # if self.spawn11:
        #     self.spawn11.set_simulate_physics(False)
        self.spawn12 = self.world.try_spawn_actor(self.spawn12, self.transform13)
        # if self.spawn12:

        #     self.spawn12.set_simulate_physics(False)
        # self.spawn13 = self.world.try_spawn_actor(self.spawn13, self.transform14)
        # if self.spawn13:
        #     self.spawn13.set_simulate_physics(False)
        self.spawn14 = self.world.try_spawn_actor(self.spawn14, self.transform15)
        # if self.spawn14:
        #     self.spawn14.set_simulate_physics(False)
        # self.spawn15 = self.world.try_spawn_actor(self.spawn15, self.transform16)
        # if self.spawn15:
        #     self.spawn15.set_simulate_physics(False)

        self.spawn16 = self.world.try_spawn_actor(self.spawn16, self.transform17)
        if self.spawn16:
            self.spawn16.set_simulate_physics(False)
        self.spawn17 = self.world.try_spawn_actor(self.spawn17, self.transform18)
        # if self.spawn17:
        #     self.spawn17.set_simulate_physics(False)
        # self.spawn18 = self.world.try_spawn_actor(self.spawn18, self.transform19)
        # if self.spawn18:
        #     self.spawn18.set_simulate_physics(False)
        # self.spawn19 = self.world.try_spawn_actor(self.spawn19, self.transform20)
        # if self.spawn19:
        #     self.spawn19.set_simulate_physics(False)
        # self.spawn20 = self.world.try_spawn_actor(self.spawn20, self.transform21)
        # if self.spawn20:
        #     self.spawn20.set_simulate_physics(False)
        self.spawn21 = self.world.try_spawn_actor(self.spawn21, self.transform22)
        # if self.spawn21:
        #     self.spawn21.set_simulate_physics(False)
        # self.spawn22 = self.world.try_spawn_actor(self.spawn22, self.transform23)

        # if self.spawn22:
        #     self.spawn22.set_simulate_physics(False)
        self.spawn23 = self.world.try_spawn_actor(self.spawn23, self.transform24)
        # if self.spawn23:
        #     self.spawn23.set_simulate_physics(False)
        # self.spawn24 = self.world.try_spawn_actor(self.spawn24, self.transform25)
        # if self.spawn24:
        #     self.spawn24.set_simulate_physics(False)
        self.spawn25 = self.world.try_spawn_actor(self.spawn25, self.transform26)
        # if self.spawn25:
        #     self.spawn25.set_simulate_physics(False)
        self.spawn26 = self.world.try_spawn_actor(self.spawn26, self.transform27)
        # if self.spawn26:
        #     self.spawn26.set_simulate_physics(False)
        # self.spawn27 = self.world.try_spawn_actor(self.spawn27, self.transform28)
        # if self.spawn27:
        #     self.spawn27.set_simulate_physics(False)
        # self.spawn7 = self.world.try_spawn_actor(self.spawn7, self.transform8)
        # self.spawn8 = self.world.try_spawn_actor(self.spawn8, self.transform9)
        # self.spawn9 = self.world.try_spawn_actor(self.spawn9, self.transform10)
        # self.spawn10 = self.world.try_spawn_actor(self.spawn10, self.transform11)
        # self.spawn11 = self.world.try_spawn_actor(self.spawn11, self.transform12)
        if self.vehicle is None:
            print("Failed to spawn vehicle")
            return
        else:
            print("Vehicle spawned successfully")

        self.throttle_time = 0  # 记录油门开始的时间
        
        self.throttle_applied = False  # 油门是否被应用
        signal.signal(signal.SIGTERM, signal_handler)

    def reset_environment(self):
        self.destroy_vehicle()
        self.destroy_tesla()
        # self.destroy_spawn1()
        self.destroy_spawn2()
        # self.destroy_spawn3()
        self.destroy_spawn4()
        self.destroy_spawn5()
        self.destroy_spawn6()
        # self.destroy_spawn7()
        # self.destroy_spawn8()
        # self.destroy_spawn9()
        # self.destroy_spawn10()
        # self.destroy_spawn11()
        self.destroy_spawn12()
        # self.destroy_spawn13()
        self.destroy_spawn14()
        # self.destroy_spawn15()
        self.destroy_spawn16()
        self.destroy_spawn17()
        # self.destroy_spawn18()
        # self.destroy_spawn19()
        # self.destroy_spawn20()
        self.destroy_spawn21()
        # self.destroy_spawn22()
        self.destroy_spawn23()
        # self.destroy_spawn24()
        self.destroy_spawn25()
        self.destroy_spawn26()
        # self.destroy_spawn27()
 
        time.sleep(1)
        # 确保 Tesla 已经销毁，使用 get_actors() 检查是否有 Tesla 存在
        actors = self.world.get_actors()
        for actor in actors:
            if actor.type_id == "vehicle.dodge.charger_police":
                actor.destroy()
                print("Existing Tesla actor destroyed")
        # 重新生成障碍物车辆
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, self.spawn_point)
        self.vehicle_tesla = self.world.try_spawn_actor(self.vehicle_tesla_bp, self.transform1)
        # self.spawn1 = self.world.try_spawn_actor(self.spawn1, self.transform2)
        if self.vehicle_tesla is None:
            print("111Failed to spawn vehicle")
        else:
            print("111Vehicle spawned successfully")
    

    def control_vehicle(self):
        while True:
            if self.vehicle is not None:
                if self.is_within_circle(self.vehicle_pose[0], abs(self.vehicle_pose[1]), self.park_location.x, abs(self.park_location.y), 10.0):
                    control = carla.VehicleControl(throttle=0.5, steer=-0.35)
                    self.vehicle.apply_control(control)
                    # print("Vehicle is moving")
                else:
                    control = carla.VehicleControl(throttle=0, steer=0.0)
                    self.vehicle.apply_control(control)
                   
            if self.vehicle_tesla is not None:
                if self.is_within_circle(self.vehicle_pose[0], abs(self.vehicle_pose[1]), self.park_location.x, abs(self.park_location.y), 5.5):
                    control = carla.VehicleControl(throttle=1, steer=-0.35)
                    self.vehicle_tesla.apply_control(control)
                    # print("Vehicle is moving")
                else:
                    control = carla.VehicleControl(throttle=0, steer=0.0)
                    self.vehicle_tesla.apply_control(control)


                

    


    

    def vehicle_odom_callback(self, odom_msg: Odometry):
        self.vehicle_pose = [
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z,
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
        ]

    def is_within_circle(self, vehicle_x, vehicle_y, center_x, center_y, radius):
        distance = math.sqrt((vehicle_x - center_x) ** 2 + (vehicle_y - center_y) ** 2)
        return distance <= radius + 0.56

    def destroy_vehicle(self):
        if self.vehicle:
            self.vehicle.destroy()
            print("Vehicle destroyed")
            self.vehicle = None  # 确保不再引用已销毁的车辆
    def destroy_tesla(self):
        if self.vehicle_tesla:
            self.vehicle_tesla.destroy()
            print("Vehicle Tesla destroyed")
            self.vehicle_tesla = None  # 确保不再引用已销毁的车辆
        else:
            print("tesla arlready destroyed")
    def destroy_spawn1(self):
        if self.spawn1:
            self.spawn1.destroy()
            self.spawn1 = None
    def destroy_spawn2(self):
        if self.spawn2:
            self.spawn2.destroy()
            self.spawn2 = None
    def destroy_spawn3(self):
        if self.spawn3:
            self.spawn3.destroy()
            self.spawn3 = None
    def destroy_spawn4(self):
        if self.spawn4:
            self.spawn4.destroy()
            self.spawn4 = None
    def destroy_spawn5(self):
        if self.spawn5:
            self.spawn5.destroy()
            self.spawn5 = None
    def destroy_spawn6(self):
        if self.spawn6:
            self.spawn6.destroy()
            self.spawn6 = None
    def destroy_spawn7(self):
        if self.spawn7:
            self.spawn7.destroy()
            self.spawn7 = None
    def destroy_spawn8(self):
        if self.spawn8:
            self.spawn8.destroy()
            self.spawn8 = None
    def destroy_spawn9(self):
        if self.spawn9:
            self.spawn9.destroy()
            self.spawn9 = None
    def destroy_spawn10(self):
        if self.spawn10:
            self.spawn10.destroy()
            self.spawn10 = None
    def destroy_spawn11(self):
        if self.spawn11:
            self.spawn11.destroy()
            self.spawn11 = None
    def destroy_spawn12(self):
        if self.spawn12:
            self.spawn12.destroy()
            self.spawn12 = None
    def destroy_spawn13(self):
        if self.spawn13:
            self.spawn13.destroy()
            self.spawn13 = None
    def destroy_spawn14(self):
        if self.spawn14:
            self.spawn14.destroy()
            self.spawn14 = None
    def destroy_spawn15(self):
        if self.spawn15:
            self.spawn15.destroy()
            self.spawn15 = None
    def destroy_spawn16(self):
        if self.spawn16:
            self.spawn16.destroy()
            self.spawn16 = None
    def destroy_spawn17(self):
        if self.spawn17:
            self.spawn17.destroy()
            self.spawn17 = None
    def destroy_spawn18(self):
        if self.spawn18:
            self.spawn18.destroy()
            self.spawn18 = None
    def destroy_spawn19(self):
        if self.spawn19:
            self.spawn19.destroy()
            self.spawn19 = None
    def destroy_spawn20(self):
        if self.spawn20:
            self.spawn20.destroy()
            self.spawn20 = None
    def destroy_spawn21(self):
        if self.spawn21:
            self.spawn21.destroy()
            self.spawn21 = None
    def destroy_spawn22(self):
        if self.spawn22:
            self.spawn22.destroy()
            self.spawn22 = None
    def destroy_spawn23(self):
        if self.spawn23:
            self.spawn23.destroy()
            self.spawn23 = None
    def destroy_spawn24(self):
        if self.spawn24:
            self.spawn24.destroy()
            self.spawn24 = None
    def destroy_spawn25(self):
        if self.spawn25:
            self.spawn25.destroy()
            self.spawn25 = None
    def destroy_spawn26(self):
        if self.spawn26:
            self.spawn26.destroy()
            self.spawn26 = None
    def destroy_spawn27(self):
        if self.spawn27:
            self.spawn27.destroy()
            self.spawn27 = None
    


    def run(self):
        try:
            while not rospy.is_shutdown():
                #  return
                self.control_vehicle()
                time.sleep(10)  # 重新开始前的等待时间
        finally:
                self.destroy_vehicle()
                self.destroy_tesla()
                # self.destroy_spawn1()
                self.destroy_spawn2()
                # self.destroy_spawn3()
                self.destroy_spawn4()
                self.destroy_spawn5()
                self.destroy_spawn6()
                # self.destroy_spawn7()
                # self.destroy_spawn8()
                # self.destroy_spawn9()
                # self.destroy_spawn10()
                # self.destroy_spawn11()
                self.destroy_spawn12()
                # self.destroy_spawn13()
                self.destroy_spawn14()
                # self.destroy_spawn15()
                self.destroy_spawn16()
                self.destroy_spawn17()
                # self.destroy_spawn18()
                # self.destroy_spawn19()
                # self.destroy_spawn20()
                self.destroy_spawn21()
                # self.destroy_spawn22()
                self.destroy_spawn23()
                # self.destroy_spawn24()
                self.destroy_spawn25()
                self.destroy_spawn26()
                # self.destroy_spawn27()
                # print("222audi===",temp.vehicle)
                # print("222tesla===", temp.vehicle_tesla)

if __name__ == '__main__':
    
    temp = ObstaclePark()
    temp.reset_environment()
    temp.run()