# HRL-KCAP


**Autonomous Parking Examples**
<figure>
  <img src="images/total.gif" style="width: 90%;">
</figure>

<figure>
  <img src="images/real-world.gif">
</figure>

## Environment
This project depends on these following packages, please use **pip install** command to install them. At the same time, please configure the simulation environment according to the official instructions, including [carla-0.9.12](https://github.com/carla-simulator/carla.git) and [carla-ros-bridge-0.9.12](https://github.com/carla-simulator/ros-bridge.git).
```
python 3.8.10
numpy 1.24.4
carla 0.9.12
cv2 4.2.0
torch 2.0.1+cu117
skimage 0.16.2
scipy 1.10.1
```
## Setup

### 1.1 Build the conda environment 
``` bash
conda create -n parking python=3.8.10
conda activate parking
git clone https://github.com/pearlxy/HRL-KCAP.git
cd HRL-KCMS
```


### 1.2 Launch the Carla server
``` bash
./CarlaUE4.sh
```


### 1.3 Launch ros-bridge
``` bash
# first, modify the config file for cross-vehicle compatibility 
cd ./carla-twelve-ros-bridge/catkin_ws/src/catkin_ws/src/ros-bridge/carla_spawn_objects/config 

(a) dsb_objects_Micro.json (b) dsb_objects_tesla.json (c) dsb_objects_jeep.json (d) dsb_objects_Vol.json

cd ./carla-twelve-ros-bridge/catkin_ws/src/catkin_ws/src/ros-bridge/carla_ros_bridge/launch

(a) ssb_Micro.launch (b) ssb_tesla.launch (c) ssb_jeep.launch (d) ssb_rl_Vol.launch 

# second, launch ros-bridge
source devel/setup.bash
roslaunch carla_ros_bridge ssb_rl.launch
```


## Evaluation (Inference using a pre-trained model)
Please download our weight file [best.pt](https://drive.google.com/file/d/1EWbA1NmTAHE-0JlLUwhw52bkHMKgPYat/view?usp=sharing) and place it in the **checkpoint** folder.
```
python hrl_parking.py
```

## Training
coming soon...

## Reference
Thanks to these open-source projects and their contributions: [Carla 0.9.12](https://github.com/carla-simulator/carla.git), [Carla-ros-bridge 0.9.12](https://github.com/carla-simulator/ros-bridge.git), and [CADLabeler](https://github.com/BruceXSK/CADLabeler).
