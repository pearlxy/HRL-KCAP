import os
import time
import signal
import subprocess
import numpy as np
import rospy
import torch
from Algorithm.ppo_modified import generate_action_no_sampling
from tensorboardX import SummaryWriter
from perpen_parking import ParkingEnv
from Algorithm.backbone import CNNPolicy


# Set up directories and parameters
log_dir = "./bevrl_logs_01/reverse_v5"
writer = SummaryWriter(log_dir)
RL_TRAINING_DEVICE = torch.cuda.device_count() - 1
BEV_INPUT_CHANNELS = 2
action_bound = [[-1, -1], [1, 1]]
policy_checkpoints_path = "./checkpoint/best.pt"

def run(env: ParkingEnv, policy, action_bound: list):
    length_of_test_episodes = 5
    total_test_results = []

    for id in range(0, length_of_test_episodes):
        env.reset()
        spawn_obstacle = subprocess.Popen(["python3", "audi_path.py"])
        # time.sleep(1)

        terminal = False
        step = 1
        ep_reward = 0

        # Get initial state
        bev_obs, closest_north_distance, closest_north_angle = env.get_bev_img_obs()
        _, _, _, propri_obs = env.get_propriceptive_obs()
        state = [bev_obs, propri_obs]

        while not terminal and not rospy.is_shutdown():
            state_list = [state]                                    

            # Generate actions without sampling
            _, scaled_action = generate_action_no_sampling(env=env, state_list=state_list, policy=policy, action_bound=action_bound)

            # Perform action in the environment
            r, _, result = env.step(scaled_action[0])

            # Get next state
            bev_obs_next, _, _ = env.get_bev_img_obs()
            _, _, _, propri_obs_next = env.get_propriceptive_obs()
            state_next = [bev_obs_next, propri_obs_next]

            step += 1
            state = state_next

        # End subprocess and wait
        os.kill(spawn_obstacle.pid, signal.SIGTERM)
        spawn_obstacle.wait()
        # time.sleep(0.2)

        # Reset the environment
        # for i in range(100):
        #     env.step([0, 0])

    print(f"Testing completed!")

if __name__ == "__main__":
    # Initialize environment
    bev_env = ParkingEnv()
    time.sleep(1)

    # Prepare the policy network and load pre-trained weights
    policy = CNNPolicy(frames=BEV_INPUT_CHANNELS, action_space=2)
    policy.cuda(device=RL_TRAINING_DEVICE)
    policy.eval()  # Set to evaluation mode

    if os.path.exists(policy_checkpoints_path):
        print('############ Loading Model ############')
        resume_checkpoint = torch.load(policy_checkpoints_path)
        state_dict = resume_checkpoint["state_dict"]
        policy.load_state_dict(state_dict)
        print("Model loaded successfully!")
    else:
        print("Checkpoint not found, please check the path.")

    # Start testing
    try:
        run(env=bev_env, policy=policy, action_bound=action_bound)
    except KeyboardInterrupt:
        pass