import os
import gym
import math
from gym import spaces
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import torch

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

class multiAgentWithoutObstacleV0(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents, GUI=False):
        super(multiAgentWithoutObstacleV0, self).__init__()

        self.num_agents = num_agents

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, 2), dtype=float)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, 6), dtype=float)

        # Initialize PyBullet
        if GUI==False:
            self.client = p.connect(p.DIRECT)
        else:
            self.client = p.connect(p.GUI)

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)
        self.droneInitInfo = [[i * 0.5, 0, 0.5] for i in range(self.num_agents)]  # 在x轴上直线排列
        self.reset()

    def reset(self):
        # Reset the environment
        p.resetSimulation()

        # 构造plane模型的绝对路径
        plane_path = os.path.join(script_dir, "../resources/plane_implicit.urdf")
        p.loadURDF(fileName=plane_path,
                   basePosition=[0, 0, 0],
                   physicsClientId=self.client)
        
        # create target model
        radius = 0.1            # 球形的半径
        color = [1, 0, 0, 0.5]    # 红色
        self.targetPosition = [2.0, 2.0, 0.5]  # 球形模型的初始位置
        sphere_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
        self.target = p.createMultiBody(baseMass=0,  # No mass, the sphere is not affected by physics
                                        baseCollisionShapeIndex=sphere_collision_shape_id,
                                        baseVisualShapeIndex=-1,  # No visual shape
                                        basePosition=self.targetPosition)
        
        # 将球形模型悬浮在空中
        sphere_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color)  # Red and transparent
        p.createMultiBody(baseMass=0,  # No mass, the sphere is not affected by physics
                          baseCollisionShapeIndex=-1,  # No collision shape
                          baseVisualShapeIndex=sphere_visual_shape_id,
                          basePosition=self.targetPosition)
        
        # Load drone models
        self.drones = []
        drone_path = os.path.join(script_dir, "../resources/cf2x.urdf")
        for i in range(self.num_agents):
            drone = p.loadURDF(fileName=drone_path,
                               basePosition=self.droneInitInfo[i],
                               physicsClientId=self.client)
            self.drones.append(drone)
        
        self.states = []
        for i in range(self.num_agents):
            drone_position = self.droneInitInfo[i][:3]
            print("xxxxxxxxx", drone_position, type(drone_position))
            states[i] = torch.tensor(drone_position)
            position_lat = self._computeDistanceLat(drone_position, self.targetPosition)
            initial_state = self.droneInitInfo[i] + position_lat
            p.resetBasePositionAndOrientation(self.drones[i], self.droneInitInfo[i][:3], p.getQuaternionFromEuler([0, 0, 0]))
            p.resetBaseVelocity(self.drones[i], [0, 0, 0], [0, 0, 0])
            self.states.append(initial_state)

        return np.array(self.states)
    
    def step(self, actions):
        # Apply actions to the drones
        for i in range(self.num_agents):
            self.apply_action(i, actions[i])

        # Step the simulation
        p.stepSimulation()

        # Get new states
        self.states = [self.get_state(i) for i in range(self.num_agents)]

        # Calculate rewards
        rewards = [self.calculate_reward(i, self.states[i]) for i in range(self.num_agents)]

        # Check if done
        dones = [self.is_done(i, self.states[i]) for i in range(self.num_agents)]

        return np.array(self.states), np.array(rewards), np.array(dones), {}
    
    def apply_action(self, agent_idx, action):
        # print(action)
        # action is 3 dimension
        thrust_x, thrust_y = action

        # Clip thrust and torque
        thrust_x = np.clip(thrust_x, -0.1, 0.1)
        thrust_y = np.clip(thrust_y, -0.1, 0.1)
        # thrust_z = np.clip(thrust_z, -0.1, 0.1)

        p.applyExternalForce(
            self.drones[agent_idx],
            4,
            forceObj=[thrust_x, thrust_y, 0],
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
        )
    
    def get_state(self, agent_idx):
        # Get drone state
        position, _ = p.getBasePositionAndOrientation(self.drones[agent_idx])
        position_lat = self._computeDistanceLat(position, self.targetPosition)
        return list(position) + position_lat
    
    def calculate_reward(self, agent_idx, state):
        # Calculate reward based on the current state
        # ...
        reward = 0
        currentPosition = state[0:3]
        distance = self._calculateDistance(currentPosition, self.targetPosition)
        if (state[0] > 2.2 or state[0] < -0.2 or
            state[1] > 2.2 or state[1] <  -0.2 or
            state[2] > 1 or state[2] <   0):
            reward = -1500

        if distance <= 0.1:
            reward = 2000

        return reward
    
    def is_done(self, agent_idx, state):
        # Check if the episode is done based on the current state
        # ...
        done = False
        currentPosition = state[0:3]
        distance = self._calculateDistance(currentPosition, self.targetPosition)
        if (state[0] > 2.2 or state[0] < -0.2 or
            state[1] > 2.2 or state[1] <  -0.2 or
            state[2] > 1 or state[2] <   0):
            done = True

        if distance <= 0.1:
            done = True

        return done

    def close(self):
        # 关闭环境
        p.disconnect()

    def _computeDistanceLat(self, position1, position2) -> list:
        '''
        position1距离position2分别在x,y,z三个分量上的位置偏移量
        The position offset of position1 from position2 on the three components of x, y, and z respectively
        '''
        position_lat = np.zeros(3)
        position2 = np.array(position2)
        position_lat[0] = position2[0] - position1[0]
        position_lat[1] = position2[1] - position1[1]
        position_lat[2] = position2[2] - position1[2]
        return list(position_lat)
    
    def _calculateDistance(self, position1, position2) -> float:
        """
        计算position1和position2之间的距离
        Calculate distance based on distance between position1 and position2
        :param position1: Position information of coordinate 1, including x, y, z
        :param position2: Position information of coordinate 2, including x, y, z
        """
        
        return math.sqrt(
            (position1[0] - position2[0]) ** 2 +
            (position1[1] - position2[1]) ** 2 +
            (position1[2] - position2[2]) ** 2
        )
