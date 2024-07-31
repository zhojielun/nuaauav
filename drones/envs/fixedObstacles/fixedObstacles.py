import os
import gym
import math
from gym import spaces
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

class FixedObstaclesV0(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, GUI):
        super(FixedObstaclesV0, self).__init__()

        self.drone = []

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=float)

        # Initialize PyBullet
        if GUI==False:
            self.client = p.connect(p.DIRECT)
        else:
            self.client = p.connect(p.GUI)

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)
        self.droneInitInfo = [0, 0, 0.5]
        self.obstacleIds = []
        self.reset()
        self.collision = False
        self._addObstacles()

    def reset(self):
        # Reset the environment
        p.resetSimulation()

        # Load plane model
        plane_path = "/root/miniconda3/envs/myconda/lib/python3.9/site-packages/pybullet_data/plane_implicit.urdf"
        p.loadURDF(fileName=plane_path,
                   basePosition=[0, 0, 0],
                   physicsClientId=self.client)
        
        # create target model
        radius = 0.1            # 球形的半径
        color = [1, 0, 0, 0.5]    # 红色
        targetPosition = [2.0, 2.0, 0.5]  # 球形模型的初始位置
        sphere_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
        self.target = p.createMultiBody(baseMass=0,  # No mass, the sphere is not affected by physics
                                        baseCollisionShapeIndex=sphere_collision_shape_id,
                                        baseVisualShapeIndex=-1,  # No visual shape
                                        basePosition=targetPosition)
        
        # 将球形模型悬浮在空中
        sphere_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color)  # Red and transparent
        p.createMultiBody(baseMass=0,  # No mass, the sphere is not affected by physics
                          baseCollisionShapeIndex=-1,  # No collision shape
                          baseVisualShapeIndex=sphere_visual_shape_id,
                          basePosition=targetPosition)
        
        # Load drone model
        drone_path = "/home/Pybullet/drones/envs/resources/cf2x.urdf"
        self.drone = p.loadURDF(fileName=drone_path,
                                basePosition=[0, 0, 0.5],
                                physicsClientId=self.client)
        
        self._housekeeping()
        self._checkCollision()
        drone_position = self.droneInitInfo[:3]
        position_lat = self._computeDistanceLat(drone_position, targetPosition)
        obstacle_dis = self._computeObstacleDis(drone_position, [0.5, 0.5, 0.5, 1.2, 1.2, 0.5])
        initial_state = self.droneInitInfo + position_lat + [0.5, 0.5, 0.5, 1.2, 1.2, 0.5] + obstacle_dis
        p.resetBasePositionAndOrientation(self.drone, self.droneInitInfo[:3], p.getQuaternionFromEuler([0, 0, 0]))
        p.resetBaseVelocity(self.drone, [0, 0, 0], [0, 0, 0])

        return initial_state
    
    def step(self, action):
        # Apply action to the drone
        self.apply_action(action=action)

        # Step the simulation
        p.stepSimulation()

        # Get new state
        state = self.get_state()

        # Calculate reward
        reward = self.calculate_reward(state)

        # Check if done
        done = self.is_done(state)

        return state, reward, done, {}
    
    def apply_action(self, action):
        # print(action)
        # action is 3 dimension
        thrust_x, thrust_y = action

        # Clip thrust and torque
        thrust_x = np.clip(thrust_x, -0.1, 0.1)
        thrust_y = np.clip(thrust_y, -0.1, 0.1)
        # thrust_z = np.clip(thrust_z, -0.1, 0.1)

        p.applyExternalForce(
            self.drone,
            4,
            forceObj=[thrust_x, thrust_y, 0],
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
        )
    
    def get_state(self):
        # Get drone state
        self._checkCollision()
        targetPosition = [2.0, 2.0, 0.5]
        position, _ = p.getBasePositionAndOrientation(self.drone)
        position_lat = self._computeDistanceLat(position, targetPosition)
        obstacle_dis = self._computeObstacleDis(position, [0.5, 0.5, 0.5, 1.2, 1.2, 0.5])
        return list(position) + position_lat + [0.5, 0.5, 0.5, 1.2, 1.2, 0.5] + obstacle_dis
    
    def calculate_reward(self, state):
        # Calculate reward based on the current state
        # ...
        reward = 0
        currentPosition = state[0:3]
        targetPosition = [2.0, 2.0, 0.5]
        distance = self._calculateDistance(currentPosition, targetPosition)
        if (state[0] > 2.2 or state[0] < -0.2 or
            state[1] > 2.2 or state[1] <  -0.2 or
            state[2] > 1 or state[2] <   0):
            reward = -500

        if self.collision == True:
            reward = -300

        if distance <= 0.1:
            reward = 500

        return reward
    
    def is_done(self, state):
        # Check if the episode is done based on the current state
        # ...
        done = False
        currentPosition = state[0:3]
        targetPosition = [2.0, 2.0, 0.5]
        distance = self._calculateDistance(currentPosition, targetPosition)
        if (state[0] > 2.2 or state[0] < -0.2 or
            state[1] > 2.2 or state[1] <  -0.2 or
            state[2] > 1 or state[2] <   0):
            done = True

        if distance <= 0.1:
            done = True

        if self.collision == True:
            done = True
            self.collision = False

        return done

    def close(self):
        # 关闭环境
        p.disconnect()

    def _computeObstacleDis(self, position1, position2) -> list:
        dis1 = math.sqrt(
            (position1[0] - position2[0]) ** 2 +
            (position1[1] - position2[1]) ** 2 +
            (position1[2] - position2[2]) ** 2
        )

        dis2 = math.sqrt(
            (position1[0] - position2[3]) ** 2 +
            (position1[1] - position2[4]) ** 2 +
            (position1[2] - position2[5]) ** 2
        )
        return [dis1, dis2]

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

    def _housekeeping(self):
        self._addObstacles()
    
    def _addObstacles(self):
        """Add obstacles to the environment.

        Extends the superclass method and add the gate build of cubes and an architrave.

        """
        cube_small_file_path = '/root/miniconda3/envs/myconda/lib/python3.9/site-packages/pybullet_data/cube_small.urdf'

        obstaclePostion = [(0.5, 0.5), (1.2, 1.2)]

        for obstacle in obstaclePostion:
            i = obstacle[0]
            j = obstacle[1]
            obstacleId = p.loadURDF(cube_small_file_path,
                        [i, j, 0.5],
                        p.getQuaternionFromEuler([0, 0, 0]),
                        physicsClientId=self.client,
                        useFixedBase=True
                        )
            self.obstacleIds.append(obstacleId)

    def _checkCollision(self):
        for obstacleId in self.obstacleIds:
            contactPoints = p.getContactPoints(self.drone, obstacleId)
            if len(contactPoints) > 0:
                print(f"Collision with obstacle {obstacleId} detected!")
                self.collision = True
                break
    

