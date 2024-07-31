import gym
import time
import math
from drones.envs.withoutObstacles.withoutObstacles import WithoutObstaclesV0
from drones.envs.fixedObstacles.fixedObstacles import FixedObstaclesV0
from drones.envs.multiAgentWithoutObstacle.multiAgentWithoutObstacle import multiAgentWithoutObstacleV0

# parameters
# env_name = "FixedObstaclesEnv-v0"
# env_name = "WithoutObstaclesEnv-v0"
env_name = "multiAgentWithoutObstacleEnv-v0"
num_agents = 3

env = gym.make(env_name, num_agents=num_agents, GUI=True)

while True:
    s = env.reset()
    print("=========================s", s)
    time.sleep(3)
