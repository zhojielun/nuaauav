from gym.envs.registration import register

# Single Agent Env
# Without Obstacles Env
register(
    id='WithoutObstaclesEnv-v0',
    entry_point='drones.envs.withoutObstacles.withoutObstacles:WithoutObstaclesV0'
)

# Fixed Obstacles Env
register(
    id='FixedObstaclesEnv-v0',
    entry_point='drones.envs.fixedObstacles.fixedObstacles:FixedObstaclesV0'
)

# Fixed MultiAgentWithoutObstacle Env
register(
    id='multiAgentWithoutObstacleEnv-v0',
    entry_point='drones.envs.multiAgentWithoutObstacle.multiAgentWithoutObstacle:multiAgentWithoutObstacleV0'
)