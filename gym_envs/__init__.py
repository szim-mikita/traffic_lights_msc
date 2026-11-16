from gymnasium.envs.registration import register

register(
    id='StateTrafficEnv',
    entry_point='gym_envs.envs:StateTrafficEnv'
)
register(
    id='StateTrafficMOEnv',
    entry_point='gym_envs.envs:StateTrafficMOEnv'
)
register(
    id='TrafficEnv',
    entry_point='gym_envs.envs:TrafficEnv'
)
register(
    id='SumoRLEnv',
    entry_point='gym_envs.envs:SumoRLEnv'
)