import gym
import numpy as np

import gym_examples  # noqa: F401 - ensure registry


def test_sorting_center_smoke():
    env = gym.make('gym_examples/SortingCenter-v0', disable_env_checker=True)
    obs, info = env.reset(seed=123)

    # 基本键值存在
    assert isinstance(obs, dict)
    assert 'obs' in obs
    assert 'matrix' in obs
    assert isinstance(info, dict)

    # 观测形状
    assert isinstance(obs['obs'], np.ndarray)
    assert len(obs['obs'].shape) == 2  # (agent_num, features)
    assert isinstance(obs['matrix'], np.ndarray)

    # info 指标
    for k in ['throughput', 'tc_throughput', 'battery_cost', 'global_step']:
        assert k in info

    # 连续 10 步随机动作
    agent_num = obs['obs'].shape[0]
    for _ in range(10):
        actions = [env.action_space.sample() for _ in range(agent_num)]
        obs, rew, term, trunc, info = env.step(actions)
        assert isinstance(obs, dict)
        assert isinstance(rew, (list, tuple))
        assert len(rew) == agent_num
        assert isinstance(term, (bool, np.bool_))
        assert isinstance(trunc, (bool, np.bool_))
        assert isinstance(info, dict)

    env.close()


