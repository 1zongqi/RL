import time
import numpy as np


# all_agent = {}
# agent_num = 5
# for i in range(agent_num):
#     all_agent['agent_' + str(i)] = {}
#
# print(all_agent)
# start = time.time()
# for i in range(100):
#     print(time.time() - start)

# subgoals = np.stack([
#     np.random.uniform(*(1, 2), size=1024),
#     np.random.uniform(*(3, 4), size=1024),
# ], axis=-1)  # 对能选的子目标范围随机采样
# print(subgoals.shape)
import numpy as np

# 创建一个递增的obs数组
obs = np.arange(1, 25).reshape((4, 3, 2))
print("原始 obs 数组：")
print(obs)

# 创建示例的rows和cols数组，假设它们包含随机的行和列索引
rows = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2]])
cols = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0]])

# 使用高级索引从obs中选择指定的元素
selected_obs = obs[rows, cols]

# 输出选择的结果和其形状
print("\n选择的 obs 数组：")
print(selected_obs)
print("选择的 obs 数组的形状：", selected_obs.shape)
