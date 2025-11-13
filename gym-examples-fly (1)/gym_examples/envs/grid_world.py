import time
import math

import gym
from gym import spaces
import pygame
import numpy as np
import random


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, agent_num=30):
        self.all_agent = {}
        self.cell_size = 20
        self.black_cell = []
        self.black_pos = []
        for i in range(3, 36, 3):
            for j in range(2, 77, 3):
                self.black_cell.append([i, j, 0])
                self.black_pos.append([i, j])
        self.black_cell.append([0, 0, 0])
        self.black_cell.append([0, 76, 0])
        self.black_cell.append([36, 0, 0])
        self.black_cell.append(([36, 76, 0]))

        self.row_size = 37 * self.cell_size
        self.rol_size = 77 * self.cell_size
        self.map = [[0] * 77 for _ in range(37)]
        for i in range(agent_num):
            self.all_agent['agent_' + str(i)] = {'agent_location': [], 'agent_obs': {}, 'action_space': [],
                                                 'capacity': 1, 'task': [-1] * 7, 'target': [-1, -1]}

        self.all_location = []
        self.entry_pos = [size - 1, size - 1]
        # for i in range(agent_num):
        #     self.all_location.append(self.all_agent.get('agent_'+str(i)).get('agent_location'))
        self.center = size // 2
        self.hub_num = 2
        self.hub = []
        for i in range(self.hub_num):
            self.hub.append([self.center, self.center])
        self.hub_c = []
        for i in range(self.hub_num):
            self.hub_c.append(100)
        self.target_c = 0

        self.obs_id = []  # 所有货物的id
        self.hub1_obs = []
        self.hub2_obs = []
        self.entr_obs = []
        self.exit = [[0, 0, 0], [0, 3, 0], [0, 6, 0], [0, 9, 0]]
        self.entrance_obs = [[] for _ in range(50)]
        self.entrance = [[9, 0, 0], [9, 3, 0], [9, 6, 0], [9, 9, 0]]
        self.entrance_pos = []
        self.take = []
        for i in range(2, 77, 3):  # 取货点
            self.take.append([0, i, 0])
            self.take.append([36, i, 0])
            self.entrance_pos.append([0, i])
            self.entrance_pos.append([36, i])
        self.entrance = self.take

        self.agent_num = agent_num
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(-1, 86401 - 1, shape=(400,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right",'local'
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),  # 下
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        obs = {}
        # obs = {'obs': []}
        # 按 deadline进行排序
        for i in range(len(self.entrance)):
            self.entrance_obs[i].sort(key=lambda x: x[-1], reverse=False)
        # self.hub1_obs.sort(key=lambda x: x[-1], reverse=False)
        # self.hub2_obs.sort(key=lambda x: x[-1], reverse=False)
        # self.entr_obs.sort(key=lambda x: x[-1], reverse=False)

        # for i in range(len(self.entrance)):
        #     # obs['obs'].append(self.hub1_obs[i][:])
        #     if i == 0:
        #         obs['obs'] = self.entrance_obs[i][0][:]
        #     else:
        #         obs['obs'] += self.entrance_obs[i][0][:]

        for i in range(10):
            for j in range(len(self.entrance)):
                if i == 0 and j == 0:
                    obs['obs'] = self.entrance_obs[j][i][:]
                else:
                    obs['obs'] += self.entrance_obs[j][i][:]

        # for i in range(5):
        #     obs['obs'] += self.hub2_obs[i][:]
        #     # obs['obs'].append(self.hub2_obs[i][:])
        #
        # obs['obs'] += self.entr_obs[0][:]
        # for i in range(self.agent_num):
        #     obs['obs_' + str(i)] = self.all_agent["agent_" + str(i)]["agent_location"]
        # obs['target'] = self._target_location
        # return {"agent": self._agent_location, "target": self._target_location}
        # print("new_obs:", obs['obs'])
        # print("new_obs_shape:", len(obs['obs']))
        return obs

    def get_agent_obs(self):
        obs = []
        res = {'matrix': []}
        for agent in self.all_agent:
            temp = self.all_agent[agent]['agent_location'] + self.all_agent[agent]['target']
            obs += temp[:]
        obs = np.array(obs)
        res['obs'] = obs
        for i in range(self.agent_num):
            matrix_obs = np.zeros((3, 5, 5))
            agt_pos = self.all_agent["agent_" + str(i)]["agent_location"]
            obstacles = np.ones((5, 5), dtype=int)
            for offsetx in range(-2, 3):
                for offsety in range(-2, 3):
                    x, y = agt_pos[0] + offsetx, agt_pos[1] + offsety
                    # 检查位置是否在边界内
                    if 0 <= x < 36 and 0 <= y < 76:
                        # 如果位置是障碍物，将对应的位置置为1
                        if list((x, y)) in self.black_pos:
                            obstacles[offsetx + 2, offsety + 2] = 1
                        else:
                            obstacles[offsetx + 2, offsety + 2] = 0
            matrix_obs[0] = obstacles

            agent_matrix = np.zeros((5, 5), dtype=int)
            for offsetx in range(-2, 3):
                for offsety in range(-2, 3):
                    x, y = agt_pos[0] + offsetx, agt_pos[1] + offsety
                    # 检查位置是否在边界内
                    if 0 <= x < 36 and 0 <= y < 76:
                        # 如果位置有其他智能体（包括自己），将对应的位置置为1
                        if list((x, y)) in self.all_location[329:]:
                            agent_matrix[offsetx + 2, offsety + 2] = 1
            matrix_obs[1] = agent_matrix

            target_matrix = np.zeros((5, 5), dtype=int)
            for offsetx in range(-2, 3):
                for offsety in range(-2, 3):
                    x, y = agt_pos[0] + offsetx, agt_pos[1] + offsety
                    # 检查位置是否在边界内
                    if 0 <= x < 36 and 0 <= y < 76:
                        # 如果位置有其他智能体（包括自己），将对应的位置置为1
                        if list((x, y)) == self.all_agent['agent_' + str(i)]['target']:
                            target_matrix[offsetx + 2, offsety + 2] = 1
            matrix_obs[2] = target_matrix
            res['matrix'].append(matrix_obs)
        res['obs'] = np.array(res['obs']).reshape(self.agent_num, -1)
        res['matrix'] = np.array(res['matrix'])
        return res

    def _get_info(self):
        info = {}
        for i in range(self.agent_num):
            # print("i:", i)
            # print("tar", self._target_location)
            # print("tar2", self.all_agent["agent_" + str(i)]["agent_location"])
            info["info_" + str(i)] = np.linalg.norm(np.array(self._target_location) -
                                                    np.array(self.all_agent["agent_" + str(i)]["agent_location"]),
                                                    ord=1)
        return info
        # return {
        #     "distance": np.linalg.norm(
        #         self._agent_location - self._target_location, ord=1
        #     )
        # }

    def assign_goal(self):
        task = self._get_obs()['obs'][:]
        xy_list = []
        for i in range(len(task) // 7):
            xy_list.append(task[i * 7 + 1:i * 7 + 3])
        # xy_list.sort(key=lambda x: x[-1], reverse=False)
        for agent in self.all_agent:
            target, del_task = near_goal(self.all_agent[agent]['agent_location'], xy_list)
            self.all_agent[agent]['target'] = target[:]
            xy_list.pop(del_task)

    def get_new_goal(self, agent_pos):  # 从所有的货物观测里返回最紧急的
        # version-1
        # task = []
        # for i in range(len(self.entrance)):
        #     self.entrance_obs[i].sort(key=lambda x: x[-1], reverse=False)
        # for i in range(len(self.entrance)):
        #     task.append(self.entrance_obs[i][0][:])
        # task.sort(key=lambda x: x[-1], reverse=False)
        # return task[0][1:3]
        # version -1
        min_distance = float('inf')
        closest_coordinate = None

        for coordinate in self.entrance_pos:
            distance = math.sqrt((coordinate[0] - agent_pos[0]) ** 2 + (coordinate[1] - agent_pos[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_coordinate = coordinate

        return closest_coordinate

    def get_state(self):
        state_dict = dict(agent_obs=None, task=dict(hub=[], entry=[]))
        state_dict['agent_obs'] = self.get_agent_obs()['obs']
        for i in range(len(self.hub1_obs)):
            state_dict['task']['hub'].append(self.hub1_obs[i])
        for i in range(len(self.hub2_obs)):
            state_dict['task']['hub'].append(self.hub2_obs[i])
        for i in range(len(self.entr_obs)):
            state_dict['task']['entry'].append(self.entr_obs[i])
        return state_dict

    def is_to_target(self):
        info = [{'success': False} for _ in range(self.agent_num)]
        for i in range(self.agent_num):
            if self.all_agent['agent_' + str(i)]['agent_location'] == self.all_agent['agent_' + str(i)]['target']:
                info[i]['success'] = True
        return info[:]

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # print(type(seed))
        super().reset(seed=seed)
        self.map = [[0] * 77 for _ in range(37)]
        for item in self.black_cell:
            x = item[0]
            y = item[1]
            self.map[x][y] = 1

        self.target_c = 0
        for i in range(self.agent_num):
            self.all_agent['agent_' + str(i)]["capacity"] = 1
        for i in range(self.hub_num):
            self.hub_c[i] = 100
        self._target_location = []
        self.all_location = []
        # for i in range(len(self.exit)):
        #     self.all_location.append(self.exit[i][:2])
        for i in range(len(self.black_cell)):
            self.all_location.append(self.black_cell[i][:2])
        for i in range(len(self.entrance)):
            self.all_location.append(self.entrance[i][:2])
        self._target_location.append(0)
        self._target_location.append(0)
        # self.all_location.append(self._target_location)
        # self.all_location.append(self.entry_pos)

        # self.all_location.append(self.hub[0])
        # for i in range(1, self.hub_num):
        #     while self.hub[i] in self.all_location:
        #         h1, h2 = self.np_random.integers(0, self.size, size=2, dtype=int)
        #         self.hub[i][0] = h1
        #         self.hub[i][1] = h2
        #     self.all_location.append(self.hub[i])

        a = self.np_random.integers(0, 37, size=1, dtype=int)
        b = self.np_random.integers(0, 77, size=1, dtype=int)
        self.all_agent['agent_' + str(0)]['agent_location'] = []
        # self.all_location = []
        self.all_agent['agent_' + str(0)]['agent_location'].append(3)
        self.all_agent['agent_' + str(0)]['agent_location'].append(34)
        self.all_agent['agent_' + str(0)]['task'][0] = -1
        self.all_location.append(self.all_agent['agent_' + str(0)]['agent_location'])
        for i in range(1, self.agent_num):
            self.all_agent['agent_' + str(i)]['agent_location'] = self.all_agent['agent_' + str(0)]['agent_location'][:]
            self.all_agent['agent_' + str(i)]['task'][0] = -1
        for i in range(1, self.agent_num):
            while self.all_agent['agent_' + str(i)]['agent_location'] in self.all_location:
                l1 = self.np_random.integers(0, 37, size=1, dtype=int)
                l2 = self.np_random.integers(0, 77, size=1, dtype=int)
                self.all_agent['agent_' + str(i)]['agent_location'][0] = l1[0]
                self.all_agent['agent_' + str(i)]['agent_location'][1] = l2[0]
            self.all_location.append(self.all_agent['agent_' + str(i)]['agent_location'])

        # object_obs seting
        self.obs_id = []  # 所有货物观测的存放列表
        # self.hub1_obs = []
        for i in range(len(self.entrance)):
            for j in range(20):
                temp = []
                id = random.randint(0, 1000)
                while id in self.obs_id:
                    id = random.randint(0, 1000)
                temp.append(id)
                self.obs_id.append(id)
                temp.append(self.entrance[i][0])
                temp.append(self.entrance[i][1])
                exit_id = random.randint(0, 274)
                temp.append(self.black_cell[exit_id][0])
                temp.append(self.black_cell[exit_id][1])
                temp.append(-1)
                temp.append(40000)
                self.entrance_obs[i].append(temp[:])
                self.entrance[i][-1] += 1
        # for i in range(100):
        #     temp = []
        #     id = random.randint(0, 500)
        #     while id in self.obs_id:
        #         id = random.randint(0, 500)
        #     temp.append(id)
        #     self.obs_id.append(id)  # 货物id
        #     temp.append(self.center)  # 货物xy
        #     temp.append(self.center)
        #     temp.append(self._target_location[0])
        #     temp.append(self._target_location[1])
        #     temp.append(-1)  # 飞毯id
        #     temp.append(40000)
        #     self.hub1_obs.append(temp[:])

        # self.hub2_obs = []
        # for i in range(self.hub_c[1]):
        #     temp = []
        #     id = random.randint(0, 500)
        #     while id in self.obs_id:
        #         id = random.randint(0, 500)
        #     temp.append(id)
        #     self.obs_id.append(id)
        #     temp.append(self.hub[1][0])
        #     temp.append(self.hub[1][1])
        #     temp.append(self._target_location[0])
        #     temp.append(self._target_location[1])
        #     temp.append(-1)
        #     temp.append(40000)
        #     self.hub2_obs.append(temp[:])

        # Entrance obs seting
        # self.entr_obs = []
        # for i in range(100):
        #     temp = []
        #     id = random.randint(0, 500)
        #     while id in self.obs_id:
        #         id = random.randint(0, 500)
        #     temp.append(id)
        #     self.obs_id.append(id)
        #     temp.append(self.size - 1)
        #     temp.append(self.size - 1)
        #     if i % 2 == 0:  # 进货时的目标位置
        #         temp.append(self.hub[0][0])
        #         temp.append(self.hub[0][1])
        #     else:
        #         temp.append(self.hub[1][0])
        #         temp.append(self.hub[1][1])
        #     temp.append(-1)
        #     temp.append(random.randint(40000, 80000))
        #     self.entr_obs.append(temp[:])

        self.assign_goal()  # 分配目标
        observation = self.get_agent_obs()
        # observation = np.array(observation['obs']).reshape(self.agent_num, -1)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.fill_map()
        return observation, info

    def step(self, action):
        reward = [-1] * self.agent_num
        direction = []
        repeat_list = []  # 存放发生碰撞的位置
        for act in action:
            direction.append(self._action_to_direction[act])

        # pre_list = []
        # for i in range(self.agent_num):
        #     pre_list.append(self.all_agent["agent_" + str(i)]["agent_location"])

        # pre_list = []
        # for i in range(self.agent_num):
        #     pre_list.append(self.all_agent["agent_" + str(i)]["agent_location"])

        no_move = []
        move_id = []
        for i in range(len(action)):
            agent_pos = np.array(self.all_agent["agent_" + str(i)]["agent_location"]) + np.array(direction[i])
            agent_pos_x = np.clip(agent_pos[0], 0, 36)
            agent_pos_y = np.clip(agent_pos[1], 0, 76)
            agent_pos[0] = agent_pos_x
            agent_pos[1] = agent_pos_y
            buf_pos = list(agent_pos)
            # buf_pos = list(np.clip(
            #     np.array(self.all_agent["agent_" + str(i)]["agent_location"]) + np.array(direction[i]), 0,
            #     self.size - 1))
            if buf_pos == self.all_agent["agent_" + str(i)]["agent_location"]:
                no_move.append(self.all_agent["agent_" + str(i)]["agent_location"])
            else:
                move_id.append(i)

        # random.shuffle(move_id)
        for item in move_id:
            # buf_pos = list(np.clip(np.array(self.all_agent["agent_" + str(item)]["agent_location"])
            #                        + np.array(direction[item]), 0, self.size - 1))
            agent_pos = np.array(self.all_agent["agent_" + str(item)]["agent_location"]) + np.array(direction[item])
            agent_pos_x = np.clip(agent_pos[0], 0, 36)
            agent_pos_y = np.clip(agent_pos[1], 0, 76)
            agent_pos[0] = agent_pos_x
            agent_pos[1] = agent_pos_y
            buf_pos = list(agent_pos)
            x1 = self.all_agent["agent_" + str(item)]["agent_location"][0]
            y1 = self.all_agent["agent_" + str(item)]["agent_location"][1]
            x2 = buf_pos[0]
            y2 = buf_pos[1]
            if self.map[x2][y2] == 0:
                self.all_agent["agent_" + str(item)]["agent_location"] = buf_pos[:]
                self.map[x2][y2] = 1
                self.map[x1][y1] = 0

        for i in range(len(action)):
            self.all_location[329 + i] = self.all_agent["agent_" + str(i)]["agent_location"]
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        # An episode is done iff the agent has reached the target
        set_list = []
        for item in self.all_location[329:]:
            if item not in set_list:
                set_list.append(item)
            else:
                repeat_list.append(item)

        if len(set_list) == self.agent_num:
            terminated = False
        else:
            terminated = True
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # reward = 1 if terminated else 0  # Binary sparse rewards
        if terminated:  # 碰撞直接返回
            print("all:", self.all_location)
            print("set:", set_list)
            print("pre_pos:", pre_list)
            print("act:", direction)
            time.sleep(100)
            observation = self._get_obs()
            info = self._get_info()
            if self.render_mode == "human":
                self._render_frame()
            # print("here1")
            return observation, reward, terminated, False, info

        # for i in range(len(self.entrance)):  # 在进货口附近检测进货
        #     for j in range(self.agent_num):
        #         arr1 = np.array(self.all_agent['agent_' + str(j)]["agent_location"])
        #         arr2 = np.array(self.entrance[i][:2])
        #         dist = np.sqrt(np.sum(np.square(arr1 - arr2)))
        #         if dist <= np.sqrt(2) and self.all_agent['agent_' + str(j)]["capacity"] == 1 and \
        #                 self.all_agent['agent_' + str(j)]['target'] == self.entrance[i][:2]:
        #             self.all_agent['agent_' + str(j)]["capacity"] = 0
        #             self.all_agent['agent_' + str(j)]['task'] = self.entrance_obs[i][0][:]
        #             self.all_agent['agent_' + str(j)]['target'] = self.entrance_obs[i][0][3:5]  # 设定飞毯的目标
        #             self.obs_id.remove(self.entrance_obs[i][0][0])
        #             self.entrance_obs[i].pop(0)
        #             self.entrance[i][-1] -= 1

        for ag_id in range(self.agent_num):
            arr1 = np.array(self.all_agent['agent_' + str(ag_id)]["agent_location"])
            arr2 = np.array(self.all_agent['agent_' + str(ag_id)]['target'])
            blue_list = [[arr2[0] - 1, arr2[1]],
                         [arr2[0] + 1, arr2[1]],
                         [arr2[0], arr2[1] - 1],
                         [arr2[0], arr2[1] + 1]]
            if np.sqrt(np.sum(np.square(arr1 - arr2))) <= np.sqrt(2) and \
                    self.all_agent['agent_' + str(ag_id)]["capacity"] == 1:
                reward[ag_id] += 1
                self.all_agent['agent_' + str(ag_id)]["capacity"] = 0
                self.all_agent['agent_' + str(ag_id)]['target'] = random.choice(self.black_cell[:-4])[:2]
            elif self.all_agent['agent_' + str(ag_id)]["capacity"] == 0 and \
                    self.all_agent['agent_' + str(ag_id)]["agent_location"] in blue_list:
                reward[ag_id] += 1
                self.all_agent['agent_' + str(ag_id)]["capacity"] = 1
                self.all_agent['agent_' + str(ag_id)]['target'] = random.choice(self.entrance_pos)

        # for i in range(len(self.black_cell[:-4])):
        #     for j in range(self.agent_num):
        #         # arr1 = np.array(self.all_agent['agent_' + str(j)]["agent_location"])
        #         blue_list = [[self.black_cell[i][0] - 1, self.black_cell[i][1]],
        #                      [self.black_cell[i][0] + 1, self.black_cell[i][1]],
        #                      [self.black_cell[i][0], self.black_cell[i][1] - 1],
        #                      [self.black_cell[i][0], self.black_cell[i][1] + 1]]
        #         # arr2 = np.array(self.black_cell[i][:2])
        #         # dist = np.sqrt(np.sum(np.square(arr1 - arr2)))
        #         if self.all_agent['agent_' + str(j)]["agent_location"] in blue_list and \
        #                 self.all_agent['agent_' + str(j)]["capacity"] == 0 and \
        #                 self.all_agent['agent_' + str(j)]['target'] == self.black_cell[i][:2]:
        #             self.all_agent['agent_' + str(j)]["capacity"] = 1
        #             self.black_cell[i][-1] += 1
        #             self.all_agent['agent_' + str(j)]['task'] = [-1] * 7
        #             self.all_agent['agent_' + str(j)]['target'] = self.get_new_goal(self.all_agent['agent_' + str(j)]["agent_location"])
        #             reward[j] = 1

        # info_success = self.is_to_target()
        # if self.target_c == 10:
        #     terminated = True
        # observation = self._get_obs()
        observation = self.get_agent_obs()
        info = self._get_info()
        # print("all:", self.all_location)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.rol_size, self.row_size))  # (self.window_size, self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.rol_size, self.row_size))
        canvas.fill((255, 255, 255))
        row_pix_square_size = (
                self.row_size / 37
        )  # The size of a single grid square in pixels
        rol_pix_square_size = (self.rol_size / 77)
        pix_square_size = (self.rol_size / 77)

        # number
        WHITE = (255, 255, 255)

        # draw black cell and blue_cell
        for item in self.black_cell:
            rect_black_cell = pygame.Rect(
                pix_square_size * np.array(item[:2][::-1]),
                (pix_square_size, pix_square_size),
            )
            pygame.draw.rect(
                canvas,
                (0, 0, 0), rect_black_cell,
            )
            number_black_cell = str(item[-1])
            font_black_cell = pygame.font.Font(None, 20)
            text_black_cell = font_black_cell.render(number_black_cell, True, WHITE)
            text_tect_black_cell = text_black_cell.get_rect(center=rect_black_cell.center)
            canvas.blit(text_black_cell, text_tect_black_cell)

            rect_blue_cell1 = pygame.Rect(pix_square_size * np.array([item[1] - 1, item[0]]),
                                          (pix_square_size, pix_square_size), )
            pygame.draw.rect(canvas, (0, 191, 255), rect_blue_cell1, )

            rect_blue_cell2 = pygame.Rect(pix_square_size * np.array([item[1] + 1, item[0]]),
                                          (pix_square_size, pix_square_size), )
            pygame.draw.rect(canvas, (0, 191, 255), rect_blue_cell2, )

            rect_blue_cell3 = pygame.Rect(pix_square_size * np.array([item[1], item[0] - 1]),
                                          (pix_square_size, pix_square_size), )
            pygame.draw.rect(canvas, (0, 191, 255), rect_blue_cell3, )

            rect_blue_cell4 = pygame.Rect(pix_square_size * np.array([item[1], item[0] + 1]),
                                          (pix_square_size, pix_square_size), )
            pygame.draw.rect(canvas, (0, 191, 255), rect_blue_cell4, )

        # draw exit
        # for i in range(len(self.exit)):
        #     rect_exit = pygame.Rect(
        #         pix_square_size * np.array(self.exit[i][:2]),
        #         (pix_square_size, pix_square_size),
        #     )
        #     pygame.draw.rect(
        #         canvas,
        #         (0, 0, 0), rect_exit,
        #     )
        #     number_exit = str(self.exit[i][-1])
        #     font_exit = pygame.font.Font(None, 20)
        #     text_exit = font_exit.render(number_exit, True, WHITE)
        #     text_rect_exit = text_exit.get_rect(center=rect_exit.center)
        #     canvas.blit(text_exit, text_rect_exit)

        # we draw 入口
        for i in range(len(self.entrance)):
            rect_entrance = pygame.Rect(
                pix_square_size * np.array(self.entrance[i][:2][::-1]),
                (pix_square_size, pix_square_size),
            )
            pygame.draw.rect(
                canvas,
                (255, 0, 0), rect_entrance,
            )
            number_entrance = str(self.entrance[i][-1])
            font_entrance = pygame.font.Font(None, 20)
            text_entrance = font_entrance.render(number_entrance, True, WHITE)
            text_rect_entrance = text_entrance.get_rect(center=rect_entrance.center)
            canvas.blit(text_entrance, text_rect_entrance)

        # draw agent
        for i in range(self.agent_num):
            pygame.draw.circle(canvas, (0, 0, 255),
                               (np.array(self.all_agent['agent_' + str(i)]["agent_location"][::-1]) + 0.5)
                               * pix_square_size, pix_square_size / 3, )
            if len(self.all_agent['agent_' + str(i)]["task"]):
                task_id = self.all_agent['agent_' + str(i)]["task"][0]
                font_task = pygame.font.Font(None, 15)
                text_task = font_task.render(str(task_id), True, WHITE)
                text_rect_task = text_task.get_rect(center=
                                                    (np.array(
                                                        self.all_agent['agent_' + str(i)]["agent_location"][::-1]) +
                                                     0.5) * pix_square_size)
                canvas.blit(text_task, text_rect_task)
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (self._agent_location + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )
        # Finally, add some gridlines
        for x in range(0, self.rol_size, self.cell_size):
            pygame.draw.line(canvas, 0, (x, 0), (x, self.row_size))
        for y in range(0, self.row_size, self.cell_size):
            pygame.draw.line(canvas, 0, (0, y), (self.rol_size, y))
        # for x in range(self.size + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, pix_square_size * x),
        #         (self.window_size, pix_square_size * x),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (pix_square_size * x, 0),
        #         (pix_square_size * x, self.window_size),
        #         width=3,
        #     )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def fill_map(self):
        for i in range(self.agent_num):
            x = self.all_agent['agent_' + str(i)]["agent_location"][0]
            y = self.all_agent['agent_' + str(i)]["agent_location"][1]
            self.map[x][y] = 1


def near_goal(pos, task):
    arr1 = np.array(pos)
    arr2 = np.array(task[0])
    dist = np.sqrt(np.sum(np.square(arr1 - arr2)))
    index = 0
    for i in range(len(task)):
        arr2 = np.array(task[i])
        if np.sqrt(np.sum(np.square(arr1 - arr2))) < dist:
            dist = np.sqrt(np.sum(np.square(arr1 - arr2)))
            index = i
    return task[index], index
