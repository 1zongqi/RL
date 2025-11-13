import math
import random
from typing import Dict, List, Tuple, Optional

import gym
import numpy as np
import pygame
from gym import spaces


"""
迁移说明（Migration Note）
本文件将现有的基于 Pygame 的 AGV 模拟环境重构为 MOGCRL 论文中的"Sorting-Center"仓库场景。
与旧环境（GridWorldEnv）对齐点：
- 仍然暴露 reset(), step(actions), render(), seed(), set_config(dict) 接口。
- 观测/动作接口保持兼容：
  - 动作空间 Discrete(5) 不变（上、下、左、右、停）。
  - 观测保持 get_agent_obs 的结构：{'obs': np.ndarray(shape=(agent_num, -1)), 'matrix': np.ndarray}，其中每个 agent 的 obs 为 [row, col, tgt_row, tgt_col, deadline_remaining]。
- 如需要从旧环境切换：将 gym.make 的 id 切为 'gym_examples/SortingCenter-v0' 即可。

新环境核心差异：
- 使用 77x37 大网格；
- 静态区域包含：PICKUP（上下边界均匀 50 个）、SHELVES（279 个黑色货架，既是障碍也是投递目标）、CHARGING（12 个）；
- 任务规则：从随机 pickup 出生，目标是随机货架的四邻域相邻格；
- 并行步进与碰撞回滚：同格冲突时，按 agent_id 升序，保留最小的智能体，其他回滚（保证实验可重复）；
- 可选能量模型与充电站逻辑（可配置自动充电模式）；
- 指标：throughput、TC-throughput（完成时剩余时间≥0）、battery_cost 等。
"""


# 单一 2D map 的枚举值
CELL_FREE = 0
CELL_PICKUP = 1
CELL_DELIVERY = 2  # 不再单独使用；黑色货架既是障碍也是投递目标
CELL_OBSTACLE = 3
CELL_CHARGING = 4


def is_adjacent_to_delivery(pos: List[int], delivery_cell: Tuple[int, int]) -> bool:
    return (abs(pos[0] - delivery_cell[0]) + abs(pos[1] - delivery_cell[1])) == 1


def build_sorting_center_masks(width: int = 77, height: int = 37) -> Dict[str, object]:
    """构建与参考图一致的 Sorting-Center 静态布局。

    规范（精确匹配参考图）：
    - 网格 77×37
    - 取件点：顶边 y=0 与底边 y=36，各 25 个（共 50），x=2,5,8,...,74（步长3）
    - 充电桩：左4( x=0 )、右4( x=76 )、上2( y=0 )、下2( y=36 )，精确坐标 (见代码)
    - 货架：x∈[2,74] y∈[2,34] 步长3 的点阵 (275个) + 四角4个 = 279
    - 投递集合 = 货架集合
    """
    pickup_cells: List[Tuple[int, int]] = []
    charging_cells: List[Tuple[int, int]] = []
    shelves: List[Tuple[int, int]] = []

    # 1) PICKUP 顶/底边 x=2,5,8,...,74（步长3）
    for x in range(2, 75, 3):  # 2,5,8,...,74
        pickup_cells.append((0, x))    # 顶边 y=0
        pickup_cells.append((36, x))   # 底边 y=36

    # 2) CHARGING 精确坐标
    # 左边 x=0：y=9,16,23,30
    for y in [9, 16, 23, 30]:
        charging_cells.append((y, 0))
    # 右边 x=76：y=9,16,23,30
    for y in [9, 16, 23, 30]:
        charging_cells.append((y, 76))
    # 顶边 y=0：x=24,48
    for x in [24, 48]:
        charging_cells.append((0, x))
    # 底边 y=36：x=24,48
    for x in [24, 48]:
        charging_cells.append((36, x))

    # 3) SHELVES：四角 + 内部点阵
    # 四角（黑色货架）
    corners = [(0, 0), (0, 76), (36, 0), (36, 76)]
    shelves.extend(corners)

    # 内部点阵：x∈[2,74] y∈[3,35] 步长3（下移1格）
    occupied = set(pickup_cells) | set(charging_cells) | set(corners)
    for y in range(3, 36, 3):  # 3,6,9,...,35（下移1格）
        for x in range(2, 75, 3):  # 2,5,8,...,74
            if (y, x) not in occupied:
                shelves.append((y, x))

    # 检查无重叠
    all_cells = set(pickup_cells) | set(charging_cells) | set(shelves)
    assert len(all_cells) == len(pickup_cells) + len(charging_cells) + len(shelves), \
        "存在重叠单元格"

    # 断言精确计数
    assert len(pickup_cells) == 50, f"pickup_cells count={len(pickup_cells)} != 50"
    assert len(charging_cells) == 12, f"charging_cells count={len(charging_cells)} != 12"
    assert len(shelves) == 279, f"shelves count={len(shelves)} != 279"

    return dict(
        pickup_cells=pickup_cells,
        shelves=shelves,
        obstacles=shelves,  # 对外保持 'obstacles' 字段兼容
        charging_cells=charging_cells,
    )


class SortingCenterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, size: int = 5, agent_num: int = 32, config: Optional[Dict] = None):
        # 与旧环境兼容的属性/接口占位
        self.size = size
        self.agent_num = agent_num
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # 配置
        self.config = dict(
            grid_width=77,
            grid_height=37,
            task_deadline=70,
            episode_len=400,
            use_energy=True,
            battery_init=220,
            move_cost=1,
            stay_cost=0,
            low_battery_threshold=30,
            auto_charge=True,  # 是否自动切换到充电模式（True：环境自动切换；False：由策略决定）
            fps=10,
            seed=None,
            num_agents=agent_num,
        )
        if config:
            self.config.update(config)

        self.width = int(self.config.get('grid_width', 77))
        self.height = int(self.config.get('grid_height', 37))

        # 渲染栅格缩放
        self.cell_size = 20
        self.row_size = self.height * self.cell_size
        self.rol_size = self.width * self.cell_size

        # 静态布局
        masks = build_sorting_center_masks(self.width, self.height)
        self.pickup_cells: List[Tuple[int, int]] = masks['pickup_cells']
        self.shelves: List[Tuple[int, int]] = masks['obstacles']  # 货架=障碍=投递集合
        self.obstacles: List[Tuple[int, int]] = masks['obstacles']
        self.charging_cells: List[Tuple[int, int]] = masks['charging_cells']

        # 单一 2D map
        self.map = [[CELL_FREE] * self.width for _ in range(self.height)]
        for r, c in self.pickup_cells:
            self.map[r][c] = CELL_PICKUP
        for r, c in self.obstacles:
            self.map[r][c] = CELL_OBSTACLE
        for r, c in self.charging_cells:
            self.map[r][c] = CELL_CHARGING

        # 代理结构
        self.all_agent: Dict[str, Dict] = {}
        for i in range(self.agent_num):
            self.all_agent['agent_' + str(i)] = {
                'agent_location': [],
                'agent_obs': {},
                'action_space': [],
                'capacity': 1,
                'task': [-1] * 7,  # 兼容旧键
                'target': [-1, -1],
                'battery': self.config['battery_init'],
                'task_deadline_left': self.config['task_deadline'],
                'current_delivery_cell': None,  # (r,c)
                'prefer_charge': False,
                'charging_mode': False,
                'saved_delivery_cell': None,
                'saved_target': None,
            }

        # 观测与动作空间保持一致
        # 每个agent的obs: [row, col, tgt_row, tgt_col, deadline_remaining]
        # 注意：由于gym.spaces.Box要求所有元素共享同一个low/high，我们使用最大值作为上界
        # 实际运行时，各字段的合理范围是：
        #   row/col/tgt_row/tgt_col: [0, grid_height-1] 或 [0, grid_width-1]
        #   deadline_remaining: [0, task_deadline]
        max_coord = max(self.height - 1, self.width - 1)
        max_deadline = self.config['task_deadline']
        max_val = max(max_coord, max_deadline)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=0, 
                    high=max_val, 
                    shape=(self.agent_num, 5), 
                    dtype=np.float32
                ),
                "matrix": spaces.Box(
                    low=0, 
                    high=1,  # 二值矩阵
                    shape=(self.agent_num, 3, 5, 5), 
                    dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Discrete(5)
        self._action_to_direction = {
            0: np.array([1, 0]),  # 下：row +1
            1: np.array([0, 1]),  # 右：col +1
            2: np.array([-1, 0]), # 上：row -1
            3: np.array([0, -1]), # 左：col -1
            4: np.array([0, 0]),  # 停：不变
        }

        # 统计
        self.global_step = 0
        self.throughput = 0
        self.tc_throughput = 0
        self.battery_cost_steps = 0

        # agent 占位位置集合
        self.all_location: List[List[int]] = []

        # 随机种子
        self._rng = random.Random(self.config['seed'])

    def seed(self, seed: Optional[int] = None):
        self.config['seed'] = seed
        self._rng = random.Random(seed)

    def set_config(self, cfg: Dict):
        self.config.update(cfg)

    def _sample_delivery_cell(self) -> Tuple[int, int]:
        # 从货架集合中采样一个作为投递目标
        r, c = self.shelves[self._rng.randrange(len(self.shelves))]
        return int(r), int(c)

    def _nearest_charging(self, pos: List[int]) -> Tuple[int, int]:
        best = None
        best_d = 10 ** 9
        for r, c in self.charging_cells:
            d = abs(r - pos[0]) + abs(c - pos[1])
            if d < best_d:
                best_d = d
                best = (r, c)
        return best

    def _assign_new_task(self, agent: Dict):
        dcell = self._sample_delivery_cell()
        agent['current_delivery_cell'] = dcell
        # 目标为 delivery 的四邻域之一：先设为“最近邻接格”的启发式（这里直接指向 delivery，自主策略可在外部规划到邻格）
        agent['target'] = [dcell[0], dcell[1]]
        agent['task_deadline_left'] = self.config['task_deadline']
        agent['capacity'] = 0  # 表示已取件，去投递
        # 如在充电模式，清理缓存
        agent['charging_mode'] = False
        agent['saved_delivery_cell'] = None
        agent['saved_target'] = None

    def _spawn_agents(self):
        self.all_location = []
        used = set()
        for i in range(self.agent_num):
            while True:
                r, c = self.pickup_cells[self._rng.randrange(len(self.pickup_cells))]
                if (r, c) not in used:
                    used.add((r, c))
                    break
            self.all_agent['agent_' + str(i)]['agent_location'] = [r, c]
            self.all_agent['agent_' + str(i)]['battery'] = self.config['battery_init']
            self.all_agent['agent_' + str(i)]['prefer_charge'] = False
            self._assign_new_task(self.all_agent['agent_' + str(i)])
            self.all_location.append([r, c])

    def _get_local_window(self, agent_pos: List[int], size: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        返回局部窗口内每个网格的 (local_i, local_j, global_r, global_c)
        local_i, local_j: 窗口内相对坐标 (0-4), 中心在(2,2)
        global_r, global_c: 全局坐标
        """
        half = size // 2
        agent_r, agent_c = agent_pos[0], agent_pos[1]
        window_coords = []
        
        for local_i in range(size):
            for local_j in range(size):
                # 计算全局坐标（相对于agent位置的偏移）
                global_r = agent_r + (local_i - half)
                global_c = agent_c + (local_j - half)
                
                # 检查是否在有效范围内
                if 0 <= global_r < self.height and 0 <= global_c < self.width:
                    window_coords.append((local_i, local_j, global_r, global_c))
        
        return window_coords

    def _get_info(self) -> Dict:
        return {
            'throughput': self.throughput,
            'tc_throughput': self.tc_throughput,
            'battery_cost': self.battery_cost_steps if self.config.get('use_energy', True) else 0,
            'global_step': self.global_step,
        }

    def get_agent_obs(self):
        obs_list = []
        matrices = []
        
        for i in range(self.agent_num):
            ag = self.all_agent['agent_' + str(i)]
            agent_pos = ag['agent_location']
            target = ag['target']
            deadline_remaining = max(0, ag['task_deadline_left'])
            
            # 构建5维观测：[row, col, tgt_row, tgt_col, deadline_remaining]
            agent_obs = [
                float(agent_pos[0]),  # row (行)
                float(agent_pos[1]),  # col (列)
                float(target[0]),     # tgt_row (目标行)
                float(target[1]),      # tgt_col (目标列)
                float(deadline_remaining)
            ]
            obs_list.append(agent_obs)
            
            # 构建3×5×5占用矩阵
            matrix = np.zeros((3, 5, 5), dtype=np.float32)
            window_coords = self._get_local_window(agent_pos, size=5)
            
            # 获取所有其他agent的位置集合（用于通道2）
            other_agent_positions = set()
            for j in range(self.agent_num):
                if j != i:
                    other_pos = tuple(self.all_agent['agent_' + str(j)]['agent_location'])
                    other_agent_positions.add(other_pos)
            
            # 填充占用矩阵
            for local_i, local_j, global_r, global_c in window_coords:
                # 通道0：障碍物/货架
                if self.map[global_r][global_c] == CELL_OBSTACLE:
                    matrix[0, local_i, local_j] = 1.0
                
                # 通道1：充电站
                if self.map[global_r][global_c] == CELL_CHARGING:
                    matrix[1, local_i, local_j] = 1.0
                
                # 通道2：其他智能体
                if (global_r, global_c) in other_agent_positions:
                    matrix[2, local_i, local_j] = 1.0
            
            matrices.append(matrix)
        
        res = {
            'obs': np.array(obs_list, dtype=np.float32),
            'matrix': np.array(matrices, dtype=np.float32)
        }
        return res

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.global_step = 0
        self.throughput = 0
        self.tc_throughput = 0
        self.battery_cost_steps = 0
        for i in range(self.agent_num):
            ag = self.all_agent['agent_' + str(i)]
            ag['battery'] = self.config['battery_init']
        self._spawn_agents()
        observation = self.get_agent_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, actions: List[int]):
        # 同步步进与碰撞回滚
        self.global_step += 1
        rewards = [0] * self.agent_num
        move_vectors = []
        proposed_positions = []
        valid_mask = [True] * self.agent_num

        # 能量统计
        if self.config.get('use_energy', True):
            for i in range(self.agent_num):
                ag = self.all_agent['agent_' + str(i)]
                cost = self.config['move_cost'] if actions[i] != 4 else self.config['stay_cost']
                ag['battery'] -= cost
                if ag['battery'] < 0:
                    self.battery_cost_steps += 1
                # 低电量则偏好充电
                ag['prefer_charge'] = (ag['battery'] <= self.config['low_battery_threshold'])

        # 充电站瞬时充满（无论是否启用自动充电）
        if self.config.get('use_energy', True):
            for i in range(self.agent_num):
                ag = self.all_agent['agent_' + str(i)]
                # 若当前在充电格，直接回满并恢复任务
                r, c = ag['agent_location']
                if self.map[r][c] == CELL_CHARGING:
                    ag['battery'] = self.config['battery_init']
                    ag['prefer_charge'] = False
                    if ag['charging_mode']:
                        # 恢复投递目标
                        if ag['saved_delivery_cell'] is not None:
                            ag['current_delivery_cell'] = ag['saved_delivery_cell']
                        if ag['saved_target'] is not None:
                            ag['target'] = ag['saved_target'][:]
                        else:
                            # 兜底：以投递格为目标
                            dcell = ag['current_delivery_cell']
                            if dcell is not None:
                                ag['target'] = [dcell[0], dcell[1]]
                        ag['charging_mode'] = False
                        ag['saved_delivery_cell'] = None
                        ag['saved_target'] = None
        
        # 低电量：将 target 临时切到最近充电站（进入 charging_mode），充满后恢复
        # 仅在 auto_charge=True 时启用自动切换（否则由策略决定是否去充电）
        if self.config.get('use_energy', True) and self.config.get('auto_charge', True):
            for i in range(self.agent_num):
                ag = self.all_agent['agent_' + str(i)]
                # 若低电且不在充电模式，则切换到最近充电站
                if ag['prefer_charge'] and not ag['charging_mode']:
                    ag['saved_delivery_cell'] = ag['current_delivery_cell']
                    ag['saved_target'] = ag['target'][:]
                    nr, nc = self._nearest_charging(ag['agent_location'])
                    ag['target'] = [nr, nc]
                    ag['charging_mode'] = True

        # 计算动作
        for i, act in enumerate(actions):
            vec = self._action_to_direction.get(int(act), np.array([0, 0]))
            move_vectors.append(vec)
        # 预位置
        for i in range(self.agent_num):
            cur = np.array(self.all_agent['agent_' + str(i)]['agent_location'])
            nxt = cur + move_vectors[i]
            # clip 边界
            nxt[0] = int(np.clip(nxt[0], 0, self.height - 1))
            nxt[1] = int(np.clip(nxt[1], 0, self.width - 1))
            nr, nc = int(nxt[0]), int(nxt[1])
            # 障碍与越界 -> 原地
            if self.map[nr][nc] == CELL_OBSTACLE:
                proposed_positions.append(cur.tolist())
                valid_mask[i] = False
            else:
                proposed_positions.append([nr, nc])

        # 碰撞检测：同格冲突回滚
        # 规则：同格冲突时，按 agent_id 升序排序，保留最小的智能体，其他回滚（保证确定性）
        counts: Dict[Tuple[int, int], List[int]] = {}
        for i, pos in enumerate(proposed_positions):
            key = (pos[0], pos[1])
            counts.setdefault(key, []).append(i)
        
        rollback_ids = set()
        for key, ids in counts.items():
            if len(ids) > 1:
                # 按 agent_id 升序排序，保留最小的，其他回滚
                ids_sorted = sorted(ids)
                rollback_ids.update(ids_sorted[1:])  # 除第一个外，其他都回滚
        
        for i in rollback_ids:
            proposed_positions[i] = self.all_agent['agent_' + str(i)]['agent_location'][:]

        # 提交位置
        for i in range(self.agent_num):
            self.all_agent['agent_' + str(i)]['agent_location'] = proposed_positions[i][:]

        # 任务完成检测：达到 delivery cell 的四邻域
        for i in range(self.agent_num):
            ag = self.all_agent['agent_' + str(i)]
            dcell = ag['current_delivery_cell']
            # 在充电模式时不进行交付判定
            if (not ag.get('charging_mode', False)) and dcell is not None and is_adjacent_to_delivery(ag['agent_location'], dcell):
                rewards[i] += 1
                self.throughput += 1
                # tc_throughput: 判定完成时的剩余时间是否 >= 0（不是派单时刻）
                if ag['task_deadline_left'] >= 0:
                    self.tc_throughput += 1
                # 完成后立刻分配新任务
                self._assign_new_task(ag)

        # 任务 deadline 递减
        for i in range(self.agent_num):
            self.all_agent['agent_' + str(i)]['task_deadline_left'] -= 1

        # episode 控制
        terminated = False
        truncated = (self.global_step >= self.config['episode_len'])

        observation = self.get_agent_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, rewards, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.rol_size, self.row_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # 生成静态图层（只生成一次）
        if not hasattr(self, "_static_canvas") or self._static_canvas is None:
            static_canvas = pygame.Surface((self.rol_size, self.row_size))
            static_canvas.fill((255, 255, 255))
            pix_square_size = (self.rol_size / self.width)

            COLOR_PICKUP = (0, 200, 0)
            COLOR_OBSTACLE = (0, 0, 0)
            COLOR_CHARGING = (255, 215, 0)

            # 绘制静态格（货架/取件/充电）
            for r, c in self.obstacles:
                rect = pygame.Rect(
                    pix_square_size * np.array([c, r]),
                    (pix_square_size, pix_square_size),
                )
                pygame.draw.rect(static_canvas, COLOR_OBSTACLE, rect)
            for r, c in self.pickup_cells:
                rect = pygame.Rect(
                    pix_square_size * np.array([c, r]),
                    (pix_square_size, pix_square_size),
                )
                pygame.draw.rect(static_canvas, COLOR_PICKUP, rect)
            for r, c in self.charging_cells:
                rect = pygame.Rect(
                    pix_square_size * np.array([c, r]),
                    (pix_square_size, pix_square_size),
                )
                pygame.draw.rect(static_canvas, COLOR_CHARGING, rect)

            # 网格线
            for x in range(0, self.rol_size, self.cell_size):
                pygame.draw.line(static_canvas, 0, (x, 0), (x, self.row_size))
            for y in range(0, self.row_size, self.cell_size):
                pygame.draw.line(static_canvas, 0, (0, y), (self.rol_size, y))

            self._static_canvas = static_canvas

        # 每帧复制静态层
        canvas = self._static_canvas.copy()
        pix_square_size = (self.rol_size / self.width)
        # 颜色
        COLOR_PICKUP = (0, 200, 0)
        COLOR_OBSTACLE = (0, 0, 0)
        COLOR_CHARGING = (255, 215, 0)
        COLOR_AGENT = (0, 0, 255)
        WHITE = (255, 255, 255)

        # 绘制 agent 与 ID/剩余时限条
        for i in range(self.agent_num):
            ag = self.all_agent['agent_' + str(i)]
            center = (np.array(ag['agent_location'][::-1]) + 0.5) * pix_square_size
            pygame.draw.circle(canvas, COLOR_AGENT, center, pix_square_size / 3)
            # 绘制ID
            font_task = pygame.font.Font(None, 14)
            text_task = font_task.render(str(i), True, WHITE)
            text_rect_task = text_task.get_rect(center=center)
            canvas.blit(text_task, text_rect_task)
            # 绘制deadline条（可选）
            dl = max(0, min(self.config['task_deadline'], ag['task_deadline_left']))
            frac = 0 if self.config['task_deadline'] == 0 else dl / self.config['task_deadline']
            bar_w = pix_square_size * 0.6 * frac
            bar_h = pix_square_size * 0.08
            bar_left = center[0] - pix_square_size * 0.3
            bar_top = center[1] - pix_square_size * 0.5
            bar_rect = pygame.Rect(bar_left, bar_top, bar_w, bar_h)
            pygame.draw.rect(canvas, (0, 150, 0), bar_rect)
            # 绘制电量文本
            batt = ag['battery']
            font_batt = pygame.font.Font(None, 14)
            text_batt = font_batt.render(f"{batt}", True, (0, 0, 0))
            text_rect_batt = text_batt.get_rect(center=(center[0], center[1] + pix_square_size * 0.45))
            canvas.blit(text_batt, text_rect_batt)

        # Legend（简要）
        legend_items = [
            ("PICKUP", COLOR_PICKUP),
            ("OBSTACLE", COLOR_OBSTACLE),
            ("CHARGING", COLOR_CHARGING),
            ("AGENT", COLOR_AGENT),
        ]
        font = pygame.font.Font(None, 18)
        offx, offy = 5, 5
        for name, col in legend_items:
            rect = pygame.Rect(offx, offy, 12, 12)
            pygame.draw.rect(canvas, col, rect)
            txt = font.render(name, True, (0, 0, 0))
            canvas.blit(txt, (offx + 16, offy - 2))
            offy += 16
        # 电量说明
        txt2 = font.render("Battery shown under agent", True, (0, 0, 0))
        canvas.blit(txt2, (5, offy + 2))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"] if "render_fps" in self.metadata else self.config['fps'])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


