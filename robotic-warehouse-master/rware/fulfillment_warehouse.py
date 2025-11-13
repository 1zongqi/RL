"""
Fulfillment Warehouse Environment
基于46×33网格的仓库环境，支持任务分配、能量模型和DICT格式观测
"""
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding


class FulfillmentWarehouseEnv(gym.Env):
    """Fulfillment Warehouse Environment
    
    基于46×33网格构建的仓库环境，智能体在GREEN拾取点生成，
    需要到达BLACK货架的上下邻域完成任务。
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    # 地图掩码值
    TILE_SHELF = 0      # BLACK - 货架，不可通行
    TILE_PICKUP = 1     # GREEN - 拾取点，可通行，智能体生成点
    TILE_CHARGER = 2    # YELLOW - 充电站，可通行
    TILE_FREE = 3       # WHITE - 自由空间，可通行
    
    # 动作
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4
    
    # 能量参数
    START_BATTERY = 220
    MOVE_COST = 1
    STAY_COST = 0
    LOW_THRESHOLD = 30
    TASK_DEADLINE = 40
    MAX_EPISODE_STEPS = 400

    def __init__(
        self,
        num_agents: int = 32,
        use_energy: bool = False,
        render_mode: Optional[str] = None,
        obs_mode: str = "dict",
    ):
        """初始化环境
        
        Args:
            num_agents: 智能体数量
            use_energy: 是否启用能量模型
            render_mode: 渲染模式 ("human" 或 "rgb_array")
            obs_mode: 观测模式 ("dict" 默认)
        """
        self.num_agents = num_agents
        self.use_energy = use_energy
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        
        # 生成地图掩码
        self._generate_mask_from_description()
        
        # 验证地图
        self._validate_map()
        
        # 动作空间：每个智能体5个动作（上下左右停留）
        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(5)] * num_agents)
        )
        
        # 观测空间：DICT格式
        if obs_mode == "dict":
            self.observation_space = gym.spaces.Tuple(
                tuple([
                    gym.spaces.Dict({
                        "agent_pos": gym.spaces.Box(
                            low=np.array([0, 0], dtype=np.int32),
                            high=np.array([45, 32], dtype=np.int32),
                            shape=(2,),
                            dtype=np.int32
                        ),
                        "goal_shelf": gym.spaces.Box(
                            low=np.array([0, 0], dtype=np.int32),
                            high=np.array([45, 32], dtype=np.int32),
                            shape=(2,),
                            dtype=np.int32
                        ),
                        "nearest_charger": gym.spaces.Box(
                            low=np.array([0, 0], dtype=np.int32),
                            high=np.array([45, 32], dtype=np.int32),
                            shape=(2,),
                            dtype=np.int32
                        ),
                        "battery": gym.spaces.Box(
                            low=np.array([-1000], dtype=np.int32),
                            high=np.array([1000], dtype=np.int32),
                            shape=(1,),
                            dtype=np.int32
                        ),
                        "deadline_left": gym.spaces.Box(
                            low=np.array([0], dtype=np.int32),
                            high=np.array([40], dtype=np.int32),
                            shape=(1,),
                            dtype=np.int32
                        ),
                        "local_3x3": gym.spaces.Box(
                            low=0,
                            high=3,
                            shape=(3, 3),
                            dtype=np.int8
                        ),
                    })
                ] * num_agents)
            )
        else:
            raise ValueError(f"Unknown obs_mode: {obs_mode}")
        
        # 智能体状态
        self.agents: List[Dict] = []
        self.agent_positions: np.ndarray = None  # (num_agents, 2) - [x, y]
        self.agent_tasks: List[Optional[Dict]] = []  # 每个智能体的任务信息
        self.agent_batteries: np.ndarray = None  # (num_agents,)
        
        # 指标追踪
        self.throughput = 0
        self.tc_throughput = 0  # time-constrained throughput
        self.battery_cost = 0
        self.battery_violation_steps = 0  # 电池<0的步数计数
        self.collisions = 0
        self._step_count = 0
        
        # 渲染器
        self.renderer = None
        
        # 初始化随机数生成器
        self._np_random = None
        
    def _generate_mask_from_description(self):
        """根据精确规格生成46×33网格地图掩码"""
        # 网格尺寸：46列 × 33行 (width=46, height=33)
        width, height = 46, 33
        self.grid_size = (height, width)
        
        # 初始化掩码为自由空间
        self.map_mask = np.full((height, width), self.TILE_FREE, dtype=np.int32)
        
        self.delivery_cells = []  # 货架/交付目标
        self.pickup_cells = []    # 拾取点
        self.charging_cells = []  # 充电站
        
        # 1. PICKUPS (GREEN): 32组，每组2×3（6个单元格），总计192个单元格
        # 左侧16组：边界框 x∈[1..5], y∈[1..31]
        # 右侧16组：中心对称镜像左侧
        
        # 计算左侧16组的排列
        # 假设：16组排列为2列×8行（每组间1单元格间距）
        # 每组2×3，组间水平间距1，组间垂直间距1
        
        PICKUP_GROUP_WIDTH = 2   # 每组宽2
        PICKUP_GROUP_HEIGHT = 3  # 每组高3
        PICKUP_GROUPS_LEFT = 16
        PICKUP_SPACING = 1  # 组间距1
        
        # 左侧布局：计算16组的位置
        # 2列：x列1从x=1开始，x列2从x=1+2+1=4开始
        # 8行：y从1开始，每组3高+1间距，所以y位置: 1, 5, 9, 13, 17, 21, 25, 29
        
        left_pickup_groups = []
        for row in range(8):  # 8行
            y_start = 1 + row * (PICKUP_GROUP_HEIGHT + PICKUP_SPACING)
            for col in range(2):  # 2列
                x_start = 1 + col * (PICKUP_GROUP_WIDTH + PICKUP_SPACING)
                # 检查是否在边界内
                if (x_start + PICKUP_GROUP_WIDTH - 1 <= 5 and 
                    y_start + PICKUP_GROUP_HEIGHT - 1 <= 31):
                    left_pickup_groups.append((x_start, y_start))
        
        # 左侧16组拾取点
        for x_start, y_start in left_pickup_groups[:16]:
            for dx in range(PICKUP_GROUP_WIDTH):
                for dy in range(PICKUP_GROUP_HEIGHT):
                    x = x_start + dx
                    y = y_start + dy
                    if x <= 5 and y <= 31:
                        self.map_mask[y, x] = self.TILE_PICKUP
                        self.pickup_cells.append((x, y))
        
        # 右侧16组：中心对称镜像左侧
        # 地图中心x = (width-1)/2 = 22.5
        # 镜像公式：x_right = width - 1 - x_left
        
        right_pickup_groups = []
        for x_start, y_start in left_pickup_groups[:16]:
            # 镜像x坐标
            x_start_mirror = width - 1 - (x_start + PICKUP_GROUP_WIDTH - 1)
            right_pickup_groups.append((x_start_mirror, y_start))
        
        # 右侧16组拾取点
        for x_start, y_start in right_pickup_groups:
            for dx in range(PICKUP_GROUP_WIDTH):
                for dy in range(PICKUP_GROUP_HEIGHT):
                    x = x_start + dx
                    y = y_start + dy
                    if x < width and y <= 31:
                        if self.map_mask[y, x] == self.TILE_FREE:
                            self.map_mask[y, x] = self.TILE_PICKUP
                            self.pickup_cells.append((x, y))
        
        # 2. SHELVES (BLACK): 24组，每组10×1（10个单元格），总计240个单元格
        # 3列 × 8行布局
        # 列间距：1单元格，行间距：3单元格
        # 边界框：x∈[7..38], y∈[2..30]（含端点）
        
        SHELF_GROUP_WIDTH = 10
        SHELF_GROUP_HEIGHT = 1
        SHELF_COLS = 3
        SHELF_ROWS = 8
        SHELF_COL_SPACING = 1
        SHELF_ROW_SPACING = 3
        
        # 计算列起始x位置
        # 第1列：x=7
        # 第2列：x=7+10+1=18
        # 第3列：x=18+10+1=29
        shelf_col_x_starts = [7, 18, 29]
        
        # 计算行起始y位置
        # 第1行：y=2
        # 第2行：y=2+1+3=6
        # 第3行：y=6+1+3=10
        # ... 依此类推
        shelf_row_y_starts = []
        for row in range(SHELF_ROWS):
            y_start = 2 + row * (SHELF_GROUP_HEIGHT + SHELF_ROW_SPACING)
            if y_start <= 30:
                shelf_row_y_starts.append(y_start)
        
        # 生成24组货架
        for col_idx, x_start in enumerate(shelf_col_x_starts):
            for row_idx, y_start in enumerate(shelf_row_y_starts):
                for dx in range(SHELF_GROUP_WIDTH):
                    x = x_start + dx
                    y = y_start
                    if x <= 38 and y <= 30:
                        self.map_mask[y, x] = self.TILE_SHELF
                        self.delivery_cells.append((x, y))
        
        # 3. CHARGERS (YELLOW): 8个单单元格
        charger_coords = [
            # TOP
            (6, 0), (17, 0), (28, 0), (39, 0),
            # BOTTOM
            (6, 32), (17, 32), (28, 32), (39, 32),
        ]
        
        for x, y in charger_coords:
            if self.map_mask[y, x] == self.TILE_FREE:
                self.map_mask[y, x] = self.TILE_CHARGER
                self.charging_cells.append((x, y))
            elif self.map_mask[y, x] == self.TILE_PICKUP:
                # 如果充电站位置有拾取点，移除拾取点，保留充电站
                self.pickup_cells = [cell for cell in self.pickup_cells if cell != (x, y)]
                self.map_mask[y, x] = self.TILE_CHARGER
                self.charging_cells.append((x, y))
        
    def _validate_map(self):
        """验证地图有效性"""
        # 精确计数验证
        assert len(self.pickup_cells) == 192, \
            f"拾取点数量必须为192，实际为{len(self.pickup_cells)}"
        assert len(self.delivery_cells) == 240, \
            f"货架数量必须为240，实际为{len(self.delivery_cells)}"
        assert len(self.charging_cells) == 8, \
            f"充电站数量必须为8，实际为{len(self.charging_cells)}"
        
        # 边界检查：所有单元格在 [0..45]×[0..32]
        all_cells = self.pickup_cells + self.delivery_cells + self.charging_cells
        for x, y in all_cells:
            assert 0 <= x <= 45, f"x坐标 {x} 超出范围 [0..45]"
            assert 0 <= y <= 32, f"y坐标 {y} 超出范围 [0..32]"
        
        # 集合互不相交
        pickup_set = set(self.pickup_cells)
        delivery_set = set(self.delivery_cells)
        charger_set = set(self.charging_cells)
        
        pickup_delivery_overlap = pickup_set & delivery_set
        assert len(pickup_delivery_overlap) == 0, \
            f"拾取点和货架不能重叠，发现重叠: {pickup_delivery_overlap}"
        
        pickup_charger_overlap = pickup_set & charger_set
        assert len(pickup_charger_overlap) == 0, \
            f"拾取点和充电站不能重叠，发现重叠: {pickup_charger_overlap}"
        
        delivery_charger_overlap = delivery_set & charger_set
        assert len(delivery_charger_overlap) == 0, \
            f"货架和充电站不能重叠，发现重叠: {delivery_charger_overlap}"
        
        # 验证网格大小
        assert self.grid_size == (33, 46), \
            f"网格大小应为(33, 46)，实际为{self.grid_size}"
    
    def _is_traversable(self, x: int, y: int) -> bool:
        """检查位置是否可通行"""
        if x < 0 or x >= self.grid_size[1] or y < 0 or y >= self.grid_size[0]:
            return False
        return self.map_mask[y, x] != self.TILE_SHELF
    
    def _is_adjacent_vertical_to_delivery(self, pos: Tuple[int, int], delivery_pos: Tuple[int, int]) -> bool:
        """检查位置是否在交付点的上方或下方（仅垂直方向，dx==0, dy==±1）"""
        dx = abs(pos[0] - delivery_pos[0])
        dy = pos[1] - delivery_pos[1]
        return dx == 0 and abs(dy) == 1  # 仅上方或下方
    
    def _is_adjacent_to_delivery(self, pos: Tuple[int, int], delivery_pos: Tuple[int, int]) -> bool:
        """检查位置是否与交付点4邻域相邻（保持向后兼容，实际调用垂直版本）"""
        return self._is_adjacent_vertical_to_delivery(pos, delivery_pos)
    
    def _nearest_charger(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """找到最近的充电站（曼哈顿距离）"""
        if len(self.charging_cells) == 0:
            return (0, 0)
        min_dist = float('inf')
        nearest = self.charging_cells[0]
        for charger_pos in self.charging_cells:
            dist = abs(pos[0] - charger_pos[0]) + abs(pos[1] - charger_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest = charger_pos
        return nearest
    
    def _assign_new_task(self, agent_id: int):
        """为智能体分配新任务"""
        # 随机选择拾取点和交付点
        pickup_idx = self._np_random.integers(0, len(self.pickup_cells))
        delivery_idx = self._np_random.integers(0, len(self.delivery_cells))
        
        pickup_pos = self.pickup_cells[pickup_idx]
        delivery_pos = self.delivery_cells[delivery_idx]
        
        self.agent_tasks[agent_id] = {
            "pickup_pos": pickup_pos,
            "delivery_pos": delivery_pos,
            "task_start_step": self._step_count,
            "deadline": self._step_count + self.TASK_DEADLINE,
        }
        
    def _check_task_completion(self, agent_id: int) -> bool:
        """检查智能体是否完成任务（必须在交付点的上方或下方）"""
        agent_pos = (int(self.agent_positions[agent_id, 0]), 
                    int(self.agent_positions[agent_id, 1]))
        task = self.agent_tasks[agent_id]
        
        if task is None:
            return False
        
        delivery_pos = task["delivery_pos"]
        return self._is_adjacent_vertical_to_delivery(agent_pos, delivery_pos)
    
    def _update_battery(self, agent_id: int, action: int):
        """更新智能体电池状态"""
        if not self.use_energy:
            return
        
        if action == self.ACTION_STAY:
            self.agent_batteries[agent_id] -= self.STAY_COST
        else:
            self.agent_batteries[agent_id] -= self.MOVE_COST
        
        # 如果智能体在充电站位置，充电到满
        agent_pos = (int(self.agent_positions[agent_id, 0]),
                    int(self.agent_positions[agent_id, 1]))
        if agent_pos in self.charging_cells:
            self.agent_batteries[agent_id] = self.START_BATTERY
        
        # 追踪电池违规（<0）
        if self.agent_batteries[agent_id] < 0:
            self.battery_violation_steps += 1
            self.battery_cost += 1
    
    def _get_action_target(self, agent_id: int, action: int) -> Tuple[int, int]:
        """获取动作的目标位置"""
        current_pos = (int(self.agent_positions[agent_id, 0]),
                      int(self.agent_positions[agent_id, 1]))
        x, y = current_pos
        
        if action == self.ACTION_UP:
            y = max(0, y - 1)
        elif action == self.ACTION_DOWN:
            y = min(self.grid_size[0] - 1, y + 1)
        elif action == self.ACTION_LEFT:
            x = max(0, x - 1)
        elif action == self.ACTION_RIGHT:
            x = min(self.grid_size[1] - 1, x + 1)
        # ACTION_STAY: x, y 不变
        
        return (x, y)
    
    def _validate_action(self, agent_id: int, action: int) -> int:
        """验证动作，无效动作转为STAY"""
        target_pos = self._get_action_target(agent_id, action)
        
        if not self._is_traversable(target_pos[0], target_pos[1]):
            return self.ACTION_STAY
        
        return action
    
    def _get_local_3x3(self, agent_pos: Tuple[int, int]) -> np.ndarray:
        """获取智能体周围的3×3局部视野
        
        编码：0=free, 1=pickup, 2=shelf, 3=charger
        注意：map_mask的编码不同，需要转换
        map_mask: 0=shelf, 1=pickup, 2=charger, 3=free
        local_3x3: 0=free, 1=pickup, 2=shelf, 3=charger
        """
        x, y = agent_pos
        local_map = np.zeros((3, 3), dtype=np.int8)
        
        # 编码映射：map_mask -> local_3x3
        # map_mask: 0=shelf -> local_3x3: 2=shelf
        # map_mask: 1=pickup -> local_3x3: 1=pickup
        # map_mask: 2=charger -> local_3x3: 3=charger
        # map_mask: 3=free -> local_3x3: 0=free
        encoding_map = {
            0: 2,  # shelf -> shelf
            1: 1,  # pickup -> pickup
            2: 3,  # charger -> charger
            3: 0,  # free -> free
        }
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[1] and 0 <= ny < self.grid_size[0]:
                    tile_type = self.map_mask[ny, nx]
                    local_map[dy + 1, dx + 1] = encoding_map[tile_type]
                else:
                    # 边界外视为shelf（不可通行）-> 编码为2
                    local_map[dy + 1, dx + 1] = 2
        
        return local_map
    
    def _make_observation(self, agent_id: int) -> Dict:
        """生成智能体的观测（DICT格式）"""
        agent_pos = (int(self.agent_positions[agent_id, 0]),
                    int(self.agent_positions[agent_id, 1]))
        task = self.agent_tasks[agent_id]
        
        # agent_pos: [x, y]
        agent_pos_array = np.array([agent_pos[0], agent_pos[1]], dtype=np.int32)
        
        # goal_shelf: 交付目标位置
        if task is not None:
            goal_shelf_array = np.array([task["delivery_pos"][0], task["delivery_pos"][1]], dtype=np.int32)
            deadline_left = np.array([max(0, task["deadline"] - self._step_count)], dtype=np.int32)
        else:
            goal_shelf_array = np.array([-1, -1], dtype=np.int32)
            deadline_left = np.array([0], dtype=np.int32)
        
        # nearest_charger: 最近充电站位置
        nearest_charger_pos = self._nearest_charger(agent_pos)
        nearest_charger_array = np.array([nearest_charger_pos[0], nearest_charger_pos[1]], dtype=np.int32)
        
        # battery: 电池状态
        if self.use_energy:
            battery_array = np.array([int(self.agent_batteries[agent_id])], dtype=np.int32)
        else:
            battery_array = np.array([-1], dtype=np.int32)
        
        # local_3x3: 局部3×3视野
        local_3x3 = self._get_local_3x3(agent_pos)
        
        return {
            "agent_pos": agent_pos_array,
            "goal_shelf": goal_shelf_array,
            "nearest_charger": nearest_charger_array,
            "battery": battery_array,
            "deadline_left": deadline_left,
            "local_3x3": local_3x3,
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """重置环境"""
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        elif self._np_random is None:
            # 如果随机数生成器未初始化，初始化它
            self._np_random, _ = seeding.np_random()
        
        super().reset(seed=seed)
        
        # 重置指标
        self.throughput = 0
        self.tc_throughput = 0
        self.battery_cost = 0
        self.battery_violation_steps = 0
        self.collisions = 0
        self._step_count = 0
        
        # 初始化智能体位置（在随机GREEN拾取点生成）
        self.agent_positions = np.zeros((self.num_agents, 2), dtype=np.int32)
        self.agent_tasks = [None] * self.num_agents
        self.agent_batteries = np.full(self.num_agents, self.START_BATTERY, dtype=np.float32)
        
        # 随机分配拾取点（不重复）
        pickup_indices = self._np_random.choice(
            len(self.pickup_cells),
            size=min(self.num_agents, len(self.pickup_cells)),
            replace=False
        )
        
        for agent_id in range(self.num_agents):
            if agent_id < len(pickup_indices):
                pickup_pos = self.pickup_cells[pickup_indices[agent_id]]
            else:
                # 如果智能体数量超过拾取点，随机分配
                pickup_pos = self.pickup_cells[self._np_random.integers(0, len(self.pickup_cells))]
            
            self.agent_positions[agent_id, 0] = pickup_pos[0]
            self.agent_positions[agent_id, 1] = pickup_pos[1]
            
            # 分配初始任务
            self._assign_new_task(agent_id)
        
        # 生成观测
        observations = tuple([self._make_observation(i) for i in range(self.num_agents)])
        info = self._get_info()
        
        return observations, info
    
    def step(self, actions: List[int]):
        """执行一步"""
        assert len(actions) == self.num_agents, f"动作数量({len(actions)})与智能体数量({self.num_agents})不匹配"
        
        # 1. 验证动作，处理能量模型路由
        validated_actions = []
        target_positions = {}
        
        for agent_id, action in enumerate(actions):
            # 如果启用能量模型且电量低，路由到充电站
            if self.use_energy and self.agent_batteries[agent_id] < self.LOW_THRESHOLD:
                agent_pos = (int(self.agent_positions[agent_id, 0]),
                           int(self.agent_positions[agent_id, 1]))
                charger_pos = self._nearest_charger(agent_pos)
                
                # 简单导航：移动到充电站方向
                dx = charger_pos[0] - agent_pos[0]
                dy = charger_pos[1] - agent_pos[1]
                
                if abs(dx) > abs(dy):
                    action = self.ACTION_RIGHT if dx > 0 else self.ACTION_LEFT
                else:
                    action = self.ACTION_DOWN if dy > 0 else self.ACTION_UP
                
                # 如果已经在充电站，停留
                if agent_pos == charger_pos:
                    action = self.ACTION_STAY
            
            # 验证动作
            validated_action = self._validate_action(agent_id, action)
            validated_actions.append(validated_action)
            
            # 计算目标位置
            target_pos = self._get_action_target(agent_id, validated_action)
            target_positions[agent_id] = target_pos
        
        # 2. 处理碰撞：检测多智能体进入同一单元格
        position_to_agents = {}
        for agent_id, target_pos in target_positions.items():
            if target_pos not in position_to_agents:
                position_to_agents[target_pos] = []
            position_to_agents[target_pos].append(agent_id)
        
        # 回滚冲突的智能体
        for target_pos, agent_list in position_to_agents.items():
            if len(agent_list) > 1:
                # 多个智能体想进入同一位置，都回滚为STAY
                for agent_id in agent_list:
                    validated_actions[agent_id] = self.ACTION_STAY
                    target_positions[agent_id] = (
                        int(self.agent_positions[agent_id, 0]),
                        int(self.agent_positions[agent_id, 1])
                    )
                self.collisions += len(agent_list)
        
        # 3. 执行动作
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        for agent_id, action in enumerate(validated_actions):
            target_pos = target_positions[agent_id]
            self.agent_positions[agent_id, 0] = target_pos[0]
            self.agent_positions[agent_id, 1] = target_pos[1]
            
            # 更新电池
            self._update_battery(agent_id, action)
            
            # 检查任务完成（使用垂直邻域检查）
            if self._check_task_completion(agent_id):
                task = self.agent_tasks[agent_id]
                self.throughput += 1
                
                # 检查是否在deadline内完成
                if self._step_count <= task["deadline"]:
                    self.tc_throughput += 1
                    rewards[agent_id] = 1.0
                else:
                    rewards[agent_id] = 0.5  # 超时完成，奖励较少
                
                # 立即分配新任务
                self._assign_new_task(agent_id)
            else:
                # 小负奖励鼓励完成任务
                rewards[agent_id] = -0.01
        
        self._step_count += 1
        
        # 4. 生成观测
        observations = tuple([self._make_observation(i) for i in range(self.num_agents)])
        
        # 5. 检查episode是否结束
        terminated = False
        truncated = self._step_count >= self.MAX_EPISODE_STEPS
        
        info = self._get_info()
        
        return observations, list(rewards), terminated, truncated, info
    
    def _get_info(self) -> Dict:
        """获取信息字典"""
        info = {
            "throughput": self.throughput,
            "tc_throughput": self.tc_throughput,
            "collisions": self.collisions,
        }
        if self.use_energy:
            info["battery_cost"] = self.battery_cost
            info["battery_violation_steps"] = self.battery_violation_steps
        return info
    
    def render(self):
        """渲染环境"""
        if self.renderer is None:
            from rware.fulfillment_renderer import FulfillmentViewer
            self.renderer = FulfillmentViewer(self.grid_size, self.map_mask)
        
        return self.renderer.render(
            self,
            return_rgb_array=self.render_mode == "rgb_array"
        )
    
    def close(self):
        """关闭环境"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
    
    def seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            return [seed]
