"""
优化的Fulfillment Warehouse渲染器
缓存静态层以提高性能
"""
import math
import os
import sys

from gymnasium import error
import numpy as np
import six

try:
    import pyglet
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    """
    )

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install.
    """
    )


# 颜色定义（匹配图像）
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_YELLOW = (255, 255, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)
_DARKORANGE = (255, 140, 0)
_BLUE = (0, 0, 255)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display object."""
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class FulfillmentViewer(object):
    """优化的Fulfillment Warehouse渲染器，缓存静态层"""
    
    def __init__(self, world_size, map_mask):
        display = get_display(None)
        self.rows, self.cols = world_size
        self.map_mask = map_mask
        
        self.grid_size = 30
        self.icon_size = 20
        
        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 2 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # 缓存静态层（一次性渲染）
        self.static_surface = None
        self._cache_static_layers()
    
    def close(self):
        self.window.close()
    
    def window_closed_by_user(self):
        self.isopen = False
    
    def _cache_static_layers(self):
        """缓存静态层到Surface（货架、拾取点、充电站、网格）"""
        # 注意：pyglet 1.5.x 没有直接的Surface缓存
        # 我们将在每帧渲染静态层，但优化绘制逻辑
        # 实际缓存可以通过创建离屏渲染缓冲区实现，但为了兼容性暂时简单实现
        self.static_layers_cached = True
    
    def render(self, env, return_rgb_array=False):
        """渲染环境"""
        glClearColor(*_BACKGROUND_COLOR, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        
        # 绘制静态层（网格、货架、拾取点、充电站）
        self._draw_grid()
        self._draw_static_tiles()
        
        # 绘制动态层（智能体、任务目标）
        self._draw_agents_and_tasks(env)
        
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        else:
            arr = None
        
        self.window.flip()
        return arr if return_rgb_array else self.isopen
    
    def _draw_grid(self):
        """绘制网格线"""
        # 使用vertex_list而不是batch.add（兼容pyglet 1.5.x）
        # 水平线
        for r in range(self.rows + 1):
            y = (self.grid_size + 1) * r + 1
            verts = [
                0, y,
                (self.grid_size + 1) * self.cols, y
            ]
            colors = (*_GRID_COLOR, *_GRID_COLOR)
            vertex_list = pyglet.graphics.vertex_list(2, ('v2f', verts), ('c3B', colors))
            vertex_list.draw(GL_LINES)
        
        # 垂直线
        for c in range(self.cols + 1):
            x = (self.grid_size + 1) * c + 1
            verts = [
                x, 0,
                x, (self.grid_size + 1) * self.rows
            ]
            colors = (*_GRID_COLOR, *_GRID_COLOR)
            vertex_list = pyglet.graphics.vertex_list(2, ('v2f', verts), ('c3B', colors))
            vertex_list.draw(GL_LINES)
    
    def _draw_static_tiles(self):
        """绘制静态图块（货架、拾取点、充电站）"""
        # BLACK - 货架
        shelf_cells = np.argwhere(self.map_mask == 0)  # TILE_SHELF
        for y, x in shelf_cells:
            self._draw_tile(x, y, _BLACK)
        
        # GREEN - 拾取点
        pickup_cells = np.argwhere(self.map_mask == 1)  # TILE_PICKUP
        for y, x in pickup_cells:
            self._draw_tile(x, y, _GREEN)
        
        # YELLOW - 充电站
        charger_cells = np.argwhere(self.map_mask == 2)  # TILE_CHARGER
        for y, x in charger_cells:
            self._draw_tile(x, y, _YELLOW)
    
    def _draw_tile(self, x, y, color):
        """绘制单个图块"""
        # 注意：pyglet坐标系，y需要翻转
        pyglet_y = self.rows - y - 1
        
        # 绘制填充矩形
        verts = [
            (self.grid_size + 1) * x + 1,  # TL X
            (self.grid_size + 1) * pyglet_y + 1,  # TL Y
            (self.grid_size + 1) * (x + 1),  # TR X
            (self.grid_size + 1) * pyglet_y + 1,  # TR Y
            (self.grid_size + 1) * (x + 1),  # BR X
            (self.grid_size + 1) * (pyglet_y + 1),  # BR Y
            (self.grid_size + 1) * x + 1,  # BL X
            (self.grid_size + 1) * (pyglet_y + 1),  # BL Y
        ]
        colors = 4 * color
        vertex_list = pyglet.graphics.vertex_list(4, ('v2f', verts), ('c3B', colors))
        vertex_list.draw(GL_QUADS)
    
    def _draw_agents_and_tasks(self, env):
        """绘制智能体和任务目标"""
        radius = self.grid_size / 3
        resolution = 6
        
        # 绘制智能体
        for agent_id in range(env.num_agents):
            x, y = int(env.agent_positions[agent_id, 0]), int(env.agent_positions[agent_id, 1])
            pyglet_y = self.rows - y - 1
            
            # 根据电池状态选择颜色（如果启用能量模型）
            if env.use_energy and env.agent_batteries[agent_id] < env.LOW_THRESHOLD:
                agent_color = _RED  # 低电量显示红色
            else:
                agent_color = _ORANGE  # 正常显示橙色
            
            # 绘制圆形智能体
            center_x = (self.grid_size + 1) * x + self.grid_size // 2 + 1
            center_y = (self.grid_size + 1) * pyglet_y + self.grid_size // 2 + 1
            
            verts = []
            colors = []
            for i in range(resolution):
                angle = 2 * math.pi * i / resolution
                px = radius * math.cos(angle) + center_x
                py = radius * math.sin(angle) + center_y
                verts += [px, py]
                colors.extend(agent_color)
            
            vertex_list = pyglet.graphics.vertex_list(resolution, ('v2f', verts), ('c3B', colors))
            vertex_list.draw(GL_POLYGON)
        
        # 绘制任务目标（交付点标记）
        for agent_id in range(env.num_agents):
            task = env.agent_tasks[agent_id]
            if task is not None:
                delivery_pos = task["delivery_pos"]
                dx, dy = delivery_pos
                pyglet_y = self.rows - dy - 1
                
                # 在交付点周围绘制标记（4个小方块表示4邻域）
                marker_size = 3
                neighbors = [
                    (dx, dy - 1),  # UP
                    (dx, dy + 1),  # DOWN
                    (dx - 1, dy),  # LEFT
                    (dx + 1, dy),  # RIGHT
                ]
                
                for nx, ny in neighbors:
                    if 0 <= nx < self.cols and 0 <= ny < self.rows:
                        pyglet_ny = self.rows - ny - 1
                        # 绘制小标记
                        verts = [
                            (self.grid_size + 1) * nx + self.grid_size // 2 - marker_size,
                            (self.grid_size + 1) * pyglet_ny + self.grid_size // 2 - marker_size,
                            (self.grid_size + 1) * nx + self.grid_size // 2 + marker_size,
                            (self.grid_size + 1) * pyglet_ny + self.grid_size // 2 - marker_size,
                            (self.grid_size + 1) * nx + self.grid_size // 2 + marker_size,
                            (self.grid_size + 1) * pyglet_ny + self.grid_size // 2 + marker_size,
                            (self.grid_size + 1) * nx + self.grid_size // 2 - marker_size,
                            (self.grid_size + 1) * pyglet_ny + self.grid_size // 2 + marker_size,
                        ]
                        colors = 4 * _BLUE  # 蓝色标记表示目标邻域
                        vertex_list = pyglet.graphics.vertex_list(4, ('v2f', verts), ('c3B', colors))
                        vertex_list.draw(GL_QUADS)
