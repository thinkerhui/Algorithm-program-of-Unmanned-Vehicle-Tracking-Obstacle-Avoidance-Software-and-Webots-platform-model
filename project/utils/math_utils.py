# 数学工具函数
# 提供常用的数学计算功能

import math


def normalize_angle(angle):
    """将角度归一化到 [-π, π] 范围"""
    while angle < -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle


def distance(x1, y1, x2, y2):
    """计算两点间的欧几里得距离"""
    return math.hypot(x2 - x1, y2 - y1)


def heading(x1, y1, x2, y2):
    """计算从点1到点2的朝向角"""
    return math.atan2(y2 - y1, x2 - x1)


def angle_difference(angle1, angle2):
    """计算两个角度的差值"""
    diff = angle1 - angle2
    return normalize_angle(diff)


def world_to_grid(x, y, scale=10):
    """将世界坐标转换为网格坐标"""
    return int(x * scale), int(y * scale)


def grid_to_world(grid_x, grid_y, scale=10):
    """将网格坐标转换为世界坐标"""
    return grid_x / scale, grid_y / scale


def interpolate(start, end, t):
    """线性插值"""
    return start + t * (end - start)


def generate_arc_points(center_x, center_y, radius, start_angle, end_angle, num_points=10):
    """生成圆弧上的点
    Args:
        center_x, center_y: 圆心坐标
        radius: 半径
        start_angle: 起始角度（弧度）
        end_angle: 结束角度（弧度）
        num_points: 生成的点数
    Returns:
        点列表 [(x, y), ...]
    """
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = start_angle + t * (end_angle - start_angle)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append((x, y))
    return points
