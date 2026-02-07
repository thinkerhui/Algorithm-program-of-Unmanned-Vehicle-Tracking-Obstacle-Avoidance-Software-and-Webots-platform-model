# 几何计算工具
# 提供常用的几何计算功能

import math


class Point:
    """二维点类"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance_to(self, other):
        """计算到另一点的距离"""
        return math.hypot(self.x - other.x, self.y - other.y)
    
    def angle_to(self, other):
        """计算到另一点的角度"""
        return math.atan2(other.y - self.y, other.x - self.x)
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)
    
    def __str__(self):
        return f"Point({self.x:.2f}, {self.y:.2f})"


class Line:
    """线段类"""
    
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def length(self):
        """获取线段长度"""
        return self.p1.distance_to(self.p2)
    
    def midpoint(self):
        """获取线段中点"""
        return Point((self.p1.x + self.p2.x) / 2, (self.p1.y + self.p2.y) / 2)
    
    def angle(self):
        """获取线段角度"""
        return self.p1.angle_to(self.p2)
    
    def closest_point(self, point):
        """获取线上最接近给定点的点"""
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        
        if dx == 0 and dy == 0:
            return self.p1
        
        t = ((point.x - self.p1.x) * dx + (point.y - self.p1.y) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        
        return Point(self.p1.x + t * dx, self.p1.y + t * dy)


def circle_intersection(c1, r1, c2, r2):
    """计算两圆交点
    Args:
        c1, c2: 圆心
        r1, r2: 半径
    Returns:
        交点列表，可能有0、1或2个交点
    """
    d = c1.distance_to(c2)
    
    # 无交点的情况
    if d > r1 + r2:
        return []
    if d < abs(r1 - r2):
        return []
    if d == 0 and r1 == r2:
        return []
    
    # 计算交点
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(max(0, r1 * r1 - a * a))
    
    x2 = c1.x + a * (c2.x - c1.x) / d
    y2 = c1.y + a * (c2.y - c1.y) / d
    
    if h == 0:
        return [Point(x2, y2)]
    
    rx = -(c2.y - c1.y) * (h / d)
    ry = (c2.x - c1.x) * (h / d)
    
    return [
        Point(x2 + rx, y2 + ry),
        Point(x2 - rx, y2 - ry)
    ]


def point_in_circle(point, center, radius):
    """检查点是否在圆内"""
    return point.distance_to(center) <= radius


def point_on_line(point, line, tolerance=1e-6):
    """检查点是否在线上"""
    closest = line.closest_point(point)
    return point.distance_to(closest) <= tolerance
