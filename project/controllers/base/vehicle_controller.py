# 车辆通用控制接口
# 提供麦克纳姆轮小车的标准控制接口

from math import hypot, atan2, pi


class VehicleController:
    """车辆通用控制接口"""
    
    # 速度定义
    SPEED_FORWARD = "forward"
    SPEED_BACKWARD = "backward"
    SPEED_LEFTWARD = "leftward"
    SPEED_RIGHTWARD = "rightward"
    SPEED_LEFT_CIRCLE = "left_circle"
    SPEED_RIGHT_CIRCLE = "right_circle"
    
    def __init__(self, base_velocity=7.0):
        self.base_velocity = base_velocity
        self.wheel_names = ["motor1", "motor2", "motor3", "motor4"]
        
    def get_speed_vector(self, speed_type, magnitude=None):
        """获取速度向量
        Args:
            speed_type: 速度类型
            magnitude: 速度大小（可选）
        Returns:
            速度列表 [v1, v2, v3, v4]
        """
        v = magnitude if magnitude else self.base_velocity
        
        speed_map = {
            self.SPEED_FORWARD: [v, v, v, v],
            self.SPEED_BACKWARD: [-v, -v, -v, -v],
            self.SPEED_LEFTWARD: [v, -v, v, -v],
            self.SPEED_RIGHTWARD: [-v, v, -v, v],
            self.SPEED_LEFT_CIRCLE: [v, -v, -v, v],
            self.SPEED_RIGHT_CIRCLE: [-v, v, v, -v],
        }
        
        return speed_map.get(speed_type, [0, 0, 0, 0])
    
    def calculate_heading_error(self, car_x, car_y, car_angle, target_x, target_y):
        """计算朝向误差
        Args:
            car_x, car_y: 车辆位置
            car_angle: 车辆朝向（弧度）
            target_x, target_y: 目标位置
        Returns:
            朝向误差（弧度）
        """
        heading = atan2(target_y - car_y, target_x - car_x)
        angle_delta = heading - car_angle
        
        # 归一化到 [-π, π]
        while angle_delta < -pi:
            angle_delta += 2 * pi
        while angle_delta > pi:
            angle_delta -= 2 * pi
            
        return angle_delta
    
    def calculate_distance(self, x1, y1, x2, y2):
        """计算两点间距离"""
        return hypot(x2 - x1, y2 - y1)
    
    def get_motion_command(self, angle_delta, base_velocity=None):
        """根据角度偏差获取运动命令
        Args:
            angle_delta: 角度偏差（弧度）
            base_velocity: 基础速度
        Returns:
            速度列表
        """
        v = base_velocity if base_velocity else self.base_velocity
        
        if angle_delta < -0.00001:
            # 向右转
            speed1 = self.get_speed_vector(self.SPEED_RIGHTWARD, v)
        elif angle_delta > 0.00001:
            # 向左转
            speed1 = self.get_speed_vector(self.SPEED_LEFTWARD, v)
        else:
            speed1 = [0, 0, 0, 0]
            
        speed2 = self.get_speed_vector(self.SPEED_FORWARD, v)
        
        return [x + y for x, y in zip(speed1, speed2)]
