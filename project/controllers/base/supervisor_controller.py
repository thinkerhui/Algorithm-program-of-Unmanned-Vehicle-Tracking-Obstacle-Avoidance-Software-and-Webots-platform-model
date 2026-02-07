# 基础Supervisor控制器类
# 提供Webots Supervisor的基本功能封装

from controller import Supervisor


class SupervisorController:
    """基础Supervisor控制器，提供通用的机器人控制功能"""
    
    def __init__(self):
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.motors = {}
        self.car = None
        
    def get_robot(self):
        """获取机器人实例"""
        return self.robot
    
    def get_time_step(self):
        """获取仿真时间步长"""
        return self.time_step
    
    def get_car(self):
        """获取车辆节点"""
        return self.car
    
    def init_motors(self, wheel_names):
        """初始化电机
        Args:
            wheel_names: 电机名称列表
        """
        for name in wheel_names:
            motor = self.robot.getMotor(name)
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)
            self.motors[name] = motor
    
    def set_velocity(self, speeds):
        """设置电机速度
        Args:
            speeds: 速度列表 [v1, v2, v3, v4]
        """
        for i, motor in enumerate(self.motors.values()):
            motor.setVelocity(speeds[i])
    
    def get_position(self):
        """获取车辆位置"""
        if self.car:
            pos = self.car.getField("translation").getSFVec3f()
            return [pos[0] * 10, pos[1] * 10]  # 转换为实际坐标
        return [0, 0]
    
    def get_rotation(self):
        """获取车辆旋转角度"""
        if self.car:
            rotation = self.car.getField("rotation").getSFRotation()
            return -rotation[3]  # Webots中旋转角是顺时针增长
        return 0
    
    def step(self):
        """执行仿真一步"""
        return self.robot.step(self.time_step)
    
    def cleanup(self):
        """清理资源"""
        for motor in self.motors.values():
            motor.setVelocity(0.0)
