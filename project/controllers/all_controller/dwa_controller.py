import numpy as np

from controller import Supervisor, Robot
import sys
import math
import prm算法 as prm
import dwa

robot = Supervisor()
robot.getSynchronization()
# Supervisor是一个特殊的机器人节点，记得要在webots把对应机器人节点的supervisor设置为true
car = robot.getFromDef('robot')
wheels_names = ["motor1", "motor2", "motor3", "motor4"]
motors = [robot.getMotor(name) for name in wheels_names]
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
# 小车车身属性
a = 0.15
b = 0.1
wheel_radius = 0.05
# 获取世界中的所有对象
world = robot.getRoot()
world_children = world.getField('children')
n = world_children.getCount()
print("This world contains nodes:", n)
# 遍历所有对象，找到名字符合要求的Oilbarrel对象node并存入obstacle
obstacle = []
for i in range(n):
    obj = world_children.getMFNode(i)
    # print(obj.getTypeName())
    if obj.getTypeName() == "OilBarrel":
        # print(obj.getField('name').getSFString())
        obstacle.append(obj.getField("translation").getSFVec3f())
obstacle = [[x, y] for x, y, z in obstacle]
print(obstacle)
obstacle = np.array(obstacle)

speed1 = [0] * 4

velocity = 7  # max motor velocity div 2
# goalpoint为设立的目标点
goalpoint = [4.0, 4.0]
timeStep = int(robot.getBasicTimeStep())
print("timeStep:", timeStep)
rotation = car.getField("rotation").getSFRotation()
angle = -rotation[3]
pos = car.getField("translation").getSFVec3f()
# 初始化dwa算法
dwa_object = dwa.init(obstacle, goalpoint)
now_velocity = 0
while robot.step(timeStep) != -1:
    # 其实这里有个瑕疵，就是这while的一个timestep内其实是没有发出指令的
    rotation = car.getField("rotation").getSFRotation()  # 获取小车的旋转角度，我们实际只要用到围绕z轴的旋转角度，也就是rotation[3]
    pos = car.getField("translation").getSFVec3f()  # 获取小车当前位置，我们要用到x坐标pos[0],y坐标pos[1]
    # print('pos:', pos)
    # print('rotation:', rotation)
    # getVelocity()返回一个正好包含6个值的向量。前三个分别是x、y和z方向上的线速度。后三个分别是围绕x、y和z轴的角速度。
    now_velocity = car.getVelocity()
    # 计算当前线速度大小
    linear_velocity = math.sqrt(now_velocity[0] ** 2 + now_velocity[1] ** 2)
    x = pos[0]
    y = pos[1]
    # now_state是获取到的当前的小车的状态，从左往右依次是:x坐标，y坐标，小车朝向角度，线速度大小，绕z轴的角速度大小
    now_state = np.array([pos[0], pos[1], -rotation[3], linear_velocity, now_velocity[5]])
    u, predicted_trajectory = dwa_object.dwa_control(now_state, goalpoint, obstacle)
    # 预测十步轨迹，把这十步轨迹用完
    # 这里的配置和dwa的相关参数有关
    # 例如：timestep在这里是32ms，然后dwa的dt可以设为32ms的十倍也就是320ms,然后predict_time设为3.2s，这样就预测10步(实际上不止10步，有12步)
    # 我们在下面的for循环把这十步走完，就是设置轮子的速度
    for i in range(10):
        v = predicted_trajectory[i][3]
        w = predicted_trajectory[i][4]
        # 通过速度和角速度计算出每个轮子的控制速度
        v1 = v - w * (a + b)
        v2 = v - w * (a + b)
        v3 = v + w * (a + b)
        v4 = v + w * (a + b)
        # 要注意和真实模拟世界的轮子对应
        speed1 = [v1, v2, v3, v4]
        # 要注意我们计算的应该是角速度 v=w*r w=v/r
        speed1 = [x / wheel_radius for x in speed1]
        print(speed1)
        # print("control:",u," predicted_trajectory:",predicted_trajectory)
        # 取预测轨迹的最后一个点作为当前的目标状态
        goal_state = predicted_trajectory[-1]
        # speed2 = speed_forward.copy()
        print('nowgoal:', (goal_state[0], goal_state[1]))
        for i in range(4):
            motors[i].setVelocity(speed1[i])
        robot.step(timeStep * 10)

    sys.stdout.flush()
for i in range(4):
    motors[i].setVelocity(0)
