from controller import Supervisor, Robot
import sys
import math
from a_star import *

robot = Supervisor()
# Supervisor是一个特殊的机器人节点，记得要在webots把对应机器人节点的supervisor设置为true
car = robot.getFromDef('robot')
wheels_names = ["motor1", "motor2", "motor3", "motor4"]
motors = [robot.getMotor(name) for name in wheels_names]
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
# 获取世界中的所有对象
world = robot.getRoot()
world_children = world.getField('children')
n = world_children.getCount()
print("This world contains nodes:", n)
# 遍历所有对象，找到名字符合要求的Oilbarrel对象node并存入obstacle
obstacle = []
for i in range(n):
    obj = world_children.getMFNode(i)
    if obj.getTypeName() == "OilBarrel":
        obstacle.append(obj.getField("translation").getSFVec3f()[:2])  # 只保留x, y两个坐标
obstacle = [(x * 10, y * 10) for x, y in obstacle]
speed1 = [0] * 4
speed2 = [0] * 4
velocity = 5  # max motor velocity div 2
goal_point = (45, 45)
path_index = 0
speed_forward = [velocity, velocity, velocity, velocity]
speed_backward = [-velocity, -velocity, -velocity, -velocity]
speed_leftward = [velocity, -velocity, velocity, -velocity]
speed_rightward = [-velocity, velocity, -velocity, velocity]
speed_leftCircle = [velocity, -velocity, -velocity, velocity]
speed_rightCircle = [-velocity, velocity, velocity, -velocity]
timeStep = int(robot.getBasicTimeStep())
pos = car.getField("translation").getSFVec3f()
pos = [x * 10 for x in pos]
start_point = (pos[0], pos[1])
# 若使用先验地图信息则先进行路径规划
# 得到结果后再令机器人按照规划的结果行进即可

# 运行a_star算法
# create env
print('start_point: ', start_point)
print('obstacle: ', obstacle)
env = Env(100, 100, obstacle)

# a_star path planning
a_star = AStar(start_point, goal_point, env, 2)
path, _ = a_star.searching()

# 此处为控制机器人按照路径行进
while robot.step(timeStep) != -1:
    rotation = car.getField("rotation").getSFRotation()
    # print('pos:', pos)
    # print('rotation:', rotation)
    pos = car.getField("translation").getSFVec3f()
    pos = [x * 10 for x in pos]
    x = pos[0]
    y = pos[1]
    # 发现在这个webots世界中旋转角是顺时针增长的,和atan2的计算的角度相反（互为正负）
    angle = -rotation[3]
    heading = math.atan2(path[path_index][1] - y, path[path_index][0] - x)
    angle_delta = heading - angle
    # print('car angle:', angle)
    # print('obstacle angle:', heading)
    if angle_delta < -3.1416:
        angle_delta = angle_delta + 2 * 3.141596
    elif angle_delta > 3.1416:
        angle_delta = angle_delta - 2 * 3.141596
    # print("angle_diff:", angle_delta)
    if angle_delta < -0.05:
        speed1 = speed_rightward.copy()
    elif angle_delta > 0.05:
        speed1 = speed_leftward.copy()
    speed2 = speed_forward.copy()
    distance = math.sqrt((path[path_index][0] - x) ** 2 + (path[path_index][1] - y) ** 2)
    print('now goal:', path[path_index], "  distance:", distance)
    if distance < 0.05:
        print("到达目标:", path[path_index])
        if path_index < len(path):
            path_index = path_index + 1
        if path_index == len(path):
            print("到达目标点")
            break
    # set velocity
    for i in range(4):
        motors[i].setVelocity(speed1[i] + speed2[i])

    sys.stdout.flush()

