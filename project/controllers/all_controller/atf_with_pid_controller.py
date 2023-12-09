"""atf_controller controller."""
import sys
import math
import 人工势场 as atf
import pid
# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot,Supervisor

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
    # print(obj.getTypeName())
    if obj.getTypeName() == "OilBarrel":
        print(obj.getField('name').getSFString())
        obstacle.append(obj.getField("translation").getSFVec3f())
obstacle = [(x * 10, y * 10, z * 10) for x, y, z in obstacle]
speed1 = [0] * 4
speed2 = [0] * 4
velocity = 5  # max motor velocity div 2
goalpoint = [45, 45]
pathindex = 0
speed_forward = [velocity, velocity, velocity, velocity]
speed_backward = [-velocity, -velocity, -velocity, -velocity]
speed_leftward = [velocity, -velocity, velocity, -velocity]
speed_rightward = [-velocity, velocity, -velocity, velocity]
speed_leftCircle = [velocity, -velocity, -velocity, velocity]
speed_rightCircle = [-velocity, velocity, velocity, -velocity]
timeStep = int(robot.getBasicTimeStep())
pos = car.getField("translation").getSFVec3f()
pos = [x * 10 for x in pos]
# 初始化prm算法
# 地图中floor size=10，这里写100，也就是放大了10倍。
pathpoint = atf.start(pos[0],pos[1],goalpoint[0],goalpoint[1],obstacle)
# 运行prm算法
print('pathpoint:',pathpoint)

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
    distance = math.sqrt((pathpoint[pathindex][0] - x) ** 2 + (pathpoint[pathindex][1] - y) ** 2)
    print('nowgoal:', pathpoint[pathindex], "  distance:", distance)
    if distance < 0.2:
        print("到达目标:", pathpoint[pathindex])
        pid.reset_integral()
        if pathindex < len(pathpoint):
            pathindex = pathindex + 1
        if pathindex == len(pathpoint):
            print("到达目标点")
            for i in range(4):
                motors[i].setVelocity(0)
            break
    speed = pid.get(x_car=x, y_car=y, angle_car=angle, x_target=pathpoint[pathindex][0], y_target=pathpoint[pathindex][1])
    # set velocity 实际上这里是
    for i in range(4):
        motors[i].setVelocity(speed[i])

    sys.stdout.flush()