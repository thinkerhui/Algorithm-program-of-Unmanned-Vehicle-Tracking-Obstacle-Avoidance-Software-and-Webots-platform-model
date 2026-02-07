from controller import Supervisor, Robot
import sys
import math

robot = Supervisor()
# Supervisor是一个特殊的机器人节点，记得要在webots把对应机器人节点的supervisor设置为true
car = robot.getFromDef('robot')
wheels_names = ["motor1", "motor2", "motor3", "motor4"]
motors = [robot.getMotor(name) for name in wheels_names]
# 获取世界中的所有对象
world = robot.getRoot()
world_children = world.getField('children')
n = world_children.getCount()
print("This world contains nodes:",n)
# 遍历所有对象，找到名字符合要求的Oilbarrel对象node并存入obstacle
obstacle = []
for i in range(n):
    obj = world_children.getMFNode(i)
    # print(obj.getTypeName())
    if obj.getTypeName() == "OilBarrel":
        print(obj.getField('name').getSFString())
        obstacle.append(obj)
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)
speed1 = [0] * 4
speed2 = [0] * 4
velocity = 7.4  # max motor velocity div 2
goalpoint = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
goalindex = 0
speed_forward = [velocity, velocity, velocity, velocity]
speed_backward = [-velocity, -velocity, -velocity, -velocity]
speed_leftward = [velocity, -velocity, velocity, -velocity]
speed_rightward = [-velocity, velocity, -velocity, velocity]
speed_leftCircle = [velocity, -velocity, -velocity, velocity]
speed_rightCircle = [-velocity, velocity, velocity, -velocity]
timeStep = int(robot.getBasicTimeStep())

while robot.step(timeStep) != -1:
    pos = car.getField("translation").getSFVec3f()
    rotation = car.getField("rotation").getSFRotation()
    # print('pos:', pos)
    # print('rotation:', rotation)
    x = pos[0]
    y = pos[1]
    # 发现在这个webots世界中旋转角是顺时针增长的,和atan2的计算的角度相反（互为正负）
    angle = -rotation[3]
    heading = math.atan2(goalpoint[goalindex][1] - y, goalpoint[goalindex][0] - x)
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
    distance = math.sqrt((goalpoint[goalindex][0] - x) ** 2 + (goalpoint[goalindex][1] - y) ** 2)
    # print('nowgoal:', goalpoint[goalindex], "  distance:", distance)
    if distance < 0.05:
        print("到达目标:",goalpoint[goalindex])
        if goalindex<3:
            goalindex = goalindex + 1
        else:
            goalindex = 0
        print("向新目标前进：",goalpoint[goalindex])
    # set velocity
    for i in range(4):
        motors[i].setVelocity(speed1[i] + speed2[i])

    sys.stdout.flush()
