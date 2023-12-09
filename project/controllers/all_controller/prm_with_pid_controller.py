# 吴仰晖创建 2023.9.24
from controller import Supervisor, Robot
import sys
import math
import prm算法 as prm

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
obstacle = [(x * 10, y * 10) for x, y, z in obstacle]
speed1 = [0] * 4
speed2 = [0] * 4
velocity = 7  # max motor velocity div 2
goalpoint = [40, 40]
pathindex = 0
timeStep = int(robot.getBasicTimeStep())
pos = car.getField("translation").getSFVec3f()
pos = [x * 10 for x in pos]
# 初始化prm算法
# 地图中floor size=10，这里写100，也就是放大了10倍。
prm.init_global(start_position_x=pos[0], start_position_y=pos[1], goal_position_x=goalpoint[0],
                goal_position_y=goalpoint[1], obstacle=obstacle, size_robot=2, x_range=100, y_range=100,
                n_sample=500, show_animat=False, n_knn=50, max_edge_len=10)
# 运行prm算法
rx, ry = prm.start()
pathpoint = [(x, y) for x, y in zip(rx, ry)]
pathpoint = pathpoint[::-1]

# Initialize PID controller constants 初始化PID
kp = 2.0  # Proportional gain
ki = 0.1  # Integral gain
kd = 0.2  # Derivative gain
prev_error = 0.0
integral = 0.0

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
    heading = math.atan2(pathpoint[pathindex][1] - y, pathpoint[pathindex][0] - x)
    # angle_delta是目前小车的方向和目标方向之间的差值
    angle_delta = heading - angle
    # print('car angle:', angle)
    # print('obstacle angle:', heading)
    if angle_delta < -3.1416:
        angle_delta = angle_delta + 2 * 3.141596
    elif angle_delta > 3.1416:
        angle_delta = angle_delta - 2 * 3.141596

    error = angle_delta  # 误差
    if integral < 500:
        integral += error  # 积分
    derivative = error - prev_error  # 微分
    prev_error = error  # 记录上一次的误差值

    # Calculate PID control output
    steering = kp * error + ki * integral + kd * derivative
    output = velocity*0.5 + steering.__abs__()
    speed_forward = [velocity, velocity, velocity, velocity]
    speed_backward = [-output, -output, -output, -output]
    speed_leftward = [output, -output, output, -output]
    speed_rightward = [-output, output, -output, output]
    speed_leftCircle = [output, -output, -output, output]
    speed_rightCircle = [-output, output, output, -output]
    # speed = [0] * 4
    # # Update wheel speeds
    # speed[0] = velocity - steering
    # speed[1] = velocity + steering
    # speed[2] = velocity - steering
    # speed[3] = velocity + steering
    print("angle_diff:", angle_delta)
    if angle_delta < -0.00001:
        speed1 = speed_rightward.copy()
    elif angle_delta > 0.00001:
        speed1 = speed_leftward.copy()
    # speed1 = speed_leftward.copy()
    speed2 = speed_forward.copy()
    distance = math.sqrt((pathpoint[pathindex][0] - x) ** 2 + (pathpoint[pathindex][1] - y) ** 2)
    print('nowgoal:', pathpoint[pathindex], "  distance:", distance)
    if distance < 0.2:
        print("到达目标:", pathpoint[pathindex])
        # prev_error = 0
        integral = 0
        if pathindex < len(pathpoint):
            pathindex = pathindex + 1
        if pathindex == len(pathpoint):
            print("到达目标点")
            break
    # set velocity 实际上这里是
    for i in range(4):
        motors[i].setVelocity(speed1[i] + speed2[i])
    # Set wheel velocities
    # for i in range(4):
    #     motors[i].setVelocity(speed[i])

    sys.stdout.flush()
