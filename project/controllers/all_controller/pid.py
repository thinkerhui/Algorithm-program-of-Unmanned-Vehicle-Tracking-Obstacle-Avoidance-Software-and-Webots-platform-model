# 吴仰晖创建 2023/09/26
import math

# Initialize PID controller constants 初始化PID控制器
kp = 2.0  # Proportional gain
ki = 0.1  # Integral gain
kd = 0.2  # Derivative gain
prev_error = 0.0
integral = 0.0
velocity = 6.0


def init_global(kp1=2.0, ki1=0.1, kd1=0.2, velocity1=7.0):
    global kp, ki, kd,velocity
    kp = kp1
    ki = ki1
    kd = kd1
    velocity = velocity1


def get(x_car, y_car, angle_car, x_target, y_target):
    global prev_error, integral
    heading = math.atan2(y_target - y_car, x_target - x_car)
    angle_delta = heading - angle_car
    if angle_delta < -3.1416:
        angle_delta = angle_delta + 2 * 3.141596
    elif angle_delta > 3.1416:
        angle_delta = angle_delta - 2 * 3.141596
    error = angle_delta
    if integral < 300:
        integral += error  # 积分
    derivative = error - prev_error  # 微分
    prev_error = error  # 记录上一次的误差值
    # Calculate PID control output
    steering = kp * error + ki * integral + kd * derivative
    output = velocity * 0.5 + steering.__abs__()
    speed_forward = [velocity, velocity, velocity, velocity]
    speed_backward = [-output, -output, -output, -output]
    speed_leftward = [output, -output, output, -output]
    speed_rightward = [-output, output, -output, output]
    speed_leftCircle = [output, -output, -output, output]
    speed_rightCircle = [-output, output, output, -output]
    speed1 = [0] * 4
    speed2 = [0] * 4
    print("angle_diff:", angle_delta)
    if angle_delta < -0.00001:
        speed1 = speed_rightward.copy()
    elif angle_delta > 0.00001:
        speed1 = speed_leftward.copy()
    speed2 = speed_forward.copy()
    return [x + y for x, y in zip(speed1, speed2)]


def reset_integral():
    global integral
    integral = 0.0
