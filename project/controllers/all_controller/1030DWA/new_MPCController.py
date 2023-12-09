import math

from scipy import sparse

from controller import Supervisor, Robot
import sys
import numpy as np
import time
import osqp
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

v_d = 7.4  # 速度
dt = 0.16  # 每一步的时间
sim_steps = 500  # 步数（参考轨迹图像可能会短，但不影响循迹）
wheel_radius = 0.05


# 控制小车移动
def move(v1, v2, v3, v4):
    newv = [v1, v2, v3, v4]
    speed = newv.copy()
    # set velocity
    for i in range(4):
        motors[i].setVelocity(speed[i])
    robot.step(timeStep * 5)
    return


# 生成参考轨迹
def load_ref_traj():
    ref_traj = np.zeros((sim_steps, 5))  # 参考轨迹列表，sim_steps行5列

    for i in range(sim_steps):  # 用参数方程的形式给出期望的轨迹，是直线
        ref_traj[i, 0] = v_d * i * dt  # x方向的期望值，期望是以v_d的速度在x方向做匀速直线运动
        ref_traj[i, 1] = -1.0  # y方向的期望值，期望在y=1的直线上运动
        ref_traj[i, 2] = 0.0  # 车身和x轴方向的偏移量
        ref_traj[i, 3] = v_d  # 小车速度，匀速运动
        ref_traj[i, 4] = 0  # 车前轮的偏角，期望不转弯

    return ref_traj


robot = Supervisor()
# Supervisor是一个特殊的机器人节点，记得要在webots把对应机器人节点的supervisor设置为true
car = robot.getFromDef('robot')

wheels_names = ["motor1", "motor2", "motor3", "motor4"]
motors = [robot.getMotor(name) for name in wheels_names]
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

speed1 = [0] * 4
speed2 = [0] * 4
stop = [0] * 4


# 基于车辆运动学模型实现一个简单的仿真环境
class UGV_model:
    def __init__(self, x0, y0, theta0, a, b, T):  # a是到车身坐标系X轴距离，b是到Y轴距离
        self.x = x0  # X
        self.y = y0  # Y
        self.theta = theta0  # 小车的朝向（与大地坐标系的x轴的夹角）
        self.a = a  # 到x轴距离
        self.b = b  # 到y轴距离
        self.k = a + b
        self.dt = T  # decision time periodic-决策周期时间

    def update(self, v1, v2, v3, v4):  # update ugv's state-更新小车状态
        # 小车状态的变化量
        dx = wheel_radius * (v1 + v2 + v3 + v4) / 4  # x方向速度
        dy = wheel_radius * (-v1 + v2 - v3 + v4) / 4  # y方向速度
        dv = wheel_radius * (v1 + v2 + v3 + v4) / 4
        dtheta = (-v1 - v2 + v3 + v4) / (4 * self.k)  # 角度变化
        print("v1:%f v2:%f v3:%f v4:%f" % (v1, v2, v3, v4))
        print("dv:%f dtheta:%f" % (dv, dtheta))
        # self.x = self.x + dv * math.cos(self.theta) * self.dt
        # self.y = self.y + dv * math.sin(self.theta) * self.dt
        self.x += dx * self.dt
        self.y += dy * self.dt
        self.theta += dtheta * self.dt

    def plot_duration(self):  # 画图像的函数
        plt.scatter(self.x, self.y, color='r')
        plt.axis([-5, 10, -4, 4])

        plt.pause(0.001)


class MPCController:  # MPC控制器设计
    def __init__(self, a, b, dt):
        self.a = a  # 到x轴距离
        self.b = b  # 到y轴距离
        self.k = a + b

        self.Nx = 3  # 状态量个数
        self.Nu = 4  # 控制量个数

        self.Nc = 5  # 控制时域
        self.Np = 30  # 预测时域

        self.T = dt

    def Solve(self, x, u_pre, ref_traj):
        # 寻找参考轨迹上的最近的参考点,以k-d树的方法去寻找
        tree = KDTree(ref_traj[:, :2])
        nearest_ref_info = tree.query(x[:2])
        nearest_ref_x = ref_traj[nearest_ref_info[1]]

        # 计算H和F矩阵，就是目标函数的第一、二项的系数矩阵
        # 计算线性时变的一个参数A
        a = np.array([
            [1.0, 0, 0],
            [0, 1.0, 0],
            [0, 0, 1.0]
        ])

        # 计算B
        b = np.array([
            [self.T / 4, self.T / 4, self.T / 4, self.T / 4],
            [-self.T / 4, self.T / 4, -self.T / 4, self.T / 4],
            [-self.T / (4 * self.k), -self.T / (4 * self.k),
             self.T / (4 * self.k), self.T / (4 * self.k)]
        ])
        b = b * wheel_radius

        # 计算tildeA
        A = np.zeros([self.Nx + self.Nu, self.Nx + self.Nu])
        A[0: self.Nx, 0: self.Nx] = a
        A[0: self.Nx, self.Nx:] = b
        A[self.Nx:, self.Nx:] = np.eye(self.Nu)

        # 计算tildeB
        B = np.zeros([self.Nx + self.Nu, self.Nu])
        B[0: self.Nx, :] = b
        B[self.Nx:, :] = np.eye(self.Nu)

        C = np.array([[1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0]])

        theta = np.zeros([self.Np * self.Nx, self.Nc * self.Nu])
        phi = np.zeros([self.Np * self.Nx, self.Nu + self.Nx])
        tmp = C

        # 计算θ值和φ值
        for i in range(1, self.Np + 1):  # i从1到Np
            phi[self.Nx * (i - 1): self.Nx * i] = np.dot(tmp, A)

            tmp_c = np.zeros([self.Nx, self.Nc * self.Nu])
            tmp_c[:, 0: self.Nu] = np.dot(tmp, B)

            if i > 1:
                tmp_c[:, self.Nu:] = theta[self.Nx * (i - 2): self.Nx * (i - 1), 0: -self.Nu]

            theta[self.Nx * (i - 1): self.Nx * i, :] = tmp_c

            tmp = np.dot(tmp, A)

        Q = np.eye(self.Nx * self.Np)  # 权重Q

        R = 5.0 * np.eye(self.Nu * self.Nc)  # 权重R

        rho = 10  # 松弛因子的权重值

        H = np.zeros((self.Nu * self.Nc + 1, self.Nu * self.Nc + 1))
        H[0: self.Nu * self.Nc, 0: self.Nu * self.Nc] = np.dot(np.dot(theta.transpose(), Q), theta) + R
        H[-1: -1] = rho  # 加入权重值

        # 计算状态量ξ
        kesi = np.zeros((self.Nx + self.Nu, 1))
        diff_x = x - nearest_ref_x[:3]
        diff_x = diff_x.reshape(-1, 1)  # 数据转换成1列
        kesi[: self.Nx, :] = diff_x
        diff_u = u_pre.reshape(-1, 1)
        kesi[self.Nx:, :] = diff_u

        F = np.zeros((1, self.Nu * self.Nc + 1))
        F_1 = 2 * np.dot(np.dot(np.dot(phi, kesi).transpose(), Q), theta)
        F[0, 0: self.Nu * self.Nc] = F_1

        # constraints-设置车速度和加速度的阈值
        umin = np.array([[-14.8], [-14.8], [-14.8], [-14.8]])
        umax = np.array([[14.8], [14.8], [14.8], [14.8]])

        delta_umin = np.array([[-7.4], [-7.4], [-7.4], [-7.4]])
        delta_umax = np.array([[7.4], [7.4], [7.4], [7.4]])

        # 计算A_I和约束条件
        A_t = np.zeros((self.Nc, self.Nc))
        for row in range(self.Nc):  # 初始化下三角矩阵
            for col in range(self.Nc):
                if row >= col:
                    A_t[row, col] = 1.0

        A_I = np.kron(A_t, np.eye(self.Nu))

        A_cons = np.zeros((self.Nc * self.Nu, self.Nc * self.Nu + 1))
        A_cons[0: self.Nc * self.Nu, 0: self.Nc * self.Nu] = A_I

        U_t = np.kron(np.ones((self.Nc, 1)), u_pre.reshape(-1, 1))

        U_min = np.kron(np.ones((self.Nc, 1)), umin)
        U_max = np.kron(np.ones((self.Nc, 1)), umax)

        LB = U_min - U_t
        UB = U_max - U_t

        delta_Umin = np.kron(np.ones((self.Nc, 1)), delta_umin)
        delta_Umax = np.kron(np.ones((self.Nc, 1)), delta_umax)

        delta_Umin = np.vstack((delta_Umin, [0]))
        delta_Umax = np.vstack((delta_Umax, [rho]))

        A_1_cons = np.eye(self.Nc * self.Nu + 1, self.Nc * self.Nu + 1)

        A_cons = np.vstack((A_cons, A_1_cons))

        LB = np.vstack((LB, delta_Umin))
        UB = np.vstack((UB, delta_Umax))

        # Create an OSQP object
        prob = osqp.OSQP()

        H = sparse.csc_matrix(H)
        A_cons = sparse.csc_matrix(A_cons)  # 压缩稀疏矩阵

        # Setup workspace
        prob.setup(H, F.transpose(), A_cons, LB, UB)

        res = prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        u_cur = u_pre + res.x[0: self.Nu]
        return u_cur, res.x[0: self.Nu]


# pos = car.getField("translation").getSFVec3f()
pos = np.array([0.0, 0.0, 0.0])  # 小车的起始位置，表示x位置，y位置和起始偏角
pre_u = np.array([0.0, 0.0, 0.0, 0.0])  # 小车当前的控制量

a = 0.15
b = 0.1

ref_traj = load_ref_traj()
plt.figure(figsize=(15, 5))
plt.plot(ref_traj[:, 0], ref_traj[:, 1], '-.b', linewidth=5.0)

history_us = np.array([])
history_delta_us = np.array([])

# 开始运动
timeStep = int(robot.getBasicTimeStep())

t1 = time.time()
ugv = UGV_model(pos[0], pos[1], pos[2], a, b, dt)
controller = MPCController(a, b, dt)

while robot.step(timeStep) != -1 and sim_steps > 0:
    # new_pos = car.getField("translation").getSFVec3f()
    # pos = np.array([new_pos[0],
    #                 -new_pos[1],
    #                 -car.getField("rotation").getSFRotation()[3]])
    # for i in range(4):
    #     pre_u[i] = motors[i].getVelocity()
    pos = [ugv.x, ugv.y, ugv.theta]
    # print(pre_u)
    u_cur, delta_u_cur = controller.Solve(pos, pre_u, ref_traj)
    abs_u = [v_d, v_d, v_d, v_d] + u_cur
    # print(abs_u)

    # move(abs_u[0], abs_u[1], abs_u[2], abs_u[3])  # 驱动小车
    ugv.update(abs_u[0], abs_u[1], abs_u[2], abs_u[3])  # 更新模型

    ugv.plot_duration()     # 画图

    history_us = np.append(history_us, abs_u)
    if len(history_delta_us) == 0:
        history_delta_us = np.array([u_cur])
    else:
        history_delta_us = np.vstack((history_delta_us, u_cur))

    rpos = pos + np.array([wheel_radius * (abs_u[0] + abs_u[1] + abs_u[2] + abs_u[3]) / 4 * dt,
                           wheel_radius * (-abs_u[0] + abs_u[1] - abs_u[2] + abs_u[3]) / 4 * dt,
                           wheel_radius * (-abs_u[0] - abs_u[1] + abs_u[2] + abs_u[3]) / 4 * (a + b) * dt])

    print('车的参考位置%s' % rpos)
    print('车现在的位置%s' % pos)
    pre_u = u_cur
    sim_steps -= 1

for i in range(4):
    motors[i].setVelocity(stop[i])
t2 = time.time()
print('程序运行时间:%s秒' % (t2 - t1))
# plt.show()
