from scipy import sparse

from controller import Supervisor, Robot
import sys
import numpy as np
import time
import osqp
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

v_d = 7.4  # 速度
dt = 0.5  # 每一步的时间
sim_steps = 100  # 步数（参考轨迹图像可能会短，但不影响循迹）


# 控制小车移动
def move(v1, v2, v3, v4):
    newv = [1.5 * v1, 1.5 * v2, 1.5 * v3, 1.5 * v4]
    speed = newv.copy()
    # set velocity
    for i in range(4):
        motors[i].setVelocity(speed[i])
    sys.stdout.flush()
    return


# 生成参考轨迹
def load_ref_traj():
    ref_traj = np.zeros((sim_steps, 5))  # 参考轨迹列表，sim_steps行5列

    for i in range(sim_steps):  # 用参数方程的形式给出期望的轨迹，是直线
        ref_traj[i, 0] = v_d * i * dt  # x方向的期望值，期望是以v_d的速度在x方向做匀速直线运动
        ref_traj[i, 1] = 1  # y方向的期望值，期望在y=1的直线上运动
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

# 小车动作控制
speed_forward = [v_d, v_d, v_d, v_d]
speed_backward = [-v_d, -v_d, -v_d, -v_d]
speed_leftward = [v_d, -v_d, v_d, -v_d]
speed_rightward = [-v_d, v_d, -v_d, v_d]
speed_leftCircle = [v_d, -v_d, -v_d, v_d]
speed_rightCircle = [-v_d, v_d, v_d, -v_d]


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
        dx = (v1 + v2 + v3 + v4) / 4  # x方向速度
        dy = (-v1 + v2 - v3 + v4) / 4  # y方向速度
        dtheta = (-v1 - v2 + v3 + v4) / (4 * self.k)  # 角度变化

        self.x += dx * self.dt
        self.y += dy * self.dt
        self.theta += dtheta * self.dt

    def plot_duration(self):  # 画图像的函数
        plt.scatter(self.x, self.y, color='r')
        plt.axis([-5, 400, -4, 4])

        plt.pause(0.008)


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
        """
        Solve函数
        它有三个输入参数：
        x，u_pre和ref_traj。
        x是一个包含当前状态的向量(x,y,yaw)，x为x坐标，y为y坐标，yaw为小车当前朝向
        u_pre是上一时刻的控制输入(，
        ref_traj是一个包含参考轨迹的矩阵。
        """
        # 寻找参考轨迹上的最近距离的参考点,以k-d树的方法去寻找
        tree = KDTree(ref_traj[:, :2])  # 将参考轨迹中的前两列（即x和y坐标）作为输入数据，构建KD树
        nearest_ref_info = tree.query(x[:2])  # 使用k-d树中的查询功能来寻找距离当前状态最近的点
        nearest_ref_x = ref_traj[nearest_ref_info[1]]  # tree.query会返回两个数组，第一个是包含最近邻居距离的数组，第二个是包含最近邻居索引的数组

        # 计算H和F矩阵，就是目标函数的第一、二项的系数矩阵
        # 计算线性时变的一个参数A
        # a和b分别是线性时变系统模型中的状态转移矩阵和控制输入矩阵
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

        # A和B分别是离散时间系统模型中的状态转移矩阵和控制输入矩阵
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

        # theta和phi分别是目标函数中$x_k - x_{ref}$和$u_k$的系数矩阵。
        theta = np.zeros([self.Np * self.Nx, self.Nc * self.Nu])
        phi = np.zeros([self.Np * self.Nx, self.Nu + self.Nx])
        tmp = C  # tmp是一个临时变量，用来存储C乘以A或B的结果。

        # 计算θ值和φ值
        # theta和phi的计算是基于预测步长Np和控制步长Nc的，
        # 它们分别表示了在未来Np个时刻内，状态变量和控制变量与参考状态和零向量的差值。
        for i in range(1, self.Np + 1):  # i从1到Np。通过循环得到theta和phi矩阵
            phi[self.Nx * (i - 1): self.Nx * i] = np.dot(tmp, A)

            tmp_c = np.zeros([self.Nx, self.Nc * self.Nu])
            tmp_c[:, 0: self.Nu] = np.dot(tmp, B)

            if i > 1:
                tmp_c[:, self.Nu:] = theta[self.Nx * (i - 2): self.Nx * (i - 1), 0: -self.Nu]

            theta[self.Nx * (i - 1): self.Nx * i, :] = tmp_c

            tmp = np.dot(tmp, A)

        Q = np.eye(self.Nx * self.Np)  # 权重Q,Q是一个对角矩阵,对角元素代表了对应状态变量的权重

        R = 5.0 * np.eye(self.Nu * self.Nc)  # 权重R，R也是一个对角矩阵，对角元素代表了对应控制量的权重

        rho = 10  # rho是一个松弛因子的权重值，它表示了对松弛变量的惩罚程度。松弛变量是为了处理不等式约束而引入的一个辅助变量。

        #
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

        delta_umin = np.array([[-2], [-2], [-2], [-2]])
        delta_umax = np.array([[2], [2], [2], [2]])

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

# 小车车身参数
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

    # 计算控制量
    u_cur, delta_u_cur = controller.Solve(pos, pre_u, ref_traj)
    abs_u = [v_d, v_d, v_d, v_d] + u_cur
    print(abs_u)

    # 更新模型
    ugv.update(abs_u[0], abs_u[1], abs_u[2], abs_u[3])  # 更新模型
    move(abs_u[0], abs_u[1], abs_u[2], abs_u[3])  # 驱动小车
    ugv.plot_duration()  # 画图

    # 记录控制量
    history_us = np.append(history_us, abs_u)
    if len(history_delta_us) == 0:
        history_delta_us = np.array([u_cur])
    else:
        history_delta_us = np.vstack((history_delta_us, u_cur))

    # 更新位置
    pos = pos + np.array([(abs_u[0] + abs_u[1] + abs_u[2] + abs_u[3]) / 4 * dt,
                          (-abs_u[0] + abs_u[1] - abs_u[2] + abs_u[3]) / 4 * dt,
                          (-abs_u[0] - abs_u[1] + abs_u[2] + abs_u[3]) / 4 * (a + b) * dt])

    # 更新前一次控制量
    pre_u = u_cur
    sim_steps -= 1

for i in range(4):
    motors[i].setVelocity(stop[i])
t2 = time.time()
print('程序运行时间:%s秒' % (t2 - t1))
plt.show()
