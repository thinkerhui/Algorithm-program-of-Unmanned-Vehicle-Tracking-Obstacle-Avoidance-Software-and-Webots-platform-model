import numpy as np
import matplotlib.pyplot as plt
import copy
from celluloid import Camera  # 保存动图时用，pip install celluloid

def start(start_x,start_y,goal_x,goal_y,obstacle):
    ## 初始化车的参数
    #d = 3.5  # 道路标准宽度

    #W = 1.8  # 汽车宽度

    #L = 4.7  # 车长

    #P0 = np.array([0, - d / 2, 1, 1])  # 车辆起点位置，分别代表x,y,vx,vy

    P0 = np.array([start_x, start_y, 0, 0])

    Pg = np.array([goal_x, goal_y, 0, 0])  # 目标位置

    # 障碍物位置

    Pobs = []
    for ob in obstacle:
        p = [ob[0],ob[1],0,0]
        Pobs.append(p)
    Pobs = np.array(Pobs)
    P = np.vstack((Pg, Pobs))  # 将目标位置和障碍物位置合放在一起，即是将Pg加入到Pobs的最后

    Eta_att = 15  # 引力的增益系数

    Eta_rep_ob = 150  # 斥力的增益系数

    #Eta_rep_edge = 50  # 道路边界斥力的增益系数

    d0 = 100  # 障碍影响的最大距离

    num = P.shape[0]  # 障碍与目标总计个数，即P有多少行

    len_step = 1  # 步长

    n = 2

    Num_iter = 500  # 最大循环迭代次数

    path = []  # 保存车走过的每个点的坐标
    delta = np.zeros((num, 2))  # 保存车辆当前位置与障碍物的方向向量，方向指向车辆；以及保存车辆当前位置与目标点的方向向量，方向指向目标点
    dists = []  # 保存车辆当前位置与障碍物的距离以及车辆当前位置与目标点的距离
    unite_vec = np.zeros((num, 2))  # 保存车辆当前位置与障碍物的单位方向向量，方向指向车辆；以及保存车辆当前位置与目标点的单位方向向量，方向指向目标点

    F_rep_ob = np.zeros((len(Pobs), 2))  # 存储每一个障碍到车辆的斥力,带方向

    ## ***************初始化结束，开始主体循环******************
    Pi = P0[0:2]  # 当前车辆位置

    # count=0
    for i in range(Num_iter):
        if ((Pi[0] - Pg[0]) ** 2 + (Pi[1] - Pg[1]) ** 2) ** 0.5 < 1:
            break         #判断车辆是否一开始就在目标点附近
        dists = []
        path.append(Pi)
        # print(count)
        # count+=1
        # 计算车辆当前位置与障碍物的单位方向向量
        for j in range(len(Pobs)):
            delta[j] = Pi[0:2] - Pobs[j, 0:2]
            dists.append(np.linalg.norm(delta[j]))
            unite_vec[j] = delta[j] / dists[j]
        # 计算车辆当前位置与目标的单位方向向量
        delta[len(Pobs)] = Pg[0:2] - Pi[0:2]
        dists.append(np.linalg.norm(delta[len(Pobs)]))
        unite_vec[len(Pobs)] = delta[len(Pobs)] / dists[len(Pobs)]

        ## 计算引力
        F_att = Eta_att * dists[len(Pobs)] * unite_vec[len(Pobs)]

        ## 计算斥力
        # 在原斥力势场函数增加目标调节因子（即车辆至目标距离），以使车辆到达目标点后斥力也为0
        for j in range(len(Pobs)):
            if dists[j] >= d0:
                F_rep_ob[j] = np.array([0, 0])
            else:
                # 障碍物的斥力1，方向由障碍物指向车辆
                F_rep_ob1_abs = Eta_rep_ob * \
                                (1 / dists[j] - 1 / d0) * \
                                (dists[len(Pobs)]) ** n / dists[j] ** 2  # 斥力大小
                F_rep_ob1 = F_rep_ob1_abs * unite_vec[j]  # 斥力向量
                # 障碍物的斥力2，方向由车辆指向目标点
                F_rep_ob2_abs = n / 2 * Eta_rep_ob * \
                                (1 / dists[j] - 1 / d0) ** 2 * \
                                (dists[len(Pobs)]) ** (n - 1)  # 斥力大小
                F_rep_ob2 = F_rep_ob2_abs * unite_vec[len(Pobs)]  # 斥力向量
                # 改进后的障碍物合斥力计算
                F_rep_ob[j] = F_rep_ob1 + F_rep_ob2
        '''
        # 增加道路边界斥力势场，根据车辆当前位置，选择对应的斥力函数
        if Pi[1] > - d + W / 2 and Pi[1] <= - d / 2:
            F_rep_edge = [0, Eta_rep_edge * v *
                        np.exp(-d / 2 - Pi[1])]  # 下道路边界区域斥力势场，方向指向y轴正向
        elif Pi[1] > - d / 2 and Pi[1] <= - W / 2:
            F_rep_edge = np.array([0, 1 / 3 * Eta_rep_edge * Pi[1] ** 2])
        elif Pi[1] > W / 2 and Pi[1] <= d / 2:
            F_rep_edge = np.array([0, - 1 / 3 * Eta_rep_edge * Pi[1] ** 2])
        elif Pi[1] > d / 2 and Pi[1] <= d - W / 2:
            F_rep_edge = np.array([0, Eta_rep_edge * v * (np.exp(Pi[1] - d / 2))])
        '''
        ## 计算合力和方向
        #F_rep = np.sum(F_rep_ob, axis=0) + F_rep_edge

        F_sum = F_att + np.sum(F_rep_ob, axis=0)

        UnitVec_Fsum = 1 / np.linalg.norm(F_sum) * F_sum
        # 计算车的下一步位置
        Pi = copy.deepcopy(Pi + len_step * UnitVec_Fsum)
        # Pi[0:2] = Pi[0:2] + len_step * UnitVec_Fsum
        # print(Pi)

    path.append(Pg[0:2])  # 最后把目标点也添加进路径中
    path = np.array(path)  # 转为numpy
    return path


