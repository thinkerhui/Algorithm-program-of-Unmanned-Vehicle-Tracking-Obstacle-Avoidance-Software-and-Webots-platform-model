# 吴仰晖创建 2023/04/21
import math

import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera  # 保存动图时用，pip install celluloid
from scipy.spatial import KDTree

plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
plt.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# parameter
N_SAMPLE = 200  # 采样点数目，即随机点集V的大小
N_KNN = 30  # 一个采样点的领域点个数
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length 边的最大长度
MAP_MAX_LENGTH = 60.0  # [m] Map maximum length
MAP_MAX_WIDTH = 60.0  # [m] Map maximum width
show_animation = True  # 生成动画的开关

'''
下面是设置障碍物，
ox[]用来存障碍物的x坐标
oy[]用来存障碍物的y坐标
'''
ox = []
oy = []

# start and goal position
# (sx,sy)为开始位置坐标，(gx,gy)为目标地点坐标，robot_size为机器人半径
sx = 10.0  # [m]
sy = 10.0  # [m]
gx = 50.0  # [m]
gy = 50.0  # [m]
robot_size = 2.5  # [m]


# 初始化函数
def init_global(start_position_x, start_position_y, goal_position_x, goal_position_y, obstacle, size_robot=2.5,
                x_range=100,
                y_range=100, n_sample=200,
                max_edge_len=30, show_animat=True, n_knn=30):
    global ox, oy, N_SAMPLE, N_KNN, MAX_EDGE_LEN, MAP_MAX_LENGTH, MAP_MAX_WIDTH
    global show_animation, sx, sy, gx, gy, robot_size
    N_SAMPLE = n_sample
    N_KNN = n_knn
    MAX_EDGE_LEN = max_edge_len
    MAP_MAX_LENGTH = x_range
    MAP_MAX_WIDTH = y_range
    show_animation = show_animat
    # 设置开始位置
    sx = start_position_x
    sy = start_position_y
    # 设置目标位置
    gx = goal_position_x
    gy = goal_position_y
    # 设置机器人半径
    robot_size = size_robot

    # 生成地图边界，相对于中心坐标
    # 生成地图上边界
    for i in range(-MAP_MAX_LENGTH // 2, MAP_MAX_LENGTH // 2):
        ox.append(i)
        oy.append(MAP_MAX_WIDTH / 2.0)

    # 生成地图右边界
    for i in range(-MAP_MAX_WIDTH // 2, MAP_MAX_WIDTH // 2):
        ox.append(MAP_MAX_LENGTH / 2.0)
        oy.append(i)

    # 生成地图下边界
    for i in range(-MAP_MAX_LENGTH // 2, MAP_MAX_LENGTH // 2):
        ox.append(i)
        oy.append(-MAP_MAX_WIDTH / 2.0)

    # 生成地图左边界
    for i in range(-MAP_MAX_WIDTH // 2, MAP_MAX_WIDTH // 2):
        ox.append(-MAP_MAX_LENGTH / 2.0)
        oy.append(i)

    # 生成障碍物及其半径范围内的点
    for i in obstacle:
        obs_radius = 3
        obs_x = i[0]
        obs_y = i[1]
        for angle in range(0, 360, 10):  # 以10度为间隔生成点
            # 将角度转换为弧度
            angle_rad = math.radians(angle)
            # 计算点的坐标
            point_x = obs_x + obs_radius * math.cos(angle_rad)
            point_y = obs_y + obs_radius * math.sin(angle_rad)

            # 将生成的点添加到障碍物周围
            ox.append(point_x)
            oy.append(point_y)


"""
kd-tree用于快速查找nearest-neighbor

query(self, x[, k, eps, p, distance_upper_bound]): 查询kd-tree附近的邻居


"""


class Node:
    """
    Node class for dijkstra search
    点类
    """

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost  # 每条边权值总和，或者说到达该点所的代价
        self.parent_index = parent_index  # 上一个点是啥，也就是从哪个点到达该点的

    # 用来打印/显示该点的信息（坐标、代价、连接的上一个点）
    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + \
               str(self.cost) + "," + str(self.parent_index)


def prm_planning(start_x, start_y, goal_x, goal_y, obstacle_x_list, obstacle_y_list, robot_radius, *, camara=None,
                 rng=None):
    """
    Run probabilistic road map planning

    :param start_x: start x position
    :param start_y: start y position
    (start_x,start_y)表示开始的坐标
    :param goal_x: goal x position
    :param goal_y: goal y position
    (goal_x,goal_y)表示目标的坐标
    :param obstacle_x_list: obstacle x positions
    :param obstacle_y_list: obstacle y positions
    (obstacle_x_list[n],obstacle_y_list[n])表示第n+1个障碍物的坐标
    :param robot_radius: robot radius   机器人的半径
    :param rng: 随机数构造器
    :return:
    """
    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)
    # 采样点集生成
    sample_x, sample_y = sample_points(start_x, start_y, goal_x, goal_y,
                                       robot_radius,
                                       obstacle_x_list, obstacle_y_list,
                                       obstacle_kd_tree, rng)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    # 生成概率路图
    road_map = generate_road_map(sample_x, sample_y, robot_radius, obstacle_kd_tree)
    # 使用迪杰斯特拉规划路径
    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y, camara)

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    """判断是否发生碰撞,true碰撞，false不碰
        rr: 机器人半径
    """
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)  # 计算角度
    d = math.hypot(dx, dy)  # 计算距离

    if d >= MAX_EDGE_LEN:  # 如果大于规定的边的长度，那么认为其碰撞，也就是不在这两点连边(实际上MAX_EDGE_LEN是构建路图的邻域的半径)
        return True

    D = rr
    n_step = round(d / D)
    # 判断走在这两个点之间的边会不会发生碰撞
    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])  # 查询kd-tree附近的邻居
        if dist <= rr:
            return True  # collision
        # 每次移动D实际不能确保100%筛选出不会碰撞边？
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    """
    概率路图生成

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    road_map = []
    n_sample = len(sample_x)
    # vstack()会将(N,)的数组转化为(1,N)的数组，再加上.T(转置),这里就是将sample_x和sample_y转化为以[sample_x,sample_y]为元素的数组
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)
    # vstack会把
    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
        # (i,ix,iy)中i是采样点的序号，ix和iy分别是采样点的x和y坐标值
        # 对采样点击V中的每个点q，选择k个邻域点，这里k=n_sample是遍历所有的点？实际是按距离排序返回点集
        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []
        # indexes返回的k个最邻近采样点的序号，由于第一个肯定是自己，for从1开始
        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]
            # 对每个领域点$q'$进行判断，如果$q$和$q'$尚未形成路径，则将其连接形成路径并进行碰撞检测，若无碰撞，则保留该路径。
            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:  # 如果邻域点够了，不停止加入
                break

        road_map.append(edge_id)
    # 下面的函数用于画出路图的所有边
    # plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y, camara):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    goal_x: goal x position [m]
    goal_y: goal y position [m]
    obstacle_x_list: x position list of Obstacles [m]
    obstacle_y_list: y position list of Obstacles [m]
    robot_radius: robot radius [m]
    road_map: 构建好的路图 [m]
    sample_x: 采样点集x [m]
    sample_y: 采样点集y [m]

    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """
    # 初始化开始点与目标点
    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)
    # 使用字典的方式构造开闭集合
    # openList表由待考察的节点(也就是与已经考察的节点关联但未考察的节点)组成， closeList表由已经考察过的节点组成。
    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    path_found = True
    num_sum = 0
    # 步骤与A星算法一致
    while True:
        # 如果open_set是空的
        if not open_set:
            print("Cannot find path")
            path_found = False
            break
        # 从未考察的节点中选取到达花费最小点加入，其中key参数为排序关键字
        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph 画图，每考察两个点画1个X？已考察的点的个数是偶数时画点
        if show_animation and len(closed_set.keys()) % 5 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            # 在当前考察的点画一个绿色（green）的X
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)
            # 画出当前的路径图
            plot_shortest_road(current, closed_set, camara, sample_x, sample_y, num_sum)
            if camara != None:
                """Capture current state of the figure."""
                camara.snap()
        # 如果c_id是目标点那么说明已经找到了路径！
        # 目标点在road_map的最后一个road_map[len-1]，开始点在road_map的倒数第二road_map[len-2]
        # 这样的位置缘于生成采样点sample_point和生成路图generate_road_map
        if c_id == (len(road_map) - 1):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        # 以下是更新路径的关键部分，遍历与c_id关联的所有边/点
        for i in range(len(road_map[c_id])):
            num_sum = num_sum + 1
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)  # 计算从current点（c_id点）到n_id点的距离，记为d
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            # 如果n_id点在已经考察过的点里，那么这条边（c_id,n_id)也一定考察过,跳过
            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                # 如果当前记录的到达n_id的代价 大于 经current点到n_id的代价，那么更新！
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            # 如果不在open_set，说明这个点与之前所有已经考察的节点都不关联！所以无须比较，直接更新
            else:
                open_set[n_id] = node

    # 如果找不到路径，返回两个空集
    if path_found is False:
        return [], []

    # generate final course
    # 根据close_set的信息生成路径，从目标点‘回溯’
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


def plot_shortest_road(current, closed_set, camara, sample_x, sample_y, num_sum):
    plt.text(50, 5, f'更新次数：{num_sum}', fontsize=12)
    # 画出地图
    if show_animation:
        plt.plot(ox, oy, ".k")  # 显示障碍物，用黑点表示
        plt.plot(sx, sy, "^r")  # 显示开始位置，用红三角表示
        plt.plot(gx, gy, "^c")  # 显示目标地点，用蓝三角表示
    # 画出当前已经考察过的点
    for i in closed_set:
        plt.plot(sample_x[i], sample_y[i], '.b')
    rx, ry = [current.x], [current.y]
    parent_index = current.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index
    if show_animation:
        plt.plot(rx, ry, "-r")  # 给路径点连线
        plt.pause(0.001)


def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    # 采样点集生成

    # 这里其实是确定采样的区域（矩形）大小
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    # 如果没有传入随机生成器，就采用默认的（PCG64） 位生成器构造的随机生成器
    if rng is None:
        rng = np.random.default_rng()
    # 随机采样，直到采样点的数量足够
    while len(sample_x) <= N_SAMPLE:
        '''rng.random()返回[0.0,1.0)的随机浮点数
        所以下面两条语句可以生成在区域内的随机坐标
        '''
        tx = (rng.random() * (max_x - min_x)) + min_x
        ty = (rng.random() * (max_y - min_y)) + min_y

        # 查询离随机点(tx, ty)最近的障碍点的距离
        dist, index = obstacle_kd_tree.query([tx, ty])

        '''
        如果机器人与最近的障碍点的距离大于半径，说明没有碰撞.
        则将这个无碰撞的点加入采样点集中。
        '''
        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)
    # 别忘了起点和目标点也要放入采样点中！
    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")


def start(rng=None):
    print(" start!!")
    fig = plt.figure(1)

    camara = Camera(fig)  # 保存动图时使用
    # camara = None

    if show_animation:
        plt.plot(ox, oy, ".k")  # 显示障碍物，用黑点表示
        plt.plot(sx, sy, "^r")  # 显示开始位置，用红三角表示
        plt.plot(gx, gy, "^c")  # 显示目标地点，用蓝三角表示
        plt.grid(True)
        plt.axis("equal")
        if camara is not None:
            camara.snap()
    # 运行prm路径规划算法，得到路径点集
    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_size, camara=camara, rng=rng)
    # 如果rx为空，说明找不到路径
    # assert rx, 'Cannot found path'
    return rx, ry
    if show_animation:
        plt.plot(rx, ry, "-r")  # 给路径点连线
        plt.pause(0.001)
        if camara is not None:  # 保存动图
            camara.snap()
            animation = camara.animate()
            animation.save('trajectory.gif', fps=10)
    plt.savefig("result.png")  # 保存图片
    plt.show()


if __name__ == '__main__':
    start()
