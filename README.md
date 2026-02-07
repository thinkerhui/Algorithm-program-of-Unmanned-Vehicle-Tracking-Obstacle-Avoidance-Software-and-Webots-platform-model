# Algorithm-program-of-Unmanned-Vehicle-Tracking-Obstacle-Avoidance-Software-and-Webots-platform-model
## 项目介绍与团队成员
这个仓库存放的是SSE,SYSU的《基于多模态数据融合的无人车自主循迹避障软件》大创项目在研究过程过研究和实现的5个算法（A*、PRM、DWA、APF和MPC）的程序以及它们在Webots平台上对应的控制器搭载适配。这个仓库还包括和Webots相关的世界（World），World里面有大创小组自主搭建的麦克纳姆轮小车模型。

本项目的目的是设计并实现一种智能无人车自主循迹和避障的程序，通过深入研究无人车从环境识别、路径规划到与运动控制的算法，并结合各种算法的具体特点和适用条件进行多模态数据融合的输入，提高无人车循迹和避障算法的性能和鲁棒性。

本项目的研究和探索将为基于多模态数据融合的循迹避障算法贡献理论观点、模型平台和实验数据，推动无人车在无人驾驶、物流运输等领域的应用落地和优化。

项目指导老师：陈建国    组长：吴仰晖    组员：李健文、梁竞冀、孙惠祥、张凯茗

## 仓库说明

### 目录构成

```
project/
├── algorithms/                    # 算法模块
│   ├── path_planning/             # 路径规划算法
│   │   ├── a_star.py              # A*算法
│   │   └── prm.py                 # PRM算法
│   │
│   ├── local_planning/           # 局部规划算法
│   │   ├── dwa.py                 # DWA算法
│   │   ├── apf.py                 # 人工势场算法
│   │   └── mpc.py                 # MPC算法
│   │
│   └── trajectory_tracking/      # 轨迹跟踪控制器
│       └── pid_controller.py      # PID控制器
│
├── controllers/                   # Webots控制器
│   ├── base/                      # 基础控制器类
│   │   ├── supervisor_controller.py  # 基础Supervisor控制器
│   │   └── vehicle_controller.py     # 车辆通用控制接口
│   │
│   ├── examples/                 # 示例控制器
│   │   ├── a_star_controller.py
│   │   ├── prm_controller.py
│   │   ├── dwa_controller.py
│   │   ├── atf_controller.py
│   │   └── atf_with_pid_controller.py
│   │
│   └── hybrid/                   # 混合控制器
│       └── prm_pid_controller.py
│
├── utils/                         # 工具函数
│   ├── math_utils.py              # 数学工具
│   ├── geometry.py                # 几何计算
│   └── point.py                   # 点类
│
└── worlds/                        # Webots世界文件
    ├── empty.wbt
    ├── trace.wbt
    └── world_new.wbt

docs/
├── algorithms/                    # 算法文档
├── usage/                        # 使用指南
│   └── QUICK_START.md            # 快速开始指南
└── archive/                       # 归档文件
```

### 开始/运行

1. **安装Webots**: 下载并安装 [Webots](https://www.cyberbotics.com/download)
2. **打开项目**: 使用VSCode打开本仓库
3. **运行仿真**: 
   - 打开Webots
   - 选择世界文件 (`project/worlds/empty.wbt`)
   - 设置机器人控制器
   - 点击运行

详细启动指南请参考 [docs/usage/QUICK_START.md](docs/usage/QUICK_START.md)

## 算法运行动态图展示

PRM算法：
![PRMPLT](https://sse-market-source-1320172928.cos.ap-guangzhou.myqcloud.com/blog/PRMPLT.gif)

实际上这里展现的是图搜索的过程，路径运行和下面的图差不多。
除了PRM，实际上项目在执行过程中还探索了针对狭窄通道问题的OBPRM算法：

![obprm](https://sse-market-source-1320172928.cos.ap-guangzhou.myqcloud.com/blog/obprm.gif)

A*算法动图：
![ASTARPLT](https://sse-market-source-1320172928.cos.ap-guangzhou.myqcloud.com/blog/ASTARPLT.gif)

APF算法动图：

![AstarPLT](https://sse-market-source-1320172928.cos.ap-guangzhou.myqcloud.com/blog/AstarPLT.gif)

DWA算法动图：
![DWAPLT](https://sse-market-source-1320172928.cos.ap-guangzhou.myqcloud.com/blog/DWAPLT.gif)

## Webots仿真展示

PRM算法Webots仿真录像：

![PRM](https://sse-market-source-1320172928.cos.ap-guangzhou.myqcloud.com/blog/PRM.gif)

APF算法Webots仿真录像：

![APF](https://sse-market-source-1320172928.cos.ap-guangzhou.myqcloud.com/blog/APF.gif)

DWA算法Webots仿真录像：

![DWA](https://sse-market-source-1320172928.cos.ap-guangzhou.myqcloud.com/blog/DWA.gif)
