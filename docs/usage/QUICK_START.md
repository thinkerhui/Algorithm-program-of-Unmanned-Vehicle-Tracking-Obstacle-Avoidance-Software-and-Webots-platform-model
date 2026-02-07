# Webots平台无人车循迹避障算法仿真

## 1. 环境配置

### 1.1 安装依赖

```bash
pip install numpy scipy matplotlib osqp
```

### 1.2 Webots配置

确保Webots已正确安装。调试需要配置以下内容：

#### Python解释器路径
- **推荐路径**：`C:/Users/Administrator/AppData/Local/Programs/Python/Python38/python.exe`
- 确保Python版本为3.8.x（与Webots controller库兼容）

#### VSCode配置
项目已配置好 `.vscode/settings.json` 和 `.vscode/launch.json`：
- `python.defaultInterpreterPath`：设置默认Python解释器
- 调试配置包含A*、DWA、PRM、APF控制器的调试入口

## 2. 项目结构

```
project/
├── algorithms/                    # 纯算法实现（不依赖Webots）
│   ├── local_planning/           # 局部规划算法
│   │   ├── apf.py                # 人工势场算法
│   │   ├── dwa.py                # 动态窗口法
│   │   └── mpc.py                # 模型预测控制
│   ├── path_planning/            # 全局路径规划算法
│   │   ├── a_star.py             # A*算法
│   │   └── prm.py                # PRM概率路图
│   └── trajectory_tracking/       # 轨迹跟踪算法
│       └── pid_controller.py      # PID控制器
├── controllers/                   # Webots控制器（依赖controller模块）
│   ├── base/                     # 基础控制器类
│   │   ├── supervisor_controller.py
│   │   └── vehicle_controller.py
│   ├── examples/                  # 控制器示例
│   │   ├── a_star_controller.py
│   │   ├── atf_controller.py
│   │   └── dwa_controller.py
│   ├── hybrid/                    # 混合控制器
│   │   └── prm_pid_controller.py
│   └── all_controller/            # 完整控制器实现
├── utils/                         # 工具模块
│   ├── geometry.py                # 几何计算
│   ├── math_utils.py              # 数学工具
│   └── trajectory.py              # 轨迹处理
└── worlds/                        # Webots世界文件
    ├── empty.wbt                  # 空世界
    └── trace.wbt                  # 轨迹世界
```

> **注意**：`algorithms/` 目录下的文件是纯算法实现，可以独立运行测试；`controllers/` 目录下的文件是Webots控制器，需要在Webots环境中运行。

## 3. 运行指南

### 3.1 在Webots中运行

1. 打开Webots软件
2. 加载项目世界文件（`project/worlds/empty.wbt`或`project/worlds/trace.wbt`）
3. 在机器人节点中设置控制器为对应的Python文件（如`a_star_controller.py`）
4. 点击运行按钮启动仿真

### 3.2 VSCode调试配置

项目已配置好`.vscode/launch.json`，支持以下调试配置：

| 调试配置 | 说明 |
|----------|------|
| **A* 控制器调试** | 调试A*路径规划算法 |
| **DWA 控制器调试** | 调试DWA局部规划算法 |
| **PRM 控制器调试** | 调试PRM路径规划算法 |
| **APF 控制器调试** | 调试人工势场算法 |

#### 调试配置说明

每个调试配置包含以下关键设置：
- `pythonPath`：Python解释器路径
- `program`：被调试的控制器文件
- `cwd`：工作目录（`project/controllers/all_controller`）
- `env.PYTHONPATH`：Webots controller库路径
- `env.PATH`：包含Python和Webots的路径

#### 启动调试

1. 在VSCode左侧运行和调试面板中选择调试配置
2. 按下F5或点击绿色运行按钮启动调试
3. 调试时会启动Webots仿真窗口

### 3.3 查看仿真结果

仿真运行后，可以在Webots界面的`Views->Optional->Robot Console`中查看调试输出。

## 4. 算法说明

### 4.1 A*算法

- **算法文件**：`project/algorithms/path_planning/a_star.py`（纯算法实现）
- **Webots控制器**：`project/controllers/all_controller/a_star_controller.py`
- **功能**：基于启发式搜索的全局路径规划算法，可在已知地图中找到最优路径

### 4.2 PRM算法

- **算法文件**：`project/algorithms/path_planning/prm.py`（纯算法实现）
- **Webots控制器**：`project/controllers/all_controller/prm_controller.py`
- **功能**：概率路图算法，通过随机采样构建路图，适合高维空间和复杂环境

### 4.3 DWA算法

- **算法文件**：`project/algorithms/local_planning/dwa.py`（纯算法实现）
- **Webots控制器**：`project/controllers/all_controller/dwa_controller.py`
- **功能**：动态窗口法，考虑机器人运动约束，在速度空间中搜索最优速度

### 4.4 人工势场（ATF/APF）

- **算法文件**：`project/algorithms/local_planning/apf.py`（纯算法实现）
- **Webots控制器**：`project/controllers/all_controller/atf_controller.py`
- **功能**：将机器人运动建模为受力场，目标点产生引力，障碍物产生斥力

### 4.5 MPC控制

- **算法文件**：`project/algorithms/local_planning/mpc.py`
- **功能**：基于模型预测的控制方法，优化未来控制序列以跟踪参考轨迹

### 4.6 PID控制

- **算法文件**：`project/algorithms/trajectory_tracking/pid_controller.py`
- **功能**：经典的闭环控制算法，通过比例、积分、微分三个环节调节控制量

## 5. 常见问题

### 5.1 VSCode调试时提示"找不到Python解释器"

**错误信息**：`legacyGetInterpreterDetails: No executable found`

**解决方法**：
1. 检查 `.vscode/settings.json` 中的 `python.defaultInterpreterPath` 是否正确
2. 确保Python 3.8已安装在 `C:/Users/Administrator/AppData/Local/Programs/Python/Python38/`
3. 检查 Webots controller 路径是否添加到 `PYTHONPATH` 环境变量