# 算法模块

本目录包含纯算法实现，不依赖Webots的`controller`模块，可用于独立测试。

## 依赖安装

```bash
pip install numpy scipy matplotlib osqp
pip install celluloid  # 用于生成动画
pip install osqp  # 用于MPC控制器
```

## 算法说明

| 算法 | 文件 | 类型 | 说明 |
|------|------|------|------|
| A* | `path_planning/a_star.py` | 类 | 基于启发式搜索的全局路径规划 |
| PRM | `path_planning/prm.py` | 类+函数 | 概率路图算法 |
| DWA | `local_planning/dwa.py` | 类 | 动态窗口法 |
| APF | `local_planning/apf.py` | 函数 | 人工势场算法 |
| MPC | `local_planning/mpc.py` | 脚本 | 模型预测控制（依赖Webots） |
| PID | `trajectory_tracking/pid_controller.py` | 函数 | PID轨迹跟踪控制器 |

## 使用方式

### 导入使用（推荐）

```python
# 导入A*算法
from algorithms.path_planning.a_star import AStar, Env

# 导入DWA算法
from algorithms.local_planning.dwa import DWA, Config

# 导入APF算法
from algorithms.local_planning.apf import start as apf_start
```

### 独立测试

算法文件可直接作为脚本运行用于测试，但需要先设置参数：

```python
# a_star.py 测试示例
from algorithms.path_planning.a_star import AStar, Env

env = Env(x_range=(0, 50), y_range=(0, 50), obstacle=[(10, 10), (20, 20)])
a_star = AStar(s_start=(5, 5), s_goal=(45, 45), env=env)
path, visited = a_star.searching()
```

## 与Webots控制器的关系

- **`algorithms/`** - 纯算法实现，不依赖Webots
- **`controllers/`** - Webots控制器实现，依赖Webots `controller` 模块

控制器通过导入算法模块来使用这些算法：

```python
# controllers/all_controller/a_star_controller.py
from algorithms.path_planning.a_star import AStar, Env
```
