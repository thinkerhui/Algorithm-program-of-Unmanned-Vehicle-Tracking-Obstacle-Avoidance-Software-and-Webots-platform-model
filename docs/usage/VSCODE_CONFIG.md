# VSCode Pylance/Webots 类型配置文档

本文档说明了项目中 VSCode Pylance 类型检查的配置方案，用于解决 Webots 控制器模块的导入类型提示问题。

## 问题背景

### Pylance 导入解析问题

当使用 VSCode 的 Pylance 扩展进行 Python 类型检查时，Webots 控制器模块存在以下问题：

1. **模块名称冲突**：`controller` 是 Python 的内置模块名（用于命令行参数处理），与 Webots 的 `controller` 包冲突
2. **动态生成的模块**：Webots 的 `controller` 模块由 SWIG 自动生成，不包含类型注解
3. **运行时可用**：Webots 控制器模块仅在 Webots 仿真环境中可用，普通 Python 环境中不存在
4. **类型检查缺失**：没有类型信息，Pylance 无法提供代码补全、参数类型检查等功能

### 常见错误表现

```
[{
    "resource": "/d:/project/path/controllers/examples/dwa_controller.py",
    "owner": "Generated",
    "code": {
        "value": "import-resolution",
        "target": {
            "$mid": 1,
            "external": "https://github.com/microsoft/pyright/blob/main/docs/configuration.md",
            "path": "/microsoft/pyright/blob/main/docs/configuration.md",
            "scheme": "https"
        }
    },
    "message": "Import "controller" could not be resolved",
    "severity": 4,
    "messagePlain": "Import "controller" could not be resolved",
    "startLineNumber": 1,
    "startColumn": 1,
    "endLineNumber": 1,
    "endColumn": 28
}]
```

## 解决方案：类型存根文件 (Type Stubs)

### 解决方案概述

通过创建 Python 类型存根文件（`.pyi`），为 Pylance 提供模块的类型信息，而不依赖实际导入模块。

### 类型存根文件结构

```
project/
└── controllers/
    ├── controller.pyi          # Webots 控制器类型存根
    ├── base/
    │   ├── supervisor_controller.pyi
    │   └── vehicle_controller.pyi
    └── examples/
        └── dwa_controller.pyi
```

### 核心存根文件：controller.pyi

[`controller.pyi`](../../project/controllers/controller.pyi) 文件包含：

- **基础类型定义**：`Device`、`Robot`、`Supervisor`、`Motor` 等核心类
- **传感器类型**：`Camera`、`Lidar`、`GPS`、`Accelerometer`、`Gyro` 等
- **常量定义**：节点类型、设备类型等枚举值
- **类型别名**：`Vec2f`、`Vec3f`、`Vec4f` 等常用类型

关键代码结构示例：

```python
# 类型别名
Vec3f = Tuple[float, float, float]
Rotation = Tuple[float, float, float, float]

# 设备基类
class Device:
    def getName(self) -> str: ...
    def getModel(self) -> str: ...
    def getNodeType(self) -> int: ...

# 电机类
class Motor(Device):
    def setPosition(self, position: float) -> None: ...
    def setVelocity(self, velocity: float) -> None: ...
    def getPosition(self) -> float: ...
```

## VSCode 配置

### settings.json 配置

项目根目录的 [`.vscode/settings.json`](../../.vscode/settings.json) 包含以下关键配置：

```json
{
    // Python 解释器路径
    "python.defaultInterpreterPath": "C:/Users/Administrator/AppData/Local/Programs/Python/Python38/python.exe",
    
    // 自动补全额外路径
    "python.autoComplete.extraPaths": [
        "${workspaceFolder}/project/controllers/all_controller",
        "${workspaceFolder}/project/controllers"
    ],
    
    // Pylance 额外分析路径
    "python.analysis.extraPaths": [
        "${workspaceFolder}/project/controllers/all_controller",
        "${workspaceFolder}/project/controllers"
    ],
    
    // 类型存根搜索路径（关键配置）
    "python.analysis.stubPath": "${workspaceFolder}/project/controllers",
    
    // Python 环境变量
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}/project/controllers/all_controller"
    }
}
```

### 配置说明

| 配置项 | 说明 |
|--------|------|
| `python.analysis.stubPath` | 指定 Pylance 搜索 `.pyi` 存根文件的目录 |
| `python.analysis.extraPaths` | 添加到 Python 模块搜索路径，供 Pylance 分析使用 |
| `python.autoComplete.extraPaths` | 添加到自动补全的搜索路径 |

### 工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                    VSCode + Pylance                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. 导入解析                                                 │
│      from controller import Supervisor                      │
│                ↓                                             │
│   2. 搜索 stubPath 目录                                      │
│      ${workspaceFolder}/project/controllers/controller.pyi │
│                ↓                                             │
│   3. 加载类型信息                                            │
│      读取 controller.pyi 中的类型定义                        │
│                ↓                                             │
│   4. 类型检查和代码补全                                      │
│      提供参数类型提示、方法签名、属性建议等                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 替代解决方案

### 方案一：使用 Webots 安装目录的 Python 库

如果 Webots 已正确安装，可以使用其内置的 Python 库路径：

1. **查找 Webots Python 库路径**：
   ```python
   # Windows 默认路径
   C:/Program Files/Webots/lib/python38/site-packages/
   
   # 或通过 Webots 查找
   import webots
   print(webots.__file__)
   ```

2. **配置 VSCode**：
   ```json
   {
       "python.analysis.extraPaths": [
           "C:/Program Files/Webots/lib/python38/site-packages/",
           "${workspaceFolder}/project/controllers"
       ]
   }
   ```

3. **优点**：
   - 自动获取最新类型的存根
   - 与 Webots 版本匹配

4. **缺点**：
   - 路径依赖 Webots 安装位置
   - 团队成员可能使用不同安装路径

### 方案二：使用环境变量

1. **设置 PYTHONPATH**：
   ```bash
   # Windows PowerShell
   $env:PYTHONPATH = "C:\Program Files\Webots\lib\python38\site-packages;D:\project\path\controllers"
   
   # 或在 .env 文件中
   PYTHONPATH=C:\Program Files\Webots\lib\python38\site-packages
   ```

2. **VSCode 配置**：
   ```json
   {
       "python.envFile": "${workspaceFolder}/.env"
   }
   ```

### 方案三：py.typed 标记（高级）

如果需要为自己的模块提供类型信息：

1. **创建 `py.typed` 文件**：
   ```
   project/
   └── my_module/
       ├── __init__.py
       └── py.typed      # 标记此包支持类型检查
   ```

2. **配置 `setup.py`**：
   ```python
   setup(
       name="my_module",
       packages=["my_module"],
       package_data={"my_module": ["py.typed"]},
   )
   ```

## 使用示例

### 导入 Webots 控制器

```python
from controller import Supervisor, Robot, Motor, Camera

class MyController(Supervisor):
    def __init__(self):
        super().__init__()
        self.robot = self.getFromDef("MY_ROBOT")
        self.camera = self.getCamera("camera")
        self.motor = self.getMotor("motor")
    
    def run(self):
        # 类型检查正常工作
        position = self.motor.getPosition()  # 返回 float
        self.motor.setVelocity(1.0)           # 接受 float 参数
        
        # 传感器数据
        image = self.camera.getImage()       # 返回 bytes
```

### 获取类型信息

在 VSCode 中：
1. 悬停在 `Supervisor` 上查看类型提示
2. 按 `Ctrl+Click` 跳转到类型定义
3. 使用 `Ctrl+Shift+O` 查看类成员列表

## 维护指南

### 更新类型存根

当需要添加新类型或修改现有类型时：

1. **编辑 `controller.pyi`**：
   ```python
   class NewDevice(Device):
       """新增设备类型。"""
       def newMethod(self, param: int) -> str: ...
   ```

2. **验证配置**：
   ```bash
   # 检查 Pylance 是否识别新类型
   code --reload
   ```

### 添加新的控制器类型存根

为新创建的控制器添加 `.pyi` 文件：

```python
# project/controllers/my_controller.pyi
from controller import Supervisor

class MyController(Supervisor):
    """我的自定义控制器。"""
    
    def __init__(self) -> None: ...
    
    def customMethod(self) -> None: ...
```

## 常见问题

### Q1: 仍然显示 "Import could not be resolved"

检查以下配置：
1. `python.analysis.stubPath` 是否正确指向 `project/controllers`
2. `controller.pyi` 文件是否存在于该目录
3. VSCode 是否重新加载（`Ctrl+Shift+P` → "Reload Window"）

### Q2: 类型提示不完整

确保：
1. 类和方法使用 `->` 返回类型注解
2. 正确使用类型别名（`Vec3f` 等）
3. 复杂类型使用 `Union`、`Optional` 等

### Q3: 如何处理第三方库的类型？

1. 安装带类型的版本：`pip install pandas-stubs`
2. 或使用 `pip install types-pkg` 这样的存根包

## 参考资源

- [Pylance 官方文档](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [PEP 561 - Type Stubs](https://www.python.org/dev/peps/pep-0561/)
- [Webots Python API](https://www.cyberbotics.com/doc/guide/python-api)
