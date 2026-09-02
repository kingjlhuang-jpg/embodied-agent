# 具身智能技术栈全景

## 本项目技术栈

```
Python 3.9+ ─── PyBullet (仿真) ─── PyTorch/TD3+BC (AI) ─── Gymnasium (RL接口)
                    │                        │
                    │                        ├── trained_policy.zip (到达模型)
                    │                        ├── grasp_policy.zip (抓取模型)
                    │                        ├── robust_grasp_policy.zip (鲁棒抓取模型)
                    │                        ├── physical_grasp_policy.zip (纯摩擦抓取模型)
                    │                        └── pose_physical_grasp_policy.zip (随机姿态物理模型)
                    │
                    ├── Bullet 物理引擎 (碰撞/重力/摩擦/力矩)
                    ├── SDF 机器人模型 (Kuka iiwa + WSG50 夹爪)
                    └── OpenGL/Metal 3D 渲染 (GUI 模式)
```

## 生产级技术栈全景

### 仿真层

| 平台 | 物理精度 | 渲染质量 | GPU加速 | 适用场景 |
|------|---------|---------|--------|---------|
| **PyBullet** | 中 | 低 | 无 | 入门学习、快速原型 |
| **MuJoCo** (DeepMind) | 高 | 中 | 支持 | RL 研究、控制算法 |
| **Isaac Sim** (NVIDIA) | 极高 | 极高(RTX) | 原生 | 工业级仿真、合成数据、数字孪生 |
| **Gazebo** | 中高 | 中 | 有限 | ROS 2 开发、多机器人协作 |

### AI 模型层

| 模型 | 参数量 | 输入 | 输出 | 定位 |
|------|--------|------|------|------|
| **MLP 策略网络 / reach** | 19K | 关节角度+位置(13维) | 手臂关节增量(7维) | 到达基准 |
| **MLP 策略网络 / grasp** | 约74K | 关节+夹爪+方块+接触+抓取状态(24维) | 手臂7维+夹爪1维 | 接触抓取 |
| **MLP 策略网络 / robust grasp** | 约78K | grasp状态+最近2步动作(40维) | 手臂7维+夹爪1维 | 延迟/物理随机化抓取 |
| **MLP 策略网络 / physical grasp** | 约80K | grasp状态+接触阶段残差(48维) | 手臂7维+夹爪1维 | 无固定约束的摩擦抓取 |
| **MLP 策略网络 / pose physical** | 约80K | physical状态+相对yaw误差(50维) | 手臂7维+夹爪1维 | 扩大位置和±180°视觉姿态 |
| **EEGNet / CNN** | 100K-1M | RGB 图像 | 动作 | 视觉感知 |
| **RT-2** (Google) | 55B | 图像+语言 | 动作token | VLA 先驱 |
| **OpenVLA** (Stanford) | 7B | 图像+语言 | 动作 | 开源 VLA |
| **pi0** (Physical Intelligence) | 未公开 | 图像+语言 | 连续动作 | 当前最强 VLA |
| **GR00T N2** (NVIDIA) | 未公开 | 图像+语言 | 动作 | 人形机器人专用 |

### 中间件层

| 组件 | 作用 | 备注 |
|------|------|------|
| **ROS 2** | 节点通信 (Topic/Service/Action) | 机器人软件的 "Android" |
| **ros2_control** | 硬件抽象 + 实时电机控制 (1kHz) | C++ 实现 |
| **MoveIt 2** | 运动规划 (无碰撞轨迹) | 替代手写逆运动学 |
| **Nav2** | 自主导航 (路径规划+避障) | 移动机器人必备 |

### 硬件层

| 组件 | 入门 | 科研级 | 工业级 |
|------|------|--------|-------|
| **机械臂** | myCobot 280 (¥3K) | Franka Panda (¥20万) | KUKA/UR (¥30万+) |
| **主控** | Jetson Orin Nano (¥3K) | Jetson AGX Orin (¥8K) | 工控机 |
| **相机** | USB 摄像头 | RealSense D435 (¥2K) | 工业相机 |
| **电机驱动** | Arduino/STM32 | 内置驱动 | EtherCAT 伺服 |

## 从仿真到真机：代码改造清单

| 仿真代码 (PyBullet) | 真机代码 (ROS 2) | 说明 |
|---------------------|-----------------|------|
| `p.connect(p.DIRECT)` | `rclpy.init()` | 初始化方式不同 |
| `p.getLinkState(robot, 6)` | 订阅 `/joint_states` | 数据来源：仿真器 → 编码器 |
| `p.setJointMotorControl2()` | 发布 `/arm/commands` | 指令目标：仿真器 → CAN总线 |
| `p.stepSimulation()` | 删除 | 真实世界自动运行 |
| `env.reset()` | 人工复位 | 真实世界没有重置按钮 |
| `trained_policy.zip` / `grasp_policy.zip` / `robust_grasp_policy.zip` / `physical_grasp_policy.zip` / `pose_physical_grasp_policy.zip` | **直接复用** | 观测和动作接口一致时模型不变；鲁棒抓取需维护2步动作历史 |

## 关键概念索引

| 概念 | Demo | 说明 |
|------|------|------|
| **URDF** | Demo 1 | XML 格式描述机器人的关节、质量、形状 |
| **逆运动学 (IK)** | Demo 1 | 给定目标位置 → 计算关节角度 |
| **位置控制** | Demo 1 | 设置电机目标角度，PID 自动跟踪 |
| **Gymnasium 环境** | Demo 2 | 标准的 obs/action/reward/done 接口 |
| **策略网络** | Demo 2 | 神经网络：观测 → 动作 |
| **奖励设计** | Demo 2 | 靠近 < 双侧接触 < 抬升 < 稳定保持 |
| **IK 引导训练** | Demo 2 | 阶段平衡行为克隆 + DAgger 加速冷启动 |
| **TD3+BC** | Demo 2 | 经验回放 + Q-filter IK 约束的连续动作强化学习 |
| **域随机化** | Demo 2 | 随机尺寸/质量/摩擦/电机/噪声/延迟，1000回合盲测 |
| **保守残差微调** | Demo 2 | 冻结24维基线，只学习新增的动作历史或接触阶段输入 |
| **纯物理抓取** | Demo 2 | 200g 方块 + 有限夹紧力/摩擦，固定约束始终关闭 |
| **模型部署** | Demo 3 | 加载 Stable-Baselines3 模型运行 |
| **Sim-to-Real** | Demo 3 | 仿真训练 → 真机运行，模型不变 |
