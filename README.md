# embodied-agent

<p align="center">
  <img src="assets/banner.png" width="700" alt="Embodied AI"/>
</p>

<p align="center">
  <b>具身智能学习与仿真 Demo</b> — 没有真实机器人也能学机器人 AI
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> &middot;
  <a href="#demo-展示">Demo 展示</a> &middot;
  <a href="#宇树-g1-人形机器人">宇树 G1</a> &middot;
  <a href="#技术栈">技术栈</a> &middot;
  <a href="#学习路线">学习路线</a> &middot;
  <a href="docs/learning_guide.md"><b>学习指南</b></a>
</p>

---

通过 **PyBullet / MuJoCo 物理仿真器**，在你的电脑上运行完整的机器人「感知 → 决策 → 执行」控制流程。训练出的 AI 模型可直接部署到真实机器人上。

## 快速开始

```bash
git clone https://github.com/ava-agent/embodied-agent.git
cd embodied-agent
pip install -r requirements.txt

# 运行第一个 Demo
python demos/01_robot_arm_grasp.py

# Unitree Demo 另外需要官方模型仓库（与本仓库放在同一父目录即可）
git clone https://github.com/unitreerobotics/unitree_mujoco.git ../unitree_mujoco
```

## Demo 展示

### Demo 1: 机械臂到达（IK 基准）

用**逆运动学**控制 Kuka 7-DOF 机械臂分三步接近红色方块。这个 Demo
只验证末端到达，不包含可开合夹爪，作为后续 RL 抓取的 IK 基准。

| 初始状态 | 接近目标 | 到达抓取位置 |
|:---:|:---:|:---:|
| ![initial](assets/screenshot_initial.png) | ![reaching](assets/screenshot_reaching.png) | ![grasp](assets/screenshot_grasp.png) |
| 机械臂直立 | 弯向目标 | 误差 < 2cm |

```bash
python demos/01_robot_arm_grasp.py
```

### Demo 2: IK 引导的强化学习训练

先用逆运动学（IK）生成示范，通过行为克隆和 DAgger 完成冷启动，再用
TD3+BC 强化学习微调。IK 只在训练阶段担任老师，最终评估和部署仅使用 RL
策略。现在提供两个任务：`reach` 保留已经达到 97% 独立成功率的 7 维到达
基线；`grasp` 使用带 WSG50 夹爪的 Kuka、动态碰撞方块和 8 维动作完成
“接近 → 合拢 → 抬起 → 稳定保持”。

```python
# reach: 13维观测 → 7个手臂关节
# grasp: 24维观测 → 7个手臂关节 + 1个夹爪开合
# robust grasp: 再加入最近2步动作历史，共40维观测
# physical grasp: 增加接触阶段残差状态，共48维观测
action = policy_network(obs)         # 决策（AI 推理）
env.step(action)                     # 执行并计算接触/抬升奖励
```

```bash
python demos/02_rl_training.py --task reach

# 真正抓取：输出 grasp_policy.zip 和 grasp_training_metrics.json
python demos/02_rl_training.py --task grasp

# 快速验证完整流水线（不用于判断最终收敛）
python demos/02_rl_training.py --task grasp --quick

# 域随机化鲁棒训练（最终自动运行1000回合盲测）
python demos/02_rl_training.py --task grasp --robust

# 快速验证鲁棒训练流水线
python demos/02_rl_training.py --task grasp --robust --quick

# 纯物理摩擦抓取：不创建固定约束
python demos/02_rl_training.py --task grasp --physical

# 快速验证物理抓取流水线
python demos/02_rl_training.py --task grasp --physical --quick

# 扩大位置并随机旋转方块的纯物理任务
python demos/02_rl_training.py --task grasp --physical --random-pose
```

抓取奖励按任务阶段递增：靠近方块获得小奖励，左右手指同时接触获得中等奖励，
方块离开台面获得大奖励，持续抬高 3.5 cm 才判定成功。基线与鲁棒模式为避免
轻量级 PyBullet 夹爪在接触后数值打滑，只在检测到双侧真实碰撞后建立有限力
抓取约束；`--physical` 模式则完全禁用该约束，只依靠夹紧力和摩擦。

训练会平衡采样“接近 / 对准闭合 / 抬起保持”三个阶段，提高夹爪动作的行为
克隆权重，并用 Q-filter 只在 IK 动作优于当前策略时施加教师约束。固定种子的
本地参考训练在最终 100 回合独立测试中达到 98%，随后三个互不重叠的 100
回合验证集分别达到 98%、99%、99%；评估阶段均不调用 IK。

`--robust` 会在每个回合随机化方块尺寸（4–6 cm）、质量（20–80 g）、
摩擦、朝向、夹爪摩擦、电机力/速度，并加入观测噪声、动作噪声和 0–2 步
控制延迟。鲁棒策略可利用最近两步动作历史学习延迟补偿；原 24 维策略会无损扩展为
40 维网络，旧通道被冻结，只训练新增的历史残差权重。训练使用最佳检查点保护，
未超过基线的候选不会覆盖模型。固定种子的 1000 回合全随机化盲测结果为
938/1000（93.8%），其中 0/1/2 步延迟成功率分别为 95.8%/93.9%/91.7%。

`--physical` 完全禁止 `JOINT_FIXED` 抓取约束，使用 200 g 方块、0.55 方块
摩擦系数和 60 N 有限夹爪电机力。抓取状态必须由持续双指接触建立，接触丢失
会产生真实滑落并判定失败。正中 IK 夹持参考成功率为 58/60；故意沿夹爪轴
偏移 4 cm 的边缘夹持为 0/20。最终纯 RL 在 1000 回合独立测试中达到
858/1000（85.8%），且固定约束激活次数为 0。模型单独保存为
`physical_grasp_policy.zip`，不会覆盖辅助抓取模型。本轮在线候选没有超过
75 回合保护集，因此最终保留的是无损扩展后的最佳基线，而不是较差的新权重。

`--random-pose` 将纯物理任务扩大到 `x=0.31～0.49 m`、`y=-0.11～0.11 m`
和完整 `yaw=-180°～180°`。策略从回合开始观测具有方块 90° 对称性的
夹爪—方块相对偏航误差；训练教师选择最近的等价夹持方向，并在误差小于 8°
前保持夹爪张开。该任务使用独立的 `pose_physical_grasp_policy.zip`；1000 回合
全范围 RL-only 测试由修复前 459/1000（45.9%）提升到 507/1000（50.7%），
固定约束激活次数仍为 0。较低成功率主要来自 `x>0.43 m` 的远端可达性和
接触后真实滑落，不会覆盖固定姿态下 85.8% 的物理模型。

### Demo 3: 模型部署

加载训练好的模型，在仿真中运行，展示仿真 vs 真机的代码对比：

```python
# 仿真                                  # 真机
obs = env.step(action)                   obs = ros_node.get_obs()
action = policy(obs)  # ← 完全一样 →    action = policy(obs)
env.step(action)                         ros_node.send_cmd(action)
```

```bash
python demos/03_deploy_model.py --task reach
python demos/03_deploy_model.py --task grasp
python demos/03_deploy_model.py --task grasp --robust
python demos/03_deploy_model.py --task grasp --physical
python demos/03_deploy_model.py --task grasp --physical --random-pose
python demos/03_deploy_model.py --task grasp --physical --random-pose --episodes 10
```

### Demo 4: 宇树 G1 人形机器人动作

用 **PD 力矩控制** 让 G1（29个驱动器）完成一系列全身动作。

| 站立 | 挥手打招呼 | 左拳出击 |
|:---:|:---:|:---:|
| ![stand](assets/g1_act_stand.png) | ![wave](assets/g1_act_wave.png) | ![punch](assets/g1_act_punch_l.png) |
| 双腿微弯，PD 控制稳定 | 右臂抬起挥动 | 转腰出拳，重心转移 |

动作序列：站立 → 挥手 → 鞠躬 → 出拳 → 金鸡独立 → 太极起势

```bash
# 需要 conda 环境（含 MuJoCo + 宇树 SDK）
conda activate unitree
mjpython demos/04_g1_actions.py --unitree-mujoco-root ../unitree_mujoco

# 不打开窗口，只验证模型路径和 29 个驱动器
python demos/04_g1_actions.py --unitree-mujoco-root ../unitree_mujoco --validate-only
```

> **为什么有些动作会摔倒？** 出拳和抬腿等剧烈动作用关键帧控制时容易失去平衡。这正是**强化学习的价值**——RL 能学到动态平衡策略，在做动作的同时实时调整全身关节补偿重心偏移。

---

## 宇树 G1 人形机器人

<p align="center">
  <img src="assets/sim_to_real.png" width="500" alt="Sim to Real"/>
</p>

本项目集成了宇树官方 MuJoCo 仿真，支持 Go2 / G1 / H1 全系列。

官方模型不复制进本仓库。脚本按以下优先级查找 `unitree_mujoco`：命令行
`--unitree-mujoco-root`、环境变量 `UNITREE_MUJOCO_ROOT`、与本仓库同级的
`unitree_mujoco/`。也可以用通用查看器加载任一支持型号：

```bash
python demos/05_unitree_viewer.py g1 --unitree-mujoco-root ../unitree_mujoco
python demos/05_unitree_viewer.py go2 --unitree-mujoco-root ../unitree_mujoco --validate-only
```

| Go2 四足 | G1 人形 | H1 人形 |
|:---:|:---:|:---:|
| ![go2](assets/screenshot_go2_sim.png) | ![g1](assets/screenshot_g1_sim.png) | ![h1](assets/screenshot_h1_sim.png) |
| 12 驱动器 | 29 驱动器 | 20 驱动器 |

### 仿真到真机：只改一个参数

```python
# 仿真
ChannelFactory.Instance().Init(0, "lo")      # 本地回环
# 真机
ChannelFactory.Instance().Init(0, "eth0")    # 以太网
# 其他代码完全不变
```

详见 [docs/unitree_dev_guide.md](docs/unitree_dev_guide.md)

---

## 架构

<p align="center">
  <img src="assets/architecture.png" width="500" alt="Architecture"/>
</p>

```
感知 Perception          决策 Decision           执行 Action
─────────────           ─────────────           ─────────────
关节角度+位置             神经网络推理             设置关节目标
相机图像(进阶)           obs → action            电机力矩控制
                         .pt 模型文件
   │                        │                       │
   └─ 仿真: PyBullet API    │                 仿真: data.ctrl
   └─ 真机: ROS 2 Topic     └─ 两边完全一样    真机: CAN → 电机
```

## 技术栈

| 组件 | 本项目 | 生产级 |
|------|-------|-------|
| 物理仿真 | **PyBullet** / **MuJoCo** | Isaac Sim (NVIDIA) |
| AI 框架 | **PyTorch** (19K参数 MLP) | VLA 大模型 (pi0/OpenVLA, 数十亿参数) |
| 机器人 | **Kuka iiwa** / **Unitree G1** | 真实 G1 / Go2 |
| 中间件 | 直接 API 调用 | ROS 2 + MoveIt 2 |
| 部署 | 你的电脑 | Jetson Orin |

## 学习路线

<p align="center">
  <img src="assets/roadmap.png" width="600" alt="Roadmap"/>
</p>

| 阶段 | 内容 | 时间 |
|------|------|------|
| **1. 本项目** | 运行 4 个 Demo，理解感知→决策→执行 | 1-2 周 |
| **2. 进阶仿真** | MuJoCo + PPO/SAC + 图像输入 | 2-4 周 |
| **3. ROS 2** | Docker 运行 ROS 2，Gazebo 仿真 | 2-4 周 |
| **4. 真机** | 入门机械臂 / 宇树 Go2/G1 | 持续 |

> **完整学习指南**：每个阶段需要掌握的原理、推荐资料、对应 Demo 代码 → [docs/learning_guide.md](docs/learning_guide.md)

## 上真机需要什么

| 条件 | 最低方案 | 推荐方案 |
|------|---------|---------|
| 硬件 | 舵机臂 (~800元) | Unitree G1 (~7.2万) |
| 主控 | 你的电脑 | Jetson Orin Nano (~3K) |
| 传感器 | 编码器(自带) | + RealSense 相机 |
| 中间件 | 串口/SDK | ROS 2 + MoveIt 2 |

## 项目结构

```
embodied-agent/
├── demos/
│   ├── 01_robot_arm_grasp.py       # 逆运动学到达基准 (PyBullet)
│   ├── 02_rl_training.py           # IK 引导的 TD3+BC 训练入口
│   ├── 03_deploy_model.py          # 模型部署对比
│   ├── 04_g1_actions.py            # 宇树 G1 动作演示 (MuJoCo)
│   ├── 05_unitree_viewer.py        # 官方 Unitree 模型通用查看器
│   ├── unitree_paths.py            # 外部 unitree_mujoco 路径解析
│   ├── rl_training.py              # 7维到达环境与训练流水线
│   └── grasp_training.py           # 8维接触抓取环境与训练流水线
├── tests/                           # 无 GUI 环境与路径解析单测
├── docs/
│   ├── tech_stack.md               # 技术栈全景
│   └── unitree_dev_guide.md        # 宇树开发指南
├── assets/                         # 图示 + 仿真截图
├── requirements.txt
└── README.md
```

## 相关资源

| 资源 | 链接 |
|------|------|
| PyBullet | [pybullet.org](https://pybullet.org/) |
| MuJoCo | [mujoco.readthedocs.io](https://mujoco.readthedocs.io/) |
| 宇树 SDK | [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) |
| 宇树仿真 | [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) |
| 宇树 RL | [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) |
| ROS 2 | [docs.ros.org](https://docs.ros.org/en/jazzy/) |
| Isaac Sim | [developer.nvidia.com](https://developer.nvidia.com/isaac-sim) |
| Open X-Embodiment | [robotics-transformer-x](https://robotics-transformer-x.github.io/) |
