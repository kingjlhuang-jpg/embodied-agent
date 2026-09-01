"""
示例3: 加载训练好的AI模型，部署到（仿真的）机器人上
=================================================

这个程序演示"部署"阶段：
1. 加载训练好的 Stable-Baselines3 .zip 模型文件
2. 在仿真环境中运行（带3D可视化）
3. 观察AI控制机械臂的效果

在真实机器人上部署时，代码几乎一样——
只是把 RobotArmEnv 替换为真实硬件的 ROS 2 接口。
"""

import os
import sys
import time
from pathlib import Path

from stable_baselines3 import TD3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 复用训练时的环境定义
from rl_training import RobotArmEnv


def deploy_in_simulation():
    """在仿真中部署AI模型"""
    print("=" * 60)
    print("  部署AI模型到仿真机器人")
    print("=" * 60)
    print()

    # 1. 加载训练好的 TD3+BC 模型
    model_path = Path("trained_policy.zip")
    try:
        policy = TD3.load(model_path, device="auto")
        print(f"[OK] 已加载模型: {model_path}")
    except FileNotFoundError:
        print(f"[!] 未找到 {model_path}")
        print("    请先运行 demos/02_rl_training.py 训练模型")
        return

    # 2. 创建仿真环境（带3D可视化）
    print("[OK] 创建仿真环境（带可视化窗口）...")
    env = RobotArmEnv(render=True)

    print()
    print("开始运行！你可以用鼠标旋转视角。")
    print("AI将尝试控制机械臂到达绿色目标球。")
    print("按 Ctrl+C 退出。")
    print()

    try:
        episode = 0
        while True:
            episode += 1
            obs, _ = env.reset()
            total_reward = 0

            print(f"--- 第 {episode} 轮 (目标位置: "
                  f"{env.target_pos.round(3)}) ---")

            for step in range(200):
                # AI推理：观测 → 动作
                action, _ = policy.predict(obs, deterministic=True)

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # 慢放，方便观察
                time.sleep(1.0 / 60.0)

                if terminated or truncated:
                    break

            distance = info["distance"]
            result = "到达目标!" if distance < 0.05 else "未到达"
            print(f"  结果: {result} | "
                  f"最终距离: {distance:.4f}m | "
                  f"奖励: {total_reward:.1f}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n部署演示结束。")
    finally:
        env.close()


def show_deployment_comparison():
    """展示仿真部署 vs 真实部署的代码对比"""
    print()
    print("=" * 60)
    print("  仿真 vs 真实机器人：代码对比")
    print("=" * 60)
    print()
    print("  [仿真环境中运行（你现在做的）]")
    print("  +-----------------------------------------+")
    print("  | obs, _ = env.reset()       # 仿真器API  |")
    print("  | action, _ = policy.predict(obs) # 推理  |")
    print("  | obs, r, term, trunc, _ = env.step(a)   |")
    print("  +-----------------------------------------+")
    print()
    print("  [真实机器人上运行（买了机器人后）]")
    print("  +-----------------------------------------+")
    print("  | obs = ros_node.get_obs() # ROS传感器    |")
    print("  | action = policy(obs)     # 同一个AI     |")
    print("  | ros_node.send_cmd(action) # ROS电机     |")
    print("  +-----------------------------------------+")
    print()
    print("  唯一的区别：数据来源和执行目标。")
    print("  AI模型（policy）完全不变。")
    print("=" * 60)


if __name__ == "__main__":
    show_deployment_comparison()
    deploy_in_simulation()
