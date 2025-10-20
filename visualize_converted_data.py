#!/usr/bin/env python3
"""
可视化转换后的Mocap格式数据
支持播放随机选择的一个完整序列（动画）
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import random
from collections import defaultdict


# Mocap 15个关节点连接关系
MOCAP_SKELETON_CONNECTIONS = [
    (0, 1),   # Head - Neck
    (1, 2),   # Neck - R Shoulder
    (1, 3),   # Neck - L Shoulder
    (2, 4),   # R Shoulder - R Elbow
    (3, 5),   # L Shoulder - L Elbow
    (4, 6),   # R Elbow - R Hand
    (5, 7),   # L Elbow - L Hand
    (1, 8),   # Neck - Torso
    (8, 9),   # Torso - R Hip
    (8, 10),  # Torso - L Hip
    (9, 11),  # R Hip - R Knee
    (10, 12), # L Hip - L Knee
    (11, 13), # R Knee - R Foot
    (12, 14), # L Knee - L Foot
]

# Mocap关节点名称
MOCAP_JOINT_NAMES = [
    "Head", "Neck", "R Shoulder", "L Shoulder",
    "R Elbow", "L Elbow", "R Hand", "L Hand",
    "Torso", "R Hip", "L Hip", "R Knee", "L Knee",
    "R Foot", "L Foot"
]


def load_labels(label_file):
    """加载所有标签数据"""
    with h5py.File(label_file, 'r') as f:
        ids = [id_bytes.decode('utf-8') for id_bytes in f['id'][:]]
        coordinates = f['real_world_coordinates'][:]
        traj_ids = [traj.decode('utf-8') for traj in f['traj_id'][:]]

    return ids, coordinates, traj_ids


def group_samples_by_trajectory(ids, coordinates, traj_ids):
    """按轨迹ID分组样本"""
    traj_groups = defaultdict(list)

    for i, (sample_id, coords, traj_id) in enumerate(zip(ids, coordinates, traj_ids)):
        # 提取文件编号（用于排序点云文件）
        file_num = int(sample_id.split('_')[1])
        traj_groups[traj_id].append((file_num, sample_id, coords))

    # 对每个轨迹的样本按文件编号排序
    for traj_id in traj_groups:
        traj_groups[traj_id].sort(key=lambda x: x[0])

    return traj_groups


def load_sequence_data(sample_dir, traj_samples):
    """加载一个完整序列的点云和关节点数据"""
    pointclouds = []
    joints_list = []

    for file_num, sample_id, joints in traj_samples:
        pc_file = os.path.join(sample_dir, f"{file_num}.npz")

        if not os.path.exists(pc_file):
            print(f"Warning: 点云文件不存在 {pc_file}")
            continue

        data = np.load(pc_file)
        pointcloud = data['arr_0']

        # 下采样点云
        if len(pointcloud) > 2048:
            indices = np.random.choice(len(pointcloud), 2048, replace=False)
            pointcloud = pointcloud[indices]

        pointclouds.append(pointcloud)
        joints_list.append(joints)

    return pointclouds, joints_list


def animate_sequence(pointclouds, joints_list, traj_id, dataset_name):
    """动画播放一个序列"""
    if len(pointclouds) == 0:
        print("没有可播放的帧")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 计算所有帧的边界
    all_points = np.vstack(pointclouds)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    # 添加边距
    margin = 200
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    ax.set_zlim([z_min - margin, z_max + margin])

    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.view_init(elev=20, azim=45)

    # 初始化绘图对象
    pc_scatter = ax.scatter([], [], [], c='cyan', s=1.0, alpha=0.5, label='Point Cloud')
    joint_scatter = ax.scatter([], [], [], c='red', s=50, alpha=0.9, label='Joints')
    skeleton_lines = []
    for _ in MOCAP_SKELETON_CONNECTIONS:
        line, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.7)
        skeleton_lines.append(line)

    # 为每个关节点创建文本标签
    joint_texts = []
    for i in range(15):  # 15个关节点
        text = ax.text(0, 0, 0, '', fontsize=8, color='darkblue',
                      fontweight='bold', alpha=1.0)
        joint_texts.append(text)

    title_text = ax.set_title('', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')

    def init():
        """初始化动画"""
        pc_scatter._offsets3d = ([], [], [])
        joint_scatter._offsets3d = ([], [], [])
        for line in skeleton_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        for text in joint_texts:
            text.set_position((0, 0))
            text.set_3d_properties(0)
        return [pc_scatter, joint_scatter] + skeleton_lines + joint_texts + [title_text]

    def update(frame):
        """更新每一帧"""
        # 更新点云
        pc = pointclouds[frame]
        pc_scatter._offsets3d = (pc[:, 0], pc[:, 1], pc[:, 2])

        # 更新关节点
        joints = joints_list[frame]
        joint_scatter._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])

        # 更新骨骼连接
        for i, connection in enumerate(MOCAP_SKELETON_CONNECTIONS):
            if connection[0] < len(joints) and connection[1] < len(joints):
                p1 = joints[connection[0]]
                p2 = joints[connection[1]]
                skeleton_lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                skeleton_lines[i].set_3d_properties([p1[2], p2[2]])

        # 更新关节点标签
        for i in range(min(len(joints), 15)):
            joint = joints[i]
            joint_texts[i].set_text(MOCAP_JOINT_NAMES[i])
            joint_texts[i].set_position((joint[0], joint[1]))
            joint_texts[i].set_3d_properties(joint[2], 'z')

        # 更新标题
        title_text.set_text(f'{dataset_name} - {traj_id}\nFrame {frame + 1}/{len(pointclouds)}')

        return [pc_scatter, joint_scatter] + skeleton_lines + joint_texts + [title_text]

    print(f"\n播放序列: {traj_id}")
    print(f"总帧数: {len(pointclouds)}")
    print("关闭窗口以继续...")

    # 创建动画 (30 FPS，每帧33ms)
    anim = FuncAnimation(fig, update, init_func=init, frames=len(pointclouds),
                        interval=33, blit=False, repeat=True)

    plt.tight_layout()
    plt.show()


def visualize_single_frame(pointcloud, joints, sample_id, traj_id, dataset_name):
    """可视化单帧"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
               c='cyan', s=1.0, alpha=0.5, label='Point Cloud')

    # 绘制关节点
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c='red', s=50, alpha=0.9, label='Joints')

    # 绘制骨骼连接
    for connection in MOCAP_SKELETON_CONNECTIONS:
        if connection[0] < len(joints) and connection[1] < len(joints):
            p1 = joints[connection[0]]
            p2 = joints[connection[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                   'r-', linewidth=2, alpha=0.7)

    # 添加关节点名称
    for i, joint in enumerate(joints):
        ax.text(joint[0], joint[1], joint[2], f'  {MOCAP_JOINT_NAMES[i]}',
               fontsize=8, color='darkblue', fontweight='bold', alpha=1.0)

    ax.set_title(f'{dataset_name} - {traj_id}\nSample ID: {sample_id}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.legend(loc='upper left')
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()


def print_sequence_info(traj_samples, traj_id):
    """打印序列信息"""
    print(f"\n序列信息: {traj_id}")
    print(f"  帧数: {len(traj_samples)}")
    print(f"  样本ID范围: {traj_samples[0][1]} -> {traj_samples[-1][1]}")

    # 打印第一帧的关节点信息
    first_joints = traj_samples[0][2]
    print(f"\n第一帧关节点坐标 (毫米):")
    for i in range(min(5, len(first_joints))):  # 只打印前5个关节点
        joint = first_joints[i]
        print(f"  ID {i:2d}: {MOCAP_JOINT_NAMES[i]:15s} - ({joint[0]:8.2f}, {joint[1]:8.2f}, {joint[2]:8.2f})")
    if len(first_joints) > 5:
        print(f"  ... (共 {len(first_joints)} 个关节点)")


def main():
    # 测试输出路径
    train_dir = '/media/oliver/SSD/ROSE-Lab/train'
    test_dir = '/media/oliver/SSD/ROSE-Lab/test'
    train_label_file = '/media/oliver/SSD/ROSE-Lab/train_labels.h5'
    test_label_file = '/media/oliver/SSD/ROSE-Lab/test_labels.h5'

    print("=" * 80)
    print("可视化转换后的Mocap格式数据 - 序列播放模式")
    print("=" * 80)

    # 选择数据集
    dataset_choice = input("\n选择数据集 (train/test，默认train): ").strip().lower()
    if dataset_choice == 'test':
        sample_dir = test_dir
        label_file = test_label_file
        dataset_name = "Test Set"
    else:
        sample_dir = train_dir
        label_file = train_label_file
        dataset_name = "Training Set"

    # 加载标签
    print(f"\n加载标签文件: {label_file}")
    ids, coordinates, traj_ids = load_labels(label_file)
    print(f"总样本数: {len(ids)}")

    # 按轨迹分组
    traj_groups = group_samples_by_trajectory(ids, coordinates, traj_ids)
    print(f"总序列数: {len(traj_groups)}")

    # 显示序列统计
    traj_list = list(traj_groups.items())
    traj_list.sort(key=lambda x: len(x[1]), reverse=True)
    print(f"\n序列统计 (按帧数排序，显示前10):")
    for i, (traj_id, samples) in enumerate(traj_list[:10]):
        print(f"  {i+1}. {traj_id}: {len(samples)} 帧")

    # 随机选择一个序列
    selected_traj_id = random.choice(list(traj_groups.keys()))
    traj_samples = traj_groups[selected_traj_id]

    print(f"\n随机选择序列: {selected_traj_id}")
    print_sequence_info(traj_samples, selected_traj_id)

    # 加载序列数据
    print("\n加载序列数据...")
    pointclouds, joints_list = load_sequence_data(sample_dir, traj_samples)

    if len(pointclouds) == 0:
        print("错误: 无法加载序列数据")
        return

    print(f"成功加载 {len(pointclouds)} 帧")

    # 选择可视化模式
    mode = input("\n选择可视化模式 (animation/single，默认animation): ").strip().lower()

    if mode == 'single':
        # 单帧模式：显示第一帧
        frame_idx = 0
        print(f"\n显示第 {frame_idx + 1} 帧")
        file_num, sample_id, joints = traj_samples[frame_idx]
        visualize_single_frame(pointclouds[frame_idx], joints_list[frame_idx],
                              sample_id, selected_traj_id, dataset_name)
    else:
        # 动画模式：播放整个序列
        animate_sequence(pointclouds, joints_list, selected_traj_id, dataset_name)

    print("\n" + "=" * 80)
    print("可视化完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
