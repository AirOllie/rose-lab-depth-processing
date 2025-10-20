#!/usr/bin/env python3
"""
统计NTU RGB+D数据集的点云和姿态标注分布
"""

import os
from collections import defaultdict
from tqdm import tqdm


def count_depth_frames(depth_dir):
    """统计深度图目录中的帧数"""
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
    return len(depth_files)


def read_skeleton_statistics(skeleton_path):
    """读取骨架文件，统计每帧的人体数量"""
    with open(skeleton_path, 'r') as f:
        lines = f.readlines()

    num_frames = int(lines[0].strip())
    bodies_per_frame = []

    line_idx = 1
    for frame_idx in range(num_frames):
        num_bodies = int(lines[line_idx].strip())
        bodies_per_frame.append(num_bodies)
        line_idx += 1

        # 跳过人体数据
        for _ in range(num_bodies):
            line_idx += 1  # body info
            num_joints = int(lines[line_idx].strip())
            line_idx += 1
            line_idx += num_joints  # joint data

    return bodies_per_frame


def main():
    # 数据路径
    depth_base = '/media/oliver/SSD/ROSE-Lab/nturgbd_depth_masked'
    skeleton_base = '/media/oliver/SSD/ROSE-Lab/nturgbd_skeletons'

    # 获取所有动作目录
    action_dirs = sorted([d for d in os.listdir(depth_base)
                         if os.path.isdir(os.path.join(depth_base, d))])

    print("=" * 80)
    print("NTU RGB+D 数据集统计分析")
    print("=" * 80)
    print(f"\n总动作序列数: {len(action_dirs)}")
    print("\n开始统计...\n")

    # 统计变量
    total_depth_frames = 0
    total_skeleton_frames = 0
    total_annotated_frames = 0  # 至少有1个人体标注的帧
    total_bodies = 0

    bodies_distribution = defaultdict(int)  # 每帧人体数量的分布
    missing_skeleton = []  # 缺少骨架文件的动作
    mismatch_frames = []  # 深度帧数和骨架帧数不匹配的动作

    # 遍历所有动作
    for action_name in tqdm(action_dirs, desc="分析进度"):
        depth_dir = os.path.join(depth_base, action_name)
        skeleton_file = os.path.join(skeleton_base, f"{action_name}.skeleton")

        # 统计深度帧
        num_depth_frames = count_depth_frames(depth_dir)
        total_depth_frames += num_depth_frames

        # 检查骨架文件
        if not os.path.exists(skeleton_file):
            missing_skeleton.append(action_name)
            continue

        # 统计骨架数据
        try:
            bodies_per_frame = read_skeleton_statistics(skeleton_file)
            num_skeleton_frames = len(bodies_per_frame)
            total_skeleton_frames += num_skeleton_frames

            # 统计人体数量分布
            for num_bodies in bodies_per_frame:
                bodies_distribution[num_bodies] += 1
                if num_bodies > 0:
                    total_annotated_frames += 1
                    total_bodies += num_bodies

            # 检查帧数是否匹配
            if num_depth_frames != num_skeleton_frames:
                mismatch_frames.append({
                    'action': action_name,
                    'depth_frames': num_depth_frames,
                    'skeleton_frames': num_skeleton_frames
                })

        except Exception as e:
            print(f"\n处理 {action_name} 时出错: {e}")
            continue

    # 输出统计结果
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)

    print(f"\n【点云数据统计】")
    print(f"  总深度图帧数: {total_depth_frames:,}")
    print(f"  总骨架标注帧数: {total_skeleton_frames:,}")

    if total_depth_frames != total_skeleton_frames:
        print(f"  ⚠️  深度帧和骨架帧数量不匹配!")
        print(f"  差异: {abs(total_depth_frames - total_skeleton_frames):,} 帧")

    print(f"\n【人体标注统计】")
    print(f"  有标注的帧数: {total_annotated_frames:,} ({total_annotated_frames/max(total_skeleton_frames,1)*100:.2f}%)")
    print(f"  无标注的帧数: {bodies_distribution[0]:,} ({bodies_distribution[0]/max(total_skeleton_frames,1)*100:.2f}%)")
    print(f"  标注的总人体数: {total_bodies:,}")
    print(f"  平均每帧人体数: {total_bodies/max(total_annotated_frames,1):.2f}")

    print(f"\n【人体数量分布】")
    print(f"  {'人体数':<10} {'帧数':<12} {'占比':<10} {'可视化'}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*40}")

    max_bodies = max(bodies_distribution.keys()) if bodies_distribution else 0
    for num_bodies in range(max_bodies + 1):
        count = bodies_distribution[num_bodies]
        percentage = count / max(total_skeleton_frames, 1) * 100
        bar_length = int(percentage / 2)  # 每2%一个字符
        bar = '█' * bar_length
        print(f"  {num_bodies:<10} {count:<12,} {percentage:>6.2f}%   {bar}")

    # 详细分布统计
    print(f"\n【详细分布】")
    single_person_frames = bodies_distribution[1]
    multi_person_frames = sum(bodies_distribution[i] for i in range(2, max_bodies + 1))

    print(f"  单人帧 (1人): {single_person_frames:,} ({single_person_frames/max(total_skeleton_frames,1)*100:.2f}%)")
    print(f"  多人帧 (≥2人): {multi_person_frames:,} ({multi_person_frames/max(total_skeleton_frames,1)*100:.2f}%)")

    if multi_person_frames > 0:
        print(f"\n  多人帧具体分布:")
        for num_bodies in range(2, max_bodies + 1):
            if bodies_distribution[num_bodies] > 0:
                count = bodies_distribution[num_bodies]
                percentage = count / multi_person_frames * 100
                print(f"    {num_bodies}人: {count:,} 帧 ({percentage:.2f}% of 多人帧)")

    # 问题报告
    if missing_skeleton:
        print(f"\n【警告】缺少骨架文件的动作 ({len(missing_skeleton)} 个):")
        for i, action in enumerate(missing_skeleton[:10], 1):
            print(f"  {i}. {action}")
        if len(missing_skeleton) > 10:
            print(f"  ... 还有 {len(missing_skeleton) - 10} 个")

    if mismatch_frames:
        print(f"\n【警告】帧数不匹配的动作 ({len(mismatch_frames)} 个):")
        for i, item in enumerate(mismatch_frames[:10], 1):
            print(f"  {i}. {item['action']}: 深度={item['depth_frames']}, 骨架={item['skeleton_frames']}")
        if len(mismatch_frames) > 10:
            print(f"  ... 还有 {len(mismatch_frames) - 10} 个")

    print("\n" + "=" * 80)
    print("统计完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
