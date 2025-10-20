#!/usr/bin/env python3
"""
Statistics of NTU RGB+D dataset: point cloud and pose annotation distribution
"""

import os
from collections import defaultdict
from tqdm import tqdm


def count_depth_frames(depth_dir):
    """Count the number of depth frames in a directory"""
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
    return len(depth_files)


def read_skeleton_statistics(skeleton_path):
    """Read a skeleton file and count the number of human bodies per frame"""
    with open(skeleton_path, 'r') as f:
        lines = f.readlines()

    num_frames = int(lines[0].strip())
    bodies_per_frame = []

    line_idx = 1
    for frame_idx in range(num_frames):
        num_bodies = int(lines[line_idx].strip())
        bodies_per_frame.append(num_bodies)
        line_idx += 1

        # Skip body data
        for _ in range(num_bodies):
            line_idx += 1  # body info
            num_joints = int(lines[line_idx].strip())
            line_idx += 1
            line_idx += num_joints  # joint data

    return bodies_per_frame


def main():
    # Dataset paths
    depth_base = '/media/oliver/SSD/ROSE-Lab/nturgbd_depth_masked'
    skeleton_base = '/media/oliver/SSD/ROSE-Lab/nturgbd_skeletons'

    # Get all action directories
    action_dirs = sorted([d for d in os.listdir(depth_base)
                         if os.path.isdir(os.path.join(depth_base, d))])

    print("=" * 80)
    print("NTU RGB+D Dataset Statistical Analysis")
    print("=" * 80)
    print(f"\nTotal action sequences: {len(action_dirs)}")
    print("\nStarting analysis...\n")

    # Counters
    total_depth_frames = 0
    total_skeleton_frames = 0
    total_annotated_frames = 0  # Frames with at least one body annotation
    total_bodies = 0

    bodies_distribution = defaultdict(int)  # Distribution of number of bodies per frame
    missing_skeleton = []  # Actions missing skeleton files
    mismatch_frames = []  # Actions where depth and skeleton frame counts differ

    # Iterate through all actions
    for action_name in tqdm(action_dirs, desc="Progress"):
        depth_dir = os.path.join(depth_base, action_name)
        skeleton_file = os.path.join(skeleton_base, f"{action_name}.skeleton")

        # Count depth frames
        num_depth_frames = count_depth_frames(depth_dir)
        total_depth_frames += num_depth_frames

        # Check skeleton file
        if not os.path.exists(skeleton_file):
            missing_skeleton.append(action_name)
            continue

        # Count skeleton data
        try:
            bodies_per_frame = read_skeleton_statistics(skeleton_file)
            num_skeleton_frames = len(bodies_per_frame)
            total_skeleton_frames += num_skeleton_frames

            # Count body distribution
            for num_bodies in bodies_per_frame:
                bodies_distribution[num_bodies] += 1
                if num_bodies > 0:
                    total_annotated_frames += 1
                    total_bodies += num_bodies

            # Check frame count mismatch
            if num_depth_frames != num_skeleton_frames:
                mismatch_frames.append({
                    'action': action_name,
                    'depth_frames': num_depth_frames,
                    'skeleton_frames': num_skeleton_frames
                })

        except Exception as e:
            print(f"\nError processing {action_name}: {e}")
            continue

    # Output results
    print("\n" + "=" * 80)
    print("Statistics Summary")
    print("=" * 80)

    print(f"\n[Point Cloud Data]")
    print(f"  Total depth frames: {total_depth_frames:,}")
    print(f"  Total skeleton annotation frames: {total_skeleton_frames:,}")

    if total_depth_frames != total_skeleton_frames:
        print(f"  ⚠️  Mismatch between depth and skeleton frame counts!")
        print(f"  Difference: {abs(total_depth_frames - total_skeleton_frames):,} frames")

    print(f"\n[Human Annotation Statistics]")
    print(f"  Frames with annotations: {total_annotated_frames:,} ({total_annotated_frames/max(total_skeleton_frames,1)*100:.2f}%)")
    print(f"  Frames without annotations: {bodies_distribution[0]:,} ({bodies_distribution[0]/max(total_skeleton_frames,1)*100:.2f}%)")
    print(f"  Total number of annotated humans: {total_bodies:,}")
    print(f"  Average humans per annotated frame: {total_bodies/max(total_annotated_frames,1):.2f}")

    print(f"\n[Human Count Distribution]")
    print(f"  {'#Bodies':<10} {'#Frames':<12} {'Percent':<10} {'Visualization'}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*40}")

    max_bodies = max(bodies_distribution.keys()) if bodies_distribution else 0
    for num_bodies in range(max_bodies + 1):
        count = bodies_distribution[num_bodies]
        percentage = count / max(total_skeleton_frames, 1) * 100
        bar_length = int(percentage / 2)  # 1 char per 2%
        bar = '█' * bar_length
        print(f"  {num_bodies:<10} {count:<12,} {percentage:>6.2f}%   {bar}")

    # Detailed breakdown
    print(f"\n[Detailed Breakdown]")
    single_person_frames = bodies_distribution[1]
    multi_person_frames = sum(bodies_distribution[i] for i in range(2, max_bodies + 1))

    print(f"  Single-person frames (1 body): {single_person_frames:,} ({single_person_frames/max(total_skeleton_frames,1)*100:.2f}%)")
    print(f"  Multi-person frames (≥2 bodies): {multi_person_frames:,} ({multi_person_frames/max(total_skeleton_frames,1)*100:.2f}%)")

    if multi_person_frames > 0:
        print(f"\n  Multi-person frame breakdown:")
        for num_bodies in range(2, max_bodies + 1):
            if bodies_distribution[num_bodies] > 0:
                count = bodies_distribution[num_bodies]
                percentage = count / multi_person_frames * 100
                print(f"    {num_bodies} bodies: {count:,} frames ({percentage:.2f}% of multi-person frames)")

    # Issue reports
    if missing_skeleton:
        print(f"\n[Warning] Actions missing skeleton files ({len(missing_skeleton)}):")
        for i, action in enumerate(missing_skeleton[:10], 1):
            print(f"  {i}. {action}")
        if len(missing_skeleton) > 10:
            print(f"  ... and {len(missing_skeleton) - 10} more")

    if mismatch_frames:
        print(f"\n[Warning] Actions with mismatched frame counts ({len(mismatch_frames)}):")
        for i, item in enumerate(mismatch_frames[:10], 1):
            print(f"  {i}. {item['action']}: depth={item['depth_frames']}, skeleton={item['skeleton_frames']}")
        if len(mismatch_frames) > 10:
            print(f"  ... and {len(mismatch_frames) - 10} more")

    print("\n" + "=" * 80)
    print("Statistics complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
