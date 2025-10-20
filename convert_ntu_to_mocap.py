#!/usr/bin/env python3
"""
NTU RGB+D to Mocap format conversion script

Split strategy:
- Randomly select 50% of action sequences as test sources
- The last 20% of frames in test source sequences → test set (10% total data)
- The first 80% of frames in test source sequences + all frames of the remaining 50% → training set (90% total data)
- Keep temporal order within each sequence

Supported arguments:
- --ratio: Proportion of dataset to process (0.0–1.0), default 1.0 (all)
  Example: --ratio 0.01 processes only 1% of sequences (for quick testing)
"""

import numpy as np
import h5py
import cv2
import os
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, Manager, Lock
from functools import partial


# NTU to Mocap joint mapping (15 joints)
NTU_TO_MOCAP_MAPPING = [3, 2, 8, 4, 9, 5, 10, 6, 1, 16, 12, 17, 13, 18, 14]

# Kinect v2 camera intrinsics
FX, FY = 365.456, 365.456
CX, CY = 257.346, 210.347


def read_skeleton_file(skeleton_path):
    """Read skeleton file and return per-frame body data"""
    with open(skeleton_path, 'r') as f:
        lines = f.readlines()

    num_frames = int(lines[0].strip())
    all_frames = []

    line_idx = 1
    for frame_idx in range(num_frames):
        num_bodies = int(lines[line_idx].strip())
        line_idx += 1

        frame_bodies = []
        for _ in range(num_bodies):
            line_idx += 1  # body info
            num_joints = int(lines[line_idx].strip())
            line_idx += 1

            joints = []
            for _ in range(num_joints):
                joint_data = lines[line_idx].strip().split()
                x, y, z = float(joint_data[0]), float(joint_data[1]), float(joint_data[2])
                joints.append([x, y, z])
                line_idx += 1

            frame_bodies.append(np.array(joints))

        all_frames.append(frame_bodies)

    return all_frames


def depth_to_pointcloud(depth_image):
    """Convert depth image to 3D point cloud (unit: mm) - no downsampling, no filtering"""
    h, w = depth_image.shape
    points = []

    for v in range(h):
        for u in range(w):
            depth = depth_image[v, u] / 1000.0  # convert to meters

            if depth > 0:  # skip invalid depths
                # Convert to 3D coordinates (meters)
                x = (u - CX) * depth / FX
                y = -((v - CY) * depth / FY)  # flip Y axis
                z = depth

                # Convert to millimeters
                points.append([x * 1000, y * 1000, z * 1000])

    return np.array(points, dtype=np.float32)


def extract_mocap_joints(ntu_joints):
    """Extract 15 Mocap joints from NTU’s 25 joints (unit: mm)"""
    mocap_joints = ntu_joints[NTU_TO_MOCAP_MAPPING]  # shape: (15, 3)
    return mocap_joints * 1000  # meters → millimeters


def process_sequence(action_name, depth_base, skeleton_base, is_test_source, split_ratio=0.8):
    """
    Process a single action sequence

    Returns:
    - train_samples: [(pointcloud, joints), ...]
    - test_samples: [(pointcloud, joints), ...]
    """
    depth_dir = os.path.join(depth_base, action_name)
    skeleton_file = os.path.join(skeleton_base, f"{action_name}.skeleton")

    # Check file
    if not os.path.exists(skeleton_file):
        return [], []

    # Read skeleton data
    try:
        all_frames = read_skeleton_file(skeleton_file)
    except Exception as e:
        print(f"Error reading {action_name}: {e}")
        return [], []

    # Get depth image list
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])

    if len(depth_files) != len(all_frames):
        print(f"Warning: {action_name} frame count mismatch: {len(depth_files)} depth vs {len(all_frames)} skeleton")

    # Filter single-person frames
    single_person_frames = []
    multi_person_count = 0
    for frame_idx in range(min(len(depth_files), len(all_frames))):
        bodies = all_frames[frame_idx]
        if len(bodies) == 1:  # keep only single-person frames
            depth_path = os.path.join(depth_dir, depth_files[frame_idx])
            single_person_frames.append((depth_path, bodies[0], frame_idx))
        elif len(bodies) > 1:
            multi_person_count += 1

    if len(single_person_frames) == 0:
        from tqdm import tqdm
        tqdm.write(f"  Skipped {action_name}: no single-person frames (total={len(all_frames)}, multi-person={multi_person_count})")
        return [], []

    train_samples = []
    test_samples = []

    if is_test_source:
        # Test source sequences: first 80% → train, last 20% → test
        split_point = int(len(single_person_frames) * split_ratio)
        train_frames = single_person_frames[:split_point]
        test_frames = single_person_frames[split_point:]

        # Process training frames
        for depth_path, joints, _ in train_frames:
            try:
                depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_img is None:
                    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

                pointcloud = depth_to_pointcloud(depth_img)
                mocap_joints = extract_mocap_joints(joints)

                if len(pointcloud) > 0:
                    train_samples.append((pointcloud, mocap_joints, frame_idx))
            except Exception:
                continue

        # Process testing frames
        for depth_path, joints, frame_idx in test_frames:
            try:
                depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_img is None:
                    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

                pointcloud = depth_to_pointcloud(depth_img)
                mocap_joints = extract_mocap_joints(joints)

                if len(pointcloud) > 0:
                    test_samples.append((pointcloud, mocap_joints, frame_idx))
            except Exception:
                continue
    else:
        # Non-test source sequences: all frames → training
        for depth_path, joints, frame_idx in single_person_frames:
            try:
                depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_img is None:
                    depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

                pointcloud = depth_to_pointcloud(depth_img)
                mocap_joints = extract_mocap_joints(joints)

                if len(pointcloud) > 0:
                    train_samples.append((pointcloud, mocap_joints, frame_idx))
            except Exception:
                continue

    return train_samples, test_samples


def process_single_sequence(args):
    """
    Wrapper function for processing a single sequence (for multiprocessing)

    Returns: (seq_idx, action_name, train_results, test_results)
    train_results/test_results = [(pointcloud, joints, frame_idx), ...]
    """
    seq_idx, action_name, depth_base, skeleton_base, is_test_source = args

    train_samples, test_samples = process_sequence(
        action_name, depth_base, skeleton_base, is_test_source
    )

    return (seq_idx, action_name, train_samples, test_samples)


def save_batch_samples(samples, output_dir, train_temp_labels, test_temp_labels,
                       train_id_counter, test_id_counter, is_train, action_name, seq_idx):
    """
    Save samples in batch directly to disk, including traj_id and formatted IDs

    Returns: updated ID counter and temporary label list
    """
    if is_train:
        id_counter = train_id_counter
        temp_labels = train_temp_labels
    else:
        id_counter = test_id_counter
        temp_labels = test_temp_labels

    for pointcloud, joints, frame_idx in samples:
        sample_id = id_counter

        # Save point cloud immediately
        pc_file = os.path.join(output_dir, f"{sample_id}.npz")
        np.savez(pc_file, arr_0=pointcloud)

        # Format ID: seq_idx as person_id, sample_id as global frame number (5 digits)
        formatted_id = f"{seq_idx:02d}_{sample_id:05d}"

        # Record label
        temp_labels.append({
            'id': formatted_id.encode('utf-8'),
            'coords': joints,
            'traj_id': action_name.encode('utf-8')  # use action sequence name as traj_id
        })

        id_counter += 1

    return id_counter, temp_labels


def save_labels_to_h5(temp_labels, label_file):
    """Save temporary label list to H5 file, including traj_id"""
    ids = [item['id'] for item in temp_labels]
    coordinates = [item['coords'] for item in temp_labels]
    traj_ids = [item['traj_id'] for item in temp_labels]

    with h5py.File(label_file, 'w') as f:
        f.create_dataset('id', data=np.array(ids, dtype='S'))
        f.create_dataset('real_world_coordinates',
                        data=np.array(coordinates, dtype=np.float32))
        f.create_dataset('traj_id', data=np.array(traj_ids, dtype='S'))

    print(f"Saved {len(temp_labels)} labels to {label_file}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='NTU RGB+D to Mocap format conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process the entire dataset (default)
  python convert_ntu_to_mocap.py

  # Process 1% of dataset (quick test)
  python convert_ntu_to_mocap.py --ratio 0.01

  # Process 10% of dataset
  python convert_ntu_to_mocap.py --ratio 0.1

  # Process 50% of dataset with 16 workers
  python convert_ntu_to_mocap.py --ratio 0.5 --workers 16
        """
    )

    parser.add_argument(
        '--ratio',
        type=float,
        default=1.0,
        help='Proportion of dataset to process (0.0–1.0), default 1.0 for all'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel processes, default uses all CPU cores'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed, default 42'
    )

    args = parser.parse_args()

    # Validate parameters
    if not 0.0 < args.ratio <= 1.0:
        parser.error("--ratio must be within (0.0, 1.0]")

    return args


def main():
    # Parse arguments
    args = parse_args()

    # Path configuration
    depth_base = '/media/tom-wang/SSD/ROSE-Lab/nturgbd_depth_masked'
    skeleton_base = '/media/tom-wang/SSD/ROSE-Lab/nturgbd_skeletons'
    output_base = '/media/tom-wang/SSD/ROSE-Lab'

    train_dir = os.path.join(output_base, 'train_10')
    test_dir = os.path.join(output_base, 'test_10')
    train_label_file = os.path.join(output_base, 'train_labels_10.h5')
    test_label_file = os.path.join(output_base, 'test_labels_10.h5')

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("=" * 80)
    print("NTU RGB+D → Mocap format conversion")
    print("=" * 80)

    # Display configuration
    print(f"\nConversion settings:")
    print(f"  Dataset ratio: {args.ratio * 100:.1f}%")
    print(f"  Random seed: {args.seed}")
    if args.workers:
        print(f"  Parallel workers: {args.workers}")
    else:
        print(f"  Parallel workers: Auto-detect")

    # Get all action sequences
    all_action_dirs = sorted([d for d in os.listdir(depth_base)
                              if os.path.isdir(os.path.join(depth_base, d))])
    print(f"\nTotal action sequences: {len(all_action_dirs)}")

    # Select sequences by ratio
    random.seed(args.seed)
    if args.ratio < 1.0:
        num_sequences = int(len(all_action_dirs) * args.ratio)
        action_dirs = random.sample(all_action_dirs, num_sequences)
        print(f"Randomly selected {args.ratio * 100:.1f}% of sequences: {len(action_dirs)}")
    else:
        action_dirs = all_action_dirs
        print(f"Processing all sequences: {len(action_dirs)}")

    # Random shuffle and split 50% as test sources
    random.shuffle(action_dirs)
    split_idx = len(action_dirs) // 2
    test_source_sequences = set(action_dirs[:split_idx])
    print(f"Test source sequences: {len(test_source_sequences)}")
    print(f"Train source sequences: {len(action_dirs) - len(test_source_sequences)}")

    # Initialize counters and temp labels
    train_id_counter = 0
    test_id_counter = 0
    train_temp_labels = []
    test_temp_labels = []

    # Multiprocessing setup
    num_workers = args.workers if args.workers else os.cpu_count()
    print(f"\nUsing {num_workers} processes for parallel processing...")

    # Prepare tasks
    tasks = []
    for seq_idx, action_name in enumerate(action_dirs):
        is_test_source = action_name in test_source_sequences
        tasks.append((seq_idx, action_name, depth_base, skeleton_base, is_test_source))

    print("\nStart processing sequences (parallel mode)...")
    total_train = 0
    total_test = 0

    # Parallel processing
    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(process_single_sequence, tasks, chunksize=10)

        for seq_idx, action_name, train_samples, test_samples in tqdm(results, total=len(tasks), desc="Processing sequences"):
            # Save training samples
            if train_samples:
                train_id_counter, train_temp_labels = save_batch_samples(
                    train_samples, train_dir, train_temp_labels, test_temp_labels,
                    train_id_counter, test_id_counter, is_train=True,
                    action_name=action_name, seq_idx=seq_idx
                )
                total_train += len(train_samples)

            # Save testing samples
            if test_samples:
                test_id_counter, test_temp_labels = save_batch_samples(
                    test_samples, test_dir, train_temp_labels, test_temp_labels,
                    train_id_counter, test_id_counter, is_train=False,
                    action_name=action_name, seq_idx=seq_idx
                )
                total_test += len(test_samples)

    print("\n" + "=" * 80)
    print("Sequence processing complete, saving label files...")
    print("=" * 80)
    print(f"Total training samples: {total_train}")
    print(f"Total testing samples: {total_test}")
    print(f"Total: {total_train + total_test}")

    # Save label files
    print("\nSaving training labels...")
    save_labels_to_h5(train_temp_labels, train_label_file)

    print("Saving testing labels...")
    save_labels_to_h5(test_temp_labels, test_label_file)

    # Clear memory
    train_temp_labels.clear()
    test_temp_labels.clear()

    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)
    print(f"Training data: {train_dir} ({total_train} files)")
    print(f"Training labels: {train_label_file}")
    print(f"Testing data: {test_dir} ({total_test} files)")
    print(f"Testing labels: {test_label_file}")


if __name__ == '__main__':
    main()
