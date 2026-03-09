#!/usr/bin/env python3
"""
LiDAR-Camera Extrinsic Calibration
Finds R and t such that: p_camera = R @ p_lidar + t

Usage:
    python3 calibrate.py --bag /bags/calibration_bag \
                         --image-topic /oak/rgb/image_raw \
                         --lidar-topic /livox/lidar \
                         --camera-info-topic /oak/rgb/camera_info \
                         --checkerboard 8 6 \
                         --square-size 0.025 \
                         --output /bags/result
"""

import argparse
import os
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# ─────────────────────────────────────────────
# 1. PARSE ARGS
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='LiDAR-Camera Calibration')
    parser.add_argument('--bag',               required=True,  help='Path to rosbag2 folder')
    parser.add_argument('--image-topic',       default='/oak/rgb/image_raw')
    parser.add_argument('--lidar-topic',       default='/livox/lidar')
    parser.add_argument('--camera-info-topic', default='/oak/rgb/camera_info')
    parser.add_argument('--checkerboard',      nargs=2, type=int, default=[12, 6],
                        help='Inner corners (cols rows), e.g. 8 6')
    parser.add_argument('--square-size',       type=float, default=0.03,
                        help='Square size in meters')
    parser.add_argument('--output',            default='./result',
                        help='Output directory for results')
    parser.add_argument('--sync-threshold-ms', type=float, default=50.0,
                        help='Max time diff (ms) to consider frames synced')
    parser.add_argument('--lidar-min-dist',    type=float, default=0.5)
    parser.add_argument('--lidar-max-dist',    type=float, default=5.0)
    parser.add_argument('--ransac-threshold',  type=float, default=0.01,
                        help='RANSAC plane inlier threshold in meters')
    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. READ BAG
# ─────────────────────────────────────────────

def read_bag(bag_path, topics):
    """Read all messages from bag, return dict of {topic: [(timestamp, msg)]}"""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    data = {t: [] for t in topics}

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            data[connection.topic].append((timestamp, msg))

    for t in topics:
        print(f"  {t}: {len(data[t])} messages")
    return data


# ─────────────────────────────────────────────
# 3. SYNC FRAMES
# ─────────────────────────────────────────────

def sync_frames(images, lidars, threshold_ns):
    """Match each image to the closest lidar frame within threshold."""
    pairs = []
    lidar_times = np.array([t for t, _ in lidars])

    for img_time, img_msg in images:
        diffs = np.abs(lidar_times - img_time)
        idx = diffs.argmin()
        if diffs[idx] < threshold_ns:
            pairs.append((img_msg, lidars[idx][1]))

    print(f"  Synced {len(pairs)} frame pairs")
    return pairs


# ─────────────────────────────────────────────
# 4. EXTRACT CAMERA INTRINSICS
# ─────────────────────────────────────────────

def extract_intrinsics(camera_info_msgs):
    """Get K and dist from first camera_info message."""
    _, msg = camera_info_msgs[0]
    K = np.array(msg.k).reshape(3, 3)
    dist = np.array(msg.d)
    print(f"  K =\n{K}")
    print(f"  dist = {dist}")
    return K, dist


# ─────────────────────────────────────────────
# 5. PROCESS CAMERA IMAGE
# ─────────────────────────────────────────────

def process_image(img_msg, checkerboard, square_size, K, dist):
    """Detect checkerboard and return centroid in camera frame."""
    # Decode image
    data = np.frombuffer(bytes(img_msg.data), dtype=np.uint8)

    encoding = img_msg.encoding.lower()
    if 'rgb' in encoding:
        img = data.reshape(img_msg.height, img_msg.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif 'bgr' in encoding:
        img = data.reshape(img_msg.height, img_msg.width, 3)
    elif 'mono' in encoding or 'gray' in encoding:
        img = data.reshape(img_msg.height, img_msg.width)
    else:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, tuple(checkerboard), None)
    if not ret:
        return None, None

    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # Build object points
    cols, rows = checkerboard
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    # Solve PnP
    ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ret:
        return None, None

    # Board centroid in camera frame = tvec
    centroid_cam = tvec.flatten()
    return centroid_cam, img


# ─────────────────────────────────────────────
# 6. PROCESS LIDAR CLOUD
# ─────────────────────────────────────────────

def pointcloud2_to_numpy(msg):
    """Convert PointCloud2 message to Nx3 numpy array."""
    import struct

    # Parse field offsets
    fields = {f.name: f for f in msg.fields}
    point_step = msg.point_step
    data = bytes(msg.data)

    x_off = fields['x'].offset
    y_off = fields['y'].offset
    z_off = fields['z'].offset

    n_points = len(data) // point_step
    pts = np.zeros((n_points, 3), dtype=np.float32)

    for i in range(n_points):
        base = i * point_step
        pts[i, 0] = struct.unpack_from('f', data, base + x_off)[0]
        pts[i, 1] = struct.unpack_from('f', data, base + y_off)[0]
        pts[i, 2] = struct.unpack_from('f', data, base + z_off)[0]

    # Remove NaNs and infs
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]


def process_lidar(lidar_msg, min_dist, max_dist, ransac_threshold):
    pts = pointcloud2_to_numpy(lidar_msg)
    if len(pts) == 0:
        return None

    # Filter to forward-facing cone (cable at back = positive X is forward)
    azimuth = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    pts = pts[
        (azimuth > -45) & (azimuth < 45) &   # 90 degree cone forward
        (pts[:, 0] > 0.3) & (pts[:, 0] < 3.0) &  # 0.3m to 3m forward
        (pts[:, 2] > -0.5) & (pts[:, 2] < 2.0)   # reasonable height
    ]

    if len(pts) < 10:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    for attempt in range(5):
        if len(np.asarray(pcd.points)) < 10:
            return None
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=ransac_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        a, b, c, d = plane_model

        # Skip horizontal planes (floor/ceiling)
        if abs(c) > 0.5:
            pcd = pcd.select_by_index(inliers, invert=True)
            continue

        if len(inliers) < 10:
            return None

        board_pts = np.asarray(pcd.select_by_index(inliers).points)
        return board_pts.mean(axis=0)

    return None
  

# ─────────────────────────────────────────────
# 7. SOLVE R AND T (SVD)
# ─────────────────────────────────────────────

def solve_R_t(src, dst):
    """
    Find R, t such that dst ≈ R @ src + t
    src: LiDAR points  (N, 3)
    dst: Camera points (N, 3)
    """
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_dst - R @ mu_src
    return R, t


def compute_reprojection_error(src, dst, R, t):
    """Mean distance between transformed src and dst."""
    transformed = (R @ src.T).T + t
    errors = np.linalg.norm(transformed - dst, axis=1)
    return errors.mean(), errors.std()


# ─────────────────────────────────────────────
# 8. VISUALIZE
# ─────────────────────────────────────────────

def save_results(R, t, points_lidar, points_camera, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save matrices
    np.save(os.path.join(output_dir, 'R.npy'), R)
    np.save(os.path.join(output_dir, 't.npy'), t)

    # Save as text
    with open(os.path.join(output_dir, 'calibration_result.txt'), 'w') as f:
        f.write("=== LiDAR-Camera Extrinsic Calibration Result ===\n\n")
        f.write("Transform: p_camera = R @ p_lidar + t\n\n")
        f.write(f"R (rotation matrix):\n{R}\n\n")
        f.write(f"t (translation vector):\n{t}\n\n")

        # Also write as 4x4 transform matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        f.write(f"T (4x4 homogeneous transform):\n{T}\n\n")

        mean_err, std_err = compute_reprojection_error(points_lidar, points_camera, R, t)
        f.write(f"Reprojection error: {mean_err:.4f} ± {std_err:.4f} meters\n")
        f.write(f"Num poses used: {len(points_lidar)}\n")

    print(f"\n✅ Results saved to {output_dir}/")
    print(f"R =\n{R}")
    print(f"t = {t}")

    mean_err, std_err = compute_reprojection_error(points_lidar, points_camera, R, t)
    print(f"Reprojection error: {mean_err:.4f} ± {std_err:.4f} m")

    # Plot correspondence errors
    fig, ax = plt.subplots(figsize=(10, 4))
    transformed = (R @ points_lidar.T).T + t
    errors = np.linalg.norm(transformed - points_camera, axis=1)
    ax.bar(range(len(errors)), errors)
    ax.set_xlabel('Pose index')
    ax.set_ylabel('Error (m)')
    ax.set_title('Per-pose reprojection error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'errors.png'))
    print(f"Error plot saved.")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    checkerboard = tuple(args.checkerboard)
    square_size = args.square_size
    threshold_ns = int(args.sync_threshold_ms * 1e6)

    print("\n📦 Reading bag...")
    topics = [args.image_topic, args.lidar_topic, args.camera_info_topic]
    data = read_bag(args.bag, topics)

    print("\n📷 Extracting camera intrinsics...")
    if not data[args.camera_info_topic]:
        print("❌ No camera_info messages found!")
        return
    K, dist = extract_intrinsics(data[args.camera_info_topic])

    print("\n🔗 Syncing frames...")
    pairs = sync_frames(data[args.image_topic], data[args.lidar_topic], threshold_ns)

    if len(pairs) < 5:
        print(f"❌ Only {len(pairs)} synced pairs — need at least 5. Check topics or sync threshold.")
        return

    print(f"\n🔍 Processing {len(pairs)} frame pairs...")
    points_camera = []
    points_lidar = []
    failed = 0

    for i, (img_msg, lidar_msg) in enumerate(pairs):
        centroid_cam = process_image(img_msg, checkerboard, square_size, K, dist)
        centroid_lidar = process_lidar(lidar_msg, args.lidar_min_dist, args.lidar_max_dist, args.ransac_threshold)

        if centroid_cam[0] is None:
            failed += 1
            continue
        if centroid_lidar is None:
            failed += 1
            continue

        points_camera.append(centroid_cam[0])
        points_lidar.append(centroid_lidar)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(pairs)} pairs, {len(points_camera)} valid so far...")

    print(f"\n✅ Valid poses: {len(points_camera)}  |  Failed: {failed}")

    if len(points_camera) < 5:
        print("❌ Not enough valid poses. Try:")
        print("   - Better lighting for checkerboard detection")
        print("   - Open area with no competing planes for LiDAR")
        print("   - Adjust --ransac-threshold or --lidar-max-dist")
        return

    points_camera = np.array(points_camera)
    points_lidar = np.array(points_lidar)

    print("\n🧮 Solving for R and t...")
    R, t = solve_R_t(points_lidar, points_camera)

    print("\n💾 Saving results...")
    save_results(R, t, points_lidar, points_camera, args.output)


if __name__ == '__main__':
    main()