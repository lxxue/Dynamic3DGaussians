from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True) 
    root = Path("/home/lixin/mount/scratch/lixin/GSTAR/")
    args = parser.parse_args()
    seq = args.seq
    data_dir = root / seq
    ply_fname = data_dir / "sparse" / "0" / "points3D.ply"
    pc = o3d.io.read_point_cloud(str(ply_fname))
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)
    # print(points.shape)
    # colors also in the range of [0, 1]^3
    # print(colors.shape)
    # bg_pc = o3d.io.read_point_cloud(str(data_dir / "point_cloud_bg.ply"))
    # bg_points = np.asarray(bg_pc.points)
    # bg_colors = np.asarray(bg_pc.colors)

    # Add a dummy segmentation column (e.g., all ones) 
    # as the point cloud is already segmented
    seg = np.ones((points.shape[0], 1))
    pt_cld = np.hstack((points, colors, seg))
    print("Number of points in the point cloud:", pt_cld.shape[0])
    init_pt_cld = pt_cld
    # bg_seg = np.zeros((bg_points.shape[0], 1))
    # bg_colors = np.zeros_like(bg_points)
    # bg_colors[:, 1] = 1
    # bg_pt_cld = np.hstack((bg_points, bg_colors, bg_seg))
    # print(bg_pt_cld.shape)
    # init_pt_cld = np.vstack((pt_cld, bg_pt_cld))

    d3dgs_dir = data_dir / "Dynamic3DGS"
    d3dgs_dir.mkdir(exist_ok=True)
    np.savez(d3dgs_dir / "init_pt_cld.npz", data=init_pt_cld)
    print(f"Saved init_pt_cld.npz to {data_dir / 'Dynamic3DGS'}")

