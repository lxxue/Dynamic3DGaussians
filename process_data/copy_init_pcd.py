from pathlib import Path

import numpy as np
import open3d as o3d

src_dir = Path("/home/lixin/mount/mocap/containers/GS_tracking/GS_tracking/Take12/output/frames") 
dst_dir = Path("/home/lixin/mount/scratch/chengwei/GS_tracking/mocap_240724_Take12/")

# we have a source point cloud file
pcd = o3d.io.read_point_cloud(str(dst_dir / "sparse_30" / "0" / "points3D.ply"))
env_pcd = o3d.io.read_point_cloud(str(dst_dir / "bg_align.ply"))

vertices = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Create the init_pt_cld array
init_pt_cld = np.hstack((vertices, colors))

# Add a dummy segmentation column (e.g., all ones)
segmentation = np.ones((init_pt_cld.shape[0], 1))
init_pt_cld = np.hstack((init_pt_cld, segmentation))

env_vertices = np.asarray(env_pcd.points)
env_colors = np.asarray(env_pcd.colors)
# env_colors = np.zeros_like(env_vertices)
# initialize to green
# env_colors[:, 1] = 1
print(env_vertices.shape)
print(env_colors.shape)

segmentation = np.zeros((env_vertices.shape[0], 1))
env_pt_cld = np.hstack((env_vertices, env_colors, segmentation))

final_pt_cld = np.vstack((init_pt_cld, env_pt_cld))

# Save the init_pt_cld array in .npz format
np.savez(dst_dir / "init_pt_cld.npz", data=final_pt_cld)