from pathlib import Path

import numpy as np
import open3d as o3d

src_dir = Path("/home/lixin/mount/mocap/containers/GS_tracking/GS_tracking/Take12/output/frames") 
dst_dir = Path("/home/lixin/mount/scratch/chengwei/GS_tracking/mocap_240724_Take12/")

# we have a source point cloud file
pcd = o3d.io.read_point_cloud(str(dst_dir / "sparse_30" / "0" / "points3D.ply"))

vertices = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Create the init_pt_cld array
init_pt_cld = np.hstack((vertices, colors))

# Add a dummy segmentation column (e.g., all ones)
segmentation = np.ones((init_pt_cld.shape[0], 1))
init_pt_cld = np.hstack((init_pt_cld, segmentation))

# Save the init_pt_cld array in .npz format
np.savez(dst_dir / "init_pt_cld.npz", data=init_pt_cld)