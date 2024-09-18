from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

src_dir = Path("/home/lixin/mount/mocap/containers/GS_tracking/GS_tracking/Take12/output/frames") 
dst_dir = Path("/home/lixin/mount/scratch/chengwei/GS_tracking/mocap_240724_Take12/")
# Read the mesh file
first_frame_id = 31
# mesh = trimesh.load(str(src_dir / f"mesh-f{first_frame_id:05d}.obj"))
mesh = o3d.io.read_triangle_mesh(str(src_dir / f"mesh-f{first_frame_id:05d}.obj"))
mesh.compute_vertex_normals()

# Load the texture image
texture_image = Image.open(str(src_dir / f"atlas-f{first_frame_id:05d}.png"))
texture_image = np.asarray(texture_image)


# Get the UV coordinates from the mesh
# Note: Open3D stores UV coordinates for triangles (faces), not directly for vertices
triangle_uvs = np.asarray(mesh.triangle_uvs)  # UV coordinates for each triangle vertex
triangle_indices = np.asarray(mesh.triangles)  # Vertex indices for each triangle

# Now, we need to map triangle UVs to each vertex to get vertex color

# Function to get the color from the texture using UV coordinates
def get_color_from_uv(uv, texture_image):
    h, w, _ = texture_image.shape
    # UVs are in [0, 1], scale to image dimensions
    pixel_x = min(int(uv[0] * (w - 1)), w - 1)
    pixel_y = min(int((1 - uv[1]) * (h - 1)), h - 1)  # Invert y-axis for image
    return texture_image[pixel_y, pixel_x]

# Create an array to store vertex colors
vertex_colors = np.zeros((len(mesh.vertices), 3))

# Loop through each triangle and get the colors for the vertices
for tri_idx, tri_verts in enumerate(triangle_indices):
    for i in range(3):  # Each triangle has 3 vertices
        uv = triangle_uvs[tri_idx * 3 + i]  # Get UV coordinate for the vertex
        color = get_color_from_uv(uv, texture_image)  # Get the color from the texture
        vertex_colors[tri_verts[i]] += color / 255.0  # Normalize to [0, 1] and sum for smoothing

vertex_colors /= 3  # Average the colors for each vertex

# create a point cloud with vertex colors
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
pcd.colors = o3d.utility.Vector3dVector(vertex_colors / 255.0)


# Extract vertices and vertex colors
vertices = np.asarray(mesh.vertices)
colors = np.asarray(vertex_colors, dtype=float)


# Ensure colors are in the range [0, 1]
if colors.max() > 1.0:
    colors /= 255.0

# Create the init_pt_cld array
init_pt_cld = np.hstack((vertices, colors))

# Add a dummy segmentation column (e.g., all ones)
segmentation = np.ones((init_pt_cld.shape[0], 1))
init_pt_cld = np.hstack((init_pt_cld, segmentation))

# Save the init_pt_cld array in .npz format
np.savez(dst_dir / "init_pt_cld.npz", data=init_pt_cld)