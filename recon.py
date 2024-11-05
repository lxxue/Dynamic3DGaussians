import argparse
import cv2
import os
import json
import torch
import numpy as np
import open3d as o3d
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from external import build_rotation
from colormap import colormap
from copy import deepcopy
from train import get_dataset
from tqdm import trange

REMOVE_BACKGROUND = False

def load_scene_data(seq, exp, seg_as_col=False):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg

def render(w, h, k, w2c, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth

def to_cam_open3d(w, h, k, w2c):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w,
        height=h,
        cx=k[0][2],
        cy=k[1][2],
        fx=k[0][0],
        fy=k[1][1]
    )

    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = w2c
    camera.intrinsic = intrinsic

    return camera

def get_depth_edge(depth, ker_size=9, max_depth=None):
    if max_depth is None:
        depth_valid = depth[depth < 10]
        max_depth = depth_valid.max() * 1.1
    depth = np.minimum(depth, max_depth)
    mean = cv2.blur(depth, (ker_size, ker_size))
    mean_seq = mean ** 2
    seq = depth ** 2
    seq_mean = cv2.blur(seq, (ker_size, ker_size))
    var = np.maximum(seq_mean - mean_seq, 0)
    return var




root="/home/lixin/mount/scratch/lixin/GSTAR"
def recon_all_timesteps(exp_name, seq, split):
    print(f"{seq} reconstruct meshes")
    assert split in ["train", "test"]
    scene_data, is_fg = load_scene_data(seq, exp_name)
    recon_dir = f"{root}/{seq}/Dynamic3DGS/meshes"
    os.makedirs(recon_dir, exist_ok=True)
    md = json.load(open(f"{root}/{seq}/Dynamic3DGS/{split}_meta.json", 'r'))
    num_timesteps = len(md['fn'])
    num_cams = len(md['fn'][0])
    w = md['w']
    h = md['h']
    # for cam_id in md['cam_ids'][0]:
    #     os.makedirs(f"{render_dir}/{cam_id}", exist_ok=True)
    w, h = md['w'], md['h']
    for t in trange(num_timesteps):
        frame_id = md['fn'][t][0].split('/')[1].split('.')[0]
        cams, rgbs, depths =  [], [], []
        for c in range(num_cams):
            k, w2c = md['k'][t][c], md['w2c'][t][c]
            cam_o3d = to_cam_open3d(w, h, k, w2c)
            rgb, depth = render(w, h, k, w2c, scene_data[t])
            rgb = rgb.cpu().numpy().transpose(1, 2, 0)
            depth = depth.cpu().numpy().squeeze()
            cams.append(cam_o3d)
            rgbs.append(rgb)
            depths.append(depth)
        # Run TSDF fusion 
        mesh = extract_mesh_fusion(cams, rgbs, depths)
        o3d.io.write_triangle_mesh(f"{recon_dir}/{frame_id}.ply", mesh)
    print(f"{seq} render {split} views done")

def extract_mesh_fusion(cam_list, rgb_list, depth_list,
                        voxel_size=0.008, sdf_trunc=0.02, depth_trunc=6, simplify_face_num=0,
                        mask_backgrond=True, save_dir=None, smooth=False, remove_depth_edge=True):
    """
    Perform TSDF fusion given a fixed depth range, used in the paper.

    voxel_size: the voxel size of the volume
    sdf_trunc: truncation value
    depth_trunc: maximum depth range, should depended on the scene's scales
    mask_backgrond: whether to mask backgroud, only works when the dataset have masks

    return o3d.mesh
    """
    # print("Running tsdf volume integration ...")
    # print(f'voxel_size: {voxel_size}')
    # print(f'sdf_trunc: {sdf_trunc}')
    # print(f'depth_truc: {depth_trunc}')

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # os.makedirs(out_dir, exist_ok=True)
    # save_dir = out_dir
    # if save_dir:
    #     os.makedirs(save_dir, exist_ok=True)

    # cmr_list = to_cam_open3d(cam_list)
    cmr_num = len(cam_list)

    for i in range(cmr_num):
        cam_o3d = cam_list[i]
        rgb = rgb_list[i]
        depth = depth_list[i]
        # rgb = cam_list[cmr_i].original_image.permute(1, 2, 0).cpu().numpy()
        # depth = cam_list[cmr_i].original_depth.permute(1, 2, 0).cpu().numpy()

        # if we have mask provided, use it
        # if mask_backgrond:
        #     bg_mask = depth > 7
        #     # bg_mask = alpha < 0.5
        #     depth[bg_mask] = 0

        # if save_dir:
        #     cv2.imwrite(save_dir + f"color_{cmr_i:06d}.jpg", rgb[..., ::-1] * 255)
        #     render_depth_vis = np.uint8(depth / 10 * 255.0)
        #     render_depth_vis = cv2.applyColorMap(render_depth_vis, cv2.COLORMAP_JET)
        #     cv2.imwrite(save_dir + f"depth_{cmr_i:06d}.jpg", render_depth_vis)
        edge_depth = get_depth_edge(depth, ker_size=3)
        edge_vis = np.minimum(edge_depth / edge_depth.max() * 1000, 1)
        edge_mask = edge_vis > 0.5
        depth[edge_mask] = 0

        # make open3d rgbd
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.asarray(rgb * 255, order="C", dtype=np.uint8)),
            o3d.geometry.Image(np.asarray(depth, order="C")),
            depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )

        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

    mesh = volume.extract_triangle_mesh()
    if smooth:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
    if simplify_face_num > 0:
        mesh = mesh.simplify_quadric_decimation(simplify_face_num)
    return mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True)
    args = parser.parse_args()
    seq = args.seq
    exp_name = "exp_gstar"
    recon_all_timesteps(exp_name, seq, 'train')
    # render_all_timesteps(exp_name, seq, 'test')