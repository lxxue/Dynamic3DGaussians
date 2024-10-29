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


root="/home/lixin/mount/scratch/lixin/GSTAR"
def render_all_timesteps(exp_name, seq, split):
    print(f"{seq} render {split} views")
    assert split in ["train", "test"]
    scene_data, is_fg = load_scene_data(seq, exp_name)
    os.makedirs(f"{root}/{seq}/Dynamic3DGS/renders/", exist_ok=True)
    md = json.load(open(f"{root}/{seq}/Dynamic3DGS/{split}_meta.json", 'r'))
    num_timesteps = len(md['fn'])
    num_cams = len(md['fn'][0])
    w = md['w']
    h = md['h']
    for cam_id in md['cam_ids'][0]:
        os.makedirs(f"{root}/{seq}/Dynamic3DGS/renders/{cam_id}", exist_ok=True)
    for t in trange(num_timesteps):
        for c in range(num_cams):
            w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
            im, depth = render(w, h, k, w2c, scene_data[t])
            im.clamp_(0.0, 1.0)
            im = im.cpu().numpy().transpose(1, 2, 0)
            im = (im * 255).astype(np.uint8)
            fn = md['fn'][t][c]
            # print(f"./renders/{exp_name}/{seq}/{split}/{fn}")
            cv2.imwrite(f"{root}/{seq}/Dynamic3DGS/renders/{fn}", im[:, :, ::-1])
    print(f"{seq} render {split} views done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True)
    args = parser.parse_args()
    seq = args.seq
    exp_name = "exp_gstar"
    render_all_timesteps(exp_name, seq, 'train')
    render_all_timesteps(exp_name, seq, 'test')