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

def load_gt_tracking(seq):
    keypoints_fname = f"{root}/{seq}/keypoints-3d.npz"
    keypoints = np.load(keypoints_fname, allow_pickle=True)
    # key 0, 1, 2, 3, 4, 5 as we have 6 apriltags
    # for each apriltag, we have 5 keypoints, stored in homogenous coordinates
    # therefore, we have 6 * (5,4) arrays
    gt_keypoints_first_frame = [keypoints[str(i)][()]["keypoints_3d"][0] for i in range(6)]
    return gt_keypoints_first_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True)
    args = parser.parse_args()
    seq = args.seq
    exp_name = "exp_gstar"
    root = "/home/lixin/mount/scratch/lixin/GSTAR"
    scene_data = np.load(f"./output/{exp_name}/{seq}/params.npz")
    pred_points3d_all = scene_data["means3D"]
    num_frames = int(pred_points3d_all.shape[0])
    gt_points3d = load_gt_tracking(seq)

    pred_points3d = np.zeros((6, 5, num_frames, 3), dtype=np.float32)
    pred_points3d_first_frame = pred_points3d_all[0]
    print("Processing sequence", seq)
    for i in range(6):
        print("Processing apriltag", i)
        for j in range(5):
            gt_point = gt_points3d[i][j]
            gt_point = gt_point[:3] / gt_point[3]
            # compute distance
            dist = np.linalg.norm(pred_points3d_first_frame - gt_point[None, :], axis=1)
            # get index of the minimum distance
            k = np.argmin(dist)
            # print(gt_point, pred_points3d_all[0, k])

            pred_points3d[i, j] = pred_points3d_all[:, k, :]
            # exit(0)

    np.save(f"{root}/{seq}/Dynamic3DGS/pred_points3d.npy", pred_points3d)

