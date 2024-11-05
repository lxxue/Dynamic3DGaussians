import json
from pathlib import Path
import argparse

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True) 

    args = parser.parse_args()
    seq = args.seq

    root = Path("/home/lixin/mount/scratch/lixin/GSTAR/")
    data_dir = root / seq
    rgb_cameras = np.load(data_dir / "cameras" / "rgb_cameras.npz")

    ks = []
    w2cs = []
    cam_ids = []

    for i in range(len(rgb_cameras["ids"])):
        k = rgb_cameras["intrinsics"][i]
        k[:2, :] /= 2
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = rgb_cameras["extrinsics"][i]
        ks.append(k.tolist())
        w2cs.append(w2c.tolist())
        cam_ids.append(int(rgb_cameras["ids"][i]))
    
    frame_ids = []
    one_image_dir = data_dir / "images_2x" / str(cam_ids[0])
    for image_path in one_image_dir.iterdir():
        frame_ids.append(int(image_path.stem))
    frame_ids.sort()

    W = 3004 // 2
    H = 4092 // 2
    all_dict = {
        "w": W,   
        "h": H,
        "k": [],
        "w2c": [],
        "cam_ids": [],
        "fn": []
    }

    print(f"Number of frames: {len(frame_ids)}")
    for id in frame_ids:
        all_dict["k"].append(ks)
        all_dict["w2c"].append(w2cs)
        all_dict["cam_ids"].append(cam_ids)
        train_fns = []
        for cam_id in cam_ids:
            train_fns.append(f"{cam_id}/{id:06d}.jpg")
        all_dict["fn"].append(train_fns)

    d3dgs_dir = data_dir / "Dynamic3DGS"
    d3dgs_dir.mkdir(exist_ok=True)
    print(f"Saving metadata to {d3dgs_dir}")
    with open(d3dgs_dir / "all_meta.json", "w") as f:
        json.dump(all_dict, f)

