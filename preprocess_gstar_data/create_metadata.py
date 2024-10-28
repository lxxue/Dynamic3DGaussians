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
    test_ids = [4, 14, 24, 34, 44]

    train_k = []
    train_w2c = []
    test_k = []
    test_w2c = []
    train_cam_ids = []
    test_cam_ids = []

    for i in range(len(rgb_cameras["ids"])):
        k = rgb_cameras["intrinsics"][i]
        k[:2, :] /= 2
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = rgb_cameras["extrinsics"][i]
        if i in test_ids:
            test_k.append(k.tolist())
            test_w2c.append(w2c.tolist())
            test_cam_ids.append(int(rgb_cameras["ids"][i]))
        else:
            train_k.append(k.tolist())
            train_w2c.append(w2c.tolist())
            train_cam_ids.append(int(rgb_cameras["ids"][i]))
    
    frame_ids = []
    one_image_dir = data_dir / "images_2x" / str(train_cam_ids[0])
    for image_path in one_image_dir.iterdir():
        frame_ids.append(int(image_path.stem))
    frame_ids.sort()

    W = 3004 // 2
    H = 4092 // 2
    train_dict = {
        "w": W,   
        "h": H,
        "k": [],
        "w2c": [],
        "cam_ids": [],
        "fn": []
    }

    test_dict = {
        "w": W,
        "h": H,
        "k": [],
        "w2c": [],
        "cam_ids": [],
        "fn": []
    }

    print(f"Number of frames: {len(frame_ids)}")
    for id in frame_ids:
        train_dict["k"].append(train_k)
        train_dict["w2c"].append(train_w2c)
        train_dict["cam_ids"].append(train_cam_ids)
        train_fns = []
        for cam_id in train_cam_ids:
            train_fns.append(f"{cam_id}/{id:06d}.jpg")
        train_dict["fn"].append(train_fns)

        test_dict["k"].append(test_k)
        test_dict["w2c"].append(test_w2c)
        test_dict["cam_ids"].append(test_cam_ids)
        test_fns = []
        for cam_id in test_cam_ids:
            test_fns.append(f"{cam_id}/{id:06d}.jpg")
        test_dict["fn"].append(test_fns)

    d3dgs_dir = data_dir / "Dynamic3DGS"
    d3dgs_dir.mkdir(exist_ok=True)
    print(f"Saving metadata to {d3dgs_dir}")
    with open(d3dgs_dir / "train_meta.json", "w") as f:
        json.dump(train_dict, f)

    with open(d3dgs_dir / "test_meta.json", "w") as f:
        json.dump(test_dict, f)



