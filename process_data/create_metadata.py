import json
from pathlib import Path

import numpy as np

data_dir = Path("/home/lixin/mount/scratch/chengwei/GS_tracking/mocap_240724_Take12/") 
rgb_cameras = np.load(data_dir / "rgb_cameras.npz")
test_ids = [5, 15, 25, 35, 45]
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
one_image_dir = data_dir / "images" / str(train_cam_ids[0])
for image_path in one_image_dir.iterdir():
    frame_ids.append(int(image_path.stem))
frame_ids.sort()


train_dict = {
    "w": 3004 // 2,   
    "h": 4092 // 2,
    "k": [],
    "w2c": [],
    "cam_ids": [],
    "fn": []
}

test_dict = {
    "w": 3004 // 2,
    "h": 4092 // 2,
    "k": [],
    "w2c": [],
    "cam_ids": [],
    "fn": []
}


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

with open(data_dir / "train_meta.json", "w") as f:
    json.dump(train_dict, f)

with open(data_dir / "test_meta.json", "w") as f:
    json.dump(test_dict, f)
