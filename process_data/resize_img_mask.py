from pathlib import Path

from PIL import Image
from tqdm import tqdm

data_dir = Path("/home/lixin/mount/scratch/chengwei/GS_tracking/mocap_240724_Take12/")
img_dir = data_dir / "images_full_res"
mask_dir = data_dir / "seg_full_res"
dst_img_dir = data_dir / "images"
dst_img_dir.mkdir(exist_ok=True)
dst_mask_dir = data_dir / "seg"
dst_mask_dir.mkdir(exist_ok=True)

for cam_dir in tqdm(img_dir.iterdir()):
    cam_id = int(cam_dir.name)
    (dst_img_dir / str(cam_id)).mkdir(exist_ok=True)
    for img_path in cam_dir.iterdir():
        img = Image.open(img_path)
        h, w = img.size
        img = img.resize((h // 2, w // 2))
        img.save(dst_img_dir / str(cam_id) / img_path.name)

for cam_dir in tqdm(mask_dir.iterdir()):
    cam_id = int(cam_dir.name)
    (dst_mask_dir / str(cam_id)).mkdir(exist_ok=True)
    for mask_path in cam_dir.iterdir():
        mask = Image.open(mask_path)
        h, w = mask.size
        mask = mask.resize((h // 2, w // 2))
        mask.save(dst_mask_dir / str(cam_id) / mask_path.name)