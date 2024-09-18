import os

import cv2
from PIL import Image
from tqdm import tqdm

source_dir = "/home/lixin/mount/mocap/containers/GS_tracking/GS_tracking/Take12/output/bgmasks/"
destination_dir = "/home/lixin/mount/scratch/chengwei/GS_tracking/mocap_240724_Take12/seg"

# Iterate through all files in the source directory
for filename in tqdm(os.listdir(source_dir)):
    if filename.startswith('mask-cam') and filename.endswith('.png'):
        # Extract cam_id and frame_number from the filename
        parts = filename.split('-')
        cam_id = int(parts[1][3:])
        frame_number = int(parts[2].replace('.png', '').replace('f', ''))

        # Create the destination directory if it doesn't exist
        cam_dir = os.path.join(destination_dir, str(cam_id))
        os.makedirs(cam_dir, exist_ok=True)

        # Define the new filename and move the file
        new_filename = f"{frame_number:06d}.png"
        # shutil.copy(os.path.join(source_dir, filename), os.path.join(cam_dir, new_filename))
        # rotate the image by 90 degrees clockwise
        # img = Image.open(os.path.join(source_dir, filename))
        img = cv2.imread(os.path.join(source_dir, filename))
        new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # new_img.save(os.path.join(cam_dir, new_filename))
        cv2.imwrite(os.path.join(cam_dir, new_filename), new_img)