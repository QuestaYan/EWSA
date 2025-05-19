import os
import cv2
import json
import numpy as np
from tqdm import tqdm

embedded_folder = '/home/msi/yanquanfile/run_code/Real-time_watermark/EWSA/output/embedded'
json_path       = '/home/msi/yanquanfile/run_code/Real-time_watermark/EWSA/output/embed_coords.json'
save_root       = '/home/msi/yanquanfile/run_code/Real-time_watermark/EWSA/loc_train'


def ensure_dirs():
    img_dir  = os.path.join(save_root, 'img')
    mask_dir = os.path.join(save_root, 'mask')
    for d in (img_dir, mask_dir):
        if not os.path.exists(d):
            os.makedirs(d)
    return img_dir, mask_dir

def generate_dataset():
    img_dir, mask_dir = ensure_dirs()

    with open(json_path, 'r') as f:
        coords = json.load(f)

    # group by video for sequential reading
    video_map = {}
    for rec in coords:
        video_map.setdefault(rec['video'], []).append(rec)

    for vid, recs in video_map.items():
        recs_sorted = sorted(recs, key=lambda r: r['frame_idx'])
        vpath = os.path.join(embedded_folder, vid)
        cap = cv2.VideoCapture(vpath)

        cur_frame = 0
        pbar = tqdm(recs_sorted, desc=f'Extracting {vid}')
        for r in pbar:
            target_idx = r['frame_idx']
            while cur_frame <= target_idx:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f'Frame {target_idx} missing in {vid}')
                if cur_frame == target_idx:
                    h, w = frame.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    x, y, pw, ph = r['x'], r['y'], r['w'], r['h']
                    mask[y:y+ph, x:x+pw] = 255

                    name = f"{os.path.splitext(vid)[0]}_{cur_frame:04d}.png"
                    cv2.imwrite(os.path.join(img_dir,  name), frame)
                    cv2.imwrite(os.path.join(mask_dir, name), mask)
                cur_frame += 1
        cap.release()

    print(f'Dataset saved to {save_root}/img and {save_root}/mask')

if __name__ == "__main__":
    generate_dataset()
