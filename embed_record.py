import glob
import json
import numpy as np
import os
import cv2
import random
from tqdm import tqdm
import torch


root_path       = '/home/msi/yanquanfile/run_code/Real-time_watermark/EWSA/'
data_path       = os.path.join(root_path, 'data/MGTV WM')        
output_root     = os.path.join(root_path, 'output')
embedded_folder = os.path.join(output_root, 'embedded')             
json_path       = os.path.join(output_root, 'embed_coords.json')    
patch_size      = 256                                             
path_to_model   = '/home/msi/yanquanfile/run_code/Real-time_watermark/EWSA/results/2024-05-22-15-16-55/model.pt'  


class Encoder:
    def __init__(self):
        encoder, decoder, _, _ = torch.load(path_to_model, map_location='cuda')
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()

    def embed_videos(self, videos, wms, coord_records):
        if not os.path.exists(embedded_folder):
            os.makedirs(embedded_folder)
        else:
            for f in os.listdir(embedded_folder):
                os.remove(os.path.join(embedded_folder, f))

        for video_path in videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            embedded_video_path = os.path.join(embedded_folder, f'{video_name}.mp4')
            self.embed_video(video_path, embedded_video_path,
                             wms[video_name], coord_records)

    def embed_video(self, src_video, dst_video, wm_bits, coord_records):
        cap  = cv2.VideoCapture(src_video)
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps  = cap.get(cv2.CAP_PROP_FPS)
        four = cv2.VideoWriter_fourcc(*"mp4v")
        out  = cv2.VideoWriter(dst_video, four, fps, (w, h))

        frame_idx = 0
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    desc=f'Embedding {os.path.basename(src_video)}')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 随机选择嵌入位置
            max_x = w - patch_size
            max_y = h - patch_size
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            patch = frame[y:y+patch_size, x:x+patch_size].copy()

            with torch.no_grad():
                embed_patch = self._encode_patch(patch, wm_bits)

            frame[y:y+patch_size, x:x+patch_size] = embed_patch
            out.write(frame)

            coord_records.append({
                "video": os.path.basename(dst_video),
                "frame_idx": frame_idx,
                "x": x, "y": y, "w": patch_size, "h": patch_size
            })

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        out.release()

    def _encode_patch(self, patch_bgr, wm_bits):
        # 输入 uint8 BGR -> float32 tensor [-1,1]
        patch_tensor = torch.FloatTensor(patch_bgr).permute(2,0,1).unsqueeze(0).unsqueeze(0)/127.5 - 1.0
        patch_tensor = patch_tensor.cuda()
        wm_tensor    = torch.tensor(wm_bits.reshape(1,-1), dtype=torch.float32, device='cuda')

        wm_out = self.encoder(patch_tensor, wm_tensor)
        wm_out = torch.clamp(wm_out, -1.0, 1.0)
        embed_patch = ((wm_out[0,:,0].permute(1,2,0) + 1.0) * 127.5).cpu().numpy().astype('uint8')
        return embed_patch


def generate_watermarks(videos):
    wms = {}
    for vp in videos:
        vname = os.path.splitext(os.path.basename(vp))[0]
        wms[vname] = np.random.randint(0, 2, 96).astype(np.float32)
    return wms


def main():
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    videos = glob.glob(os.path.join(data_path, '*.mp4'))
    wms    = generate_watermarks(videos)
    coord_records = []

    encoder = Encoder()
    encoder.embed_videos(videos, wms, coord_records)

    with open(json_path, 'w') as f:
        json.dump(coord_records, f, indent=2)
    np.save(os.path.join(output_root, 'wms_bits.npy'), wms)
    print(f'Embedding finished, coords saved to {json_path}')


if __name__ == "__main__":
    main()
