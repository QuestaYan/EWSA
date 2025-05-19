import copy
import cv2
import heapq
import multiprocessing
import numpy as np
import os
import time
import random
from tqdm import tqdm
import torch

import sys
path_to_model = '/home/msi/yanquanfile/run_code/Real-time_watermark/EWSA/results/2024-05-22-15-16-55/model.pt'
class Encoder:
    def __init__(self):
        pass
    def embed_videos(self, embedded_folder, videos, wms):
        for video_path in videos:
            video_name = video_path.split('/')[-1].split('.')[0]
            embedded_video_path = os.path.join(embedded_folder, f'{video_name}.mp4')
            self.embed_video(wms[video_name], video_path, embedded_video_path)

    def embed_video(self, wm, video_path, output_path):
        ts = time.time()
        encoder, decoder, _, _ = torch.load(path_to_model)
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            # print('wm', wm.shape)
            wm = wm.reshape(1, wm.shape[0])
            wm = torch.tensor(wm).cuda()
            cap = cv2.VideoCapture(video_path)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame_size = (int(width), int(height))
            fps = cap.get(cv2.CAP_PROP_FPS)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, int(fourcc), fps, frame_size)


            count = 0
            # futures = []
            hp = []
            heapq.heapify(hp)
            # out_counter = [0]
            rbar = tqdm(total=length, position=0, desc="Reading")
            half_size = 32
            crop_size = half_size * 2
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    rbar.update(1)
                    #裁剪视频帧，随机取y*y的部分,half_size=y/2
                    max_row = int(height - crop_size)
                    max_col = int(width - crop_size)
                    start_row = random.randint(0, max_row)
                    start_col = random.randint(0, max_col)
                    end_row, end_col = int(height / 2 + half_size), int(width / 2 + half_size)
                    crop_frame = frame[start_row:start_row + crop_size, start_col:start_col + crop_size, :]
                    count, embed_img = Encoder.encode(crop_frame, wm, encoder, decoder, count)
                    frame[start_row:start_row + crop_size, start_col:start_col + crop_size, :] = embed_img
                    out.write(frame)
                    count += 1
                else:
                    break

        cap.release()
        out.release()


    def encode(img, wm, encoder, decoder, count):
        frame = torch.FloatTensor(np.array([img])) / 127.5 - 1.0  # (L, H, W, 3)
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()  # (1, 3, L, H, W)
        wm_frame = encoder(frame, wm)  # (1, 3, L, H, W)嵌入水印操作
        wm_frame = torch.clamp(wm_frame, min=-1.0, max=1.0)
        embed_img = ((wm_frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5).detach().cpu().numpy().astype("uint8")

        return count, embed_img


