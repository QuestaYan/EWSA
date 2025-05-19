
# import copy
import cv2
import heapq
import numpy as np
import os
import time
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
from tqdm import tqdm
import torch

import sys
# print(torch.cuda.is_available())
torch.cuda.current_device()
torch.cuda._initialized = True
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
            frame_buffer = []#用于存储当前的帧缓冲区
            # futures = []
            hp = []
            heapq.heapify(hp)
            # out_counter = [0]
            rbar = tqdm(total=length, position=0, desc="Reading")
            # wbar = tqdm(total=length, position=1, desc="Writing")


            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    rbar.update(1)
                    count += 1
                    frame_buffer.append(frame / 127.5 - 1.0)
                    if len(frame_buffer) == 8:#一次输入L帧到模型中嵌入水印
                        # print('frame_buffer_size:',np.array(frame_buffer).shape)#(5,720,1280,3)
                        embed_frames = Encoder.encode(np.array(frame_buffer), wm, encoder.cuda(), decoder.cuda())
                        for embed_frame in embed_frames:
                            out.write(embed_frame)#（H,W,3）
                        frame_buffer = []
                else:
                    break
        cap.release()
        out.release()



    def encode(img, wm, encoder, decoder):
        frame = torch.FloatTensor(img)  # (L, H, W, 3)
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()  # (1, 3, L, H, W)
        wm_frame = encoder(frame, wm)  # (1, 3, L, H, W)
        wm_frame = torch.clamp(wm_frame, min=-1.0, max=1.0)

        embed_img = ((wm_frame.squeeze().permute(1, 2, 3, 0) + 1.0) * 127.5).detach().cpu().numpy().astype("uint8")# (L, H, W, 3)

        return embed_img


