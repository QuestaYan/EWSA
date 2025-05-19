
import os
import cv2
import torch
from glob import glob
from random import randint
from torch.utils.data import Dataset, DataLoader
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, root_dir, crop_size, seq_len, max_crop_size=(360, 480)):
        self.seq_len = seq_len
        self.crop_size = crop_size
        self.max_crop_size = max_crop_size
        
        self.videos = []#保存视频地址，以及帧数
        for ext in ["avi", "mp4"]:
            __dir__ = os.path.dirname(__file__)
            for path in glob(os.path.join(__dir__, root_dir, "**/*.%s" % ext), recursive=True):
                cap = cv2.VideoCapture(path)
                nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.videos.append((path, nb_frames))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # Select time index
        path, nb_frames = self.videos[idx]
        # print(path)
        start_idx = randint(0, nb_frames - self.seq_len - 1)
        
        # Select space index
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx-1)
        ok, frame = cap.read()
        H, W, D = frame.shape
        x, dx, y, dy = 0, W, 0, H
        if self.crop_size:
            dy, dx = self.crop_size
            x = randint(0, W-dx-1)
            y = randint(0, H-dy-1)
        if self.max_crop_size[0] < dy:
            dy, dx = self.max_crop_size
            y = randint(0, H-dy-1)
        if self.max_crop_size[1] < dx:
            dy, dx = self.max_crop_size
            x = randint(0, W-dx-1)
        frames = []
        for _ in range(self.seq_len):
            ok, frame = cap.read()
            frame = frame[y:y+dy,x:x+dx]
            frames.append(frame / 127.5 - 1.0)
        x = torch.FloatTensor(np.array(frames))

        x = x.permute(3, 0, 1, 2) 
        return x

def load_train_val(seq_len, batch_size, dataset="Kinetics-600"):
    train = DataLoader(VideoDataset(
        "%s/train" % dataset,
        crop_size=(128, 128),
        seq_len=seq_len,
    ), shuffle=True, num_workers=0, batch_size=batch_size, pin_memory=True)
    val = DataLoader(VideoDataset(
        "%s/val" % dataset, 
        crop_size=False,
        seq_len=seq_len,
    ), shuffle=True, batch_size=1, pin_memory=True)
    return train, val
