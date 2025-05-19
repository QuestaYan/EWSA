import glob
import numpy as np
import os
import shutil
import time
from embed import Encoder


def generate_watermarks(videos):
    # You can generate the watermark with other ways, 
    # but you cannot use a fixed watermark.
    
    wms = {}
    for video_path in videos:
        wm = np.random.randint(0, 2, 96)#生成随机水印，最后一维是水印长度
        video_name = video_path.split('/')[-1].split('.')[0]
        wms[video_name] = wm

    return wms

def process_encode():
    video_fps = 25
    video_length = 30
    encoder = Encoder()
    
    videos = glob.glob(data_path+'/*')
    total_time_cost = 0

    ts = time.time()
    wms = generate_watermarks(videos)

    encoder.embed_videos(embedded_folder, videos, wms)
    total_time_cost = time.time()-ts

    avg_emb_fps = (len(videos) * video_fps * video_length) / total_time_cost
    print('avg_emb_fps:',avg_emb_fps)
    wms['avg_emb_fps'] = round(avg_emb_fps, 4)
    np.save(output_wms, wms)
    print('embedded done')


if __name__ == "__main__":
    root_path = '/home/msi/yanquanfile/run_code/Real-time_watermark/EWSA'
    data_path = os.path.join(root_path, '/data/MGTV WM')
    output_root = 'output'
    output_wms = 'output/wms.npy'
    embedded_folder = f'{output_root}/embedded'

    if not os.path.exists(embedded_folder):
        os.makedirs(embedded_folder)
    else:
        shutil.rmtree(embedded_folder)
        os.makedirs(embedded_folder)

    process_encode()