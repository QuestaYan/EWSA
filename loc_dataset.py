import os, cv2, random, numpy as np, torch
from torch.utils.data import Dataset

class LocDataset(Dataset):
    def __init__(self, root, train=True):
        self.img_dir  = os.path.join(root, "img")
        self.mask_dir = os.path.join(root, "mask")
        exts = ('.png', '.jpg', '.jpeg')
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(exts)])
        self.train = train

    def __len__(self):
        return len(self.files)

    def _augment(self, img, mask):
        if random.random() < .5:         
            img  = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < .5:           
            img = cv2.blur(img, (3, 3))
        if random.random() < .3:         
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        return img, mask

    def __getitem__(self, idx):
        name = self.files[idx]
        img  = cv2.imread(os.path.join(self.img_dir,  name))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # BGR→RGB
        mask = cv2.imread(os.path.join(self.mask_dir, name), 0).astype('float32')

        if self.train:
            img, mask = self._augment(img, mask)

        # to tensor
        img  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0) / 255.0   # 0/1 float
        return img, mask
