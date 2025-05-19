import os, argparse, torch, numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torchvision.utils import save_image
from model.u2net import U2NETP
from loc_dataset import LocDataset
from tqdm import tqdm

def iou_loss(pred, mask, eps=1.):
    pred = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3)) - inter
    return 1 - (inter+eps)/(union+eps)

def multi_loss(outputs, mask, bce, w_iou):
    losses = []
    for o in outputs:          
        l_bce = bce(o, mask)
        l_iou = iou_loss(o, mask).mean()
        losses.append(l_bce + w_iou*l_iou)
    return torch.mean(torch.stack(losses))

def train(args):
    model = U2NETP(3,1).cuda()
    optim = Adam(model.parameters(), lr=args.lr)
    bce = BCEWithLogitsLoss()
    
    ds = LocDataset(args.data_root, train=True)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    ds_train, ds_val = torch.utils.data.random_split(ds, [train_size, val_size])
    dl_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True, num_workers=4)
    dl_val = DataLoader(ds_val, batch_size=args.bs, shuffle=False, num_workers=4)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        for img, mask in pbar:
            img, mask = img.cuda(), mask.cuda()
            dout, d1, d2, d3 = model(img)
            loss = multi_loss([dout, d1, d2, d3], mask, bce, args.w_iou)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss)
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        if (epoch + 1) % args.save_every == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for img, mask in dl_val:
                    img, mask = img.cuda(), mask.cuda()
                    dout, d1, d2, d3 = model(img)
                    val_loss += float(multi_loss([dout, d1, d2, d3], mask, bce, args.w_iou))
            print(f"Validation Loss: {val_loss / len(dl_val):.4f}")
            model.train()
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), f"{args.ckpt_dir}/locnet_e{epoch+1}.pt")
            with torch.no_grad():
                pred = torch.sigmoid(dout[:1])
                save_image(pred, f"{args.ckpt_dir}/pred_e{epoch+1}.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/msi/yanquanfile/run_code/Real-time_watermark/EWSA/loc_train")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--w_iou", type=float, default=0.1)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--ckpt_dir", default="/home/msi/yanquanfile/run_code/Real-time_watermark/EWSA/loc_ckpt")
    args = ap.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)
