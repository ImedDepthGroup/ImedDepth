#coding:utf-8
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt
from PIL import Image
import argparse
import numpy as np
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip
import glob
import os
import re
# from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence




parser = argparse.ArgumentParser("Learnable prompt")
parser.add_argument("--image", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument("--test_img", type=str, required=True,
                    help="path to the image that used to test the model")
parser.add_argument("--test_mask", type=str, required=True,
                    help="path to the mask file for testing")
parser.add_argument("--epoch", type=int, default=1,
                    help="training epochs")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam")
parser.add_argument("--model_name", default="default", type=str,
                    help="name of the sam model, default is vit_h",
                    choices=["default", "vit_b", "vit_l", "vit_h", 'vit_g'])
parser.add_argument("--save_path", type=str, default="./ckpt_prompt",
                    help="save the weights of the model")
parser.add_argument("--num_classes", type=int, default=256)
parser.add_argument("--mix_precision", action="store_true", default=False,
                    help="whether use mix precison training")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--optimizer", default="adamw", type=str,
                    help="optimizer used to train the model")
parser.add_argument("--weight_decay", default=5e-4, type=float, 
                    help="weight decay for the optimizer")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="momentum for the sgd")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--divide", action="store_true", default=False,
                    help="whether divide the mask")
parser.add_argument("--divide_value", type=int, default=255, 
                    help="divided value")
parser.add_argument("--num_workers", "-j", type=int, default=1, 
                    help="divided value")
parser.add_argument("--device", default="0", type=str)
parser.add_argument("--model_type", default="sam", choices=["dino", "sam"], type=str,
                    help="backbone type")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

from learnerable_seg import PromptSAM, PromptDiNo
from scheduler import PolyLRScheduler
from metrics.metric import Metric

class SegDataset:
    def __init__(self, img_paths, mask_paths, 
                mask_divide=False, divide_value=255,
                pixel_mean=[0.5]*3, pixel_std=[0.5]*3,
                img_size=518) -> None:
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.length = len(img_paths)
        self.mask_divide = mask_divide
        self.divide_value = divide_value
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.img_size = img_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img)
        mask = Image.open(mask_path).convert("L")
        mask = np.asarray(mask)
        if self.mask_divide:
            mask = mask // self.divide_value
        transform = Compose(
            [
                ColorJitter(),
                VerticalFlip(),
                HorizontalFlip(),
                Resize(self.img_size, self.img_size),
                Normalize(mean=self.pixel_mean, std=self.pixel_std)
            ]
        )
        aug_data = transform(image=img, mask=mask)
        x = aug_data["image"]
        target = aug_data["mask"]
        if img.ndim == 3:
            x = np.transpose(x, axes=[2, 0, 1])
        elif img.ndim == 2:
            x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x), torch.from_numpy(target)

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)
        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

def main(args):
    img_path = args.image
    mask_path = args.mask_path
    test_img = args.test_img
    test_mask = args.test_mask
    epochs = args.epoch
    checkpoint = args.checkpoint
    model_name = args.model_name
    save_path = args.save_path
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    lr = args.lr
    momentum = args.momentum
    bs = args.batch_size
    divide = args.divide
    divide_value = args.divide_value
    num_workers = args.num_workers
    model_type = args.model_type
    # pixel_mean=[123.675, 116.28, 103.53],
    # pixel_std=[58.395, 57.12, 57.375],
    # pixel_mean = np.array(pixel_mean) / 255
    # pixel_std = np.array(pixel_std) / 255
    pixel_mean = [0.5]*3
    pixel_std = [0.5]*3
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_classes = args.num_classes
    basename = os.path.basename(img_path)
    _, ext = os.path.splitext(basename)
    if ext == "":
        regex = re.compile(".*\.(jpe?g|png|gif|tif|bmp)$", re.IGNORECASE)
        img_paths = [file for file in glob.glob(os.path.join(img_path, "*.*")) if regex.match(file)]
        print("train with {} imgs".format(len(img_paths)))
        mask_paths = [os.path.join(mask_path, os.path.basename(file)) for file in img_paths]
        # mask_paths = [os.path.join(mask_path, 'depth'+os.path.basename(file)[6:]) for file in img_paths]
    else:
        bs = 1
        img_paths = [img_path]
        mask_paths = [mask_path]
        num_workers = 1
    # add test paths
    testname = os.path.basename(test_img)
    _, ext = os.path.splitext(testname)
    if ext == "":
        regex = re.compile(".*\.(jpe?g|png|gif|tif|bmp)$", re.IGNORECASE)
        test_imgs = [file for file in glob.glob(os.path.join(test_img, "*.*")) if regex.match(file)]
        print("test with {} imgs".format(len(test_imgs)))
        test_masks = [os.path.join(test_mask, os.path.basename(file)) for file in test_imgs]
        # mask_paths = [os.path.join(mask_path, 'depth'+os.path.basename(file)[6:]) for file in img_paths]
    else:
        bs = 1
        test_imgs = [test_img]
        test_masks = [test_mask]
        num_workers = 1
    if model_type == "sam":
        model = PromptSAM(model_name, checkpoint=checkpoint, num_classes=num_classes, reduction=4, upsample_times=4, groups=8)
    elif model_type == "dino":
        model = PromptDiNo(name=model_name, checkpoint=checkpoint, num_classes=num_classes)
    dataset = SegDataset(img_paths, mask_paths=mask_paths, mask_divide=divide, divide_value=divide_value,
                        pixel_mean=pixel_mean, pixel_std=pixel_std)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    testset = SegDataset(test_imgs, mask_paths=test_masks, mask_divide=divide, divide_value=divide_value,
                        pixel_mean=pixel_mean, pixel_std=pixel_std)
    testloader = DataLoader(testset, batch_size=bs, shuffle=True, num_workers=num_workers)
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if optimizer == "adamw":
        optim = opt.AdamW([{"params":model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = opt.SGD([{"params":model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    loss_func = nn.CrossEntropyLoss()
    scheduler = PolyLRScheduler(optim, num_images=len(img_paths), batch_size=bs, epochs=epochs)
    metric = Metric(num_classes=num_classes)
    best_mse = 1e10
    for epoch in range(epochs):
        metric.reset()
        for i, (x, target) in enumerate(dataloader):
            x = x.to(device)
            target = target.to(device, dtype=torch.long)
            optim.zero_grad()
            if device_type == "cuda" and args.mix_precision:
                x = x.to(dtype=torch.float16)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    pred = model(x)
                    loss = loss_func(pred, target)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                x = x.to(detype=torch.float32)
                pred = model(x)
                loss = loss_func(pred, target)
                loss.backward()
                optim.step()
            metric.update(torch.argmax(torch.softmax(pred, dim=1),dim=1).to(device), target)
            print("epoch:{}-{}: loss:{}".format(epoch+1, i+1, loss.item()))
            scheduler.step()
        # mse = metric.evaluate()["MSE"].numpy()
        mse = np.array(metric.evaluate()["MSE"])
        print("epoch-{}: mse:{}".format(epoch, mse.item()))
        if mse < best_mse:
            best_mse = mse
            torch.save(
                model.state_dict(), os.path.join(save_path, "dino_{}_prompt.pth".format(model_name))
            )
    #test
    model.eval()
    metric.reset()
    for i, (x, target) in enumerate(testloader):
        x = x.to(device)
        target = target.to(device, dtype=torch.long)
        with torch.no_grad():
            if device_type == "cuda" and args.mix_precision:
                x = x.to(dtype=torch.float16)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    pred = model(x)
            else:
                x = x.to(detype=torch.float32)
                pred = model(x)
            np.savez(save_path, img = x.cpu().numpy(), pred = pred.cpu().numpy())
            metric.update(torch.argmax(torch.softmax(pred, dim=1),dim=1), target)
            # metrics = metric.evaluate().numpy()
            metrics = np.array(metric.evaluate())
            np.savetxt(save_path+"/prediction_metric.txt", metrics)


if __name__ == "__main__":
    
    main(args)
