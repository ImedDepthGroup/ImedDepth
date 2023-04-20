#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.common import LayerNorm2d
from segment_anything.modeling.image_encoder import Block
from segment_anything import sam_model_registry
from dino_vit import vit_base, vit_small, vit_large, vit_giant2

class ImagePool(nn.Module):
    def __init__(self, in_ch):
        super(ImagePool, self).__init__()
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, in_ch, 1, 1)

    def forward(self, x):
        net = self.gpool(x)
        net = self.conv(net)
        net = F.interpolate(net, size=x.size()[2:], mode="bilinear", align_corners=False)
        return net

class MSConv2d(nn.Module):
    def __init__(self, ch, groups=4):
        super(MSConv2d, self).__init__()
        assert ch % groups == 0
        group_ch = ch // groups
        self.convs = nn.ModuleList([
            nn.Conv2d(group_ch, group_ch, 1, 1)
        ])
        for i in range(1, groups):
            self.convs.append(
                nn.Conv2d(group_ch, group_ch, 3, 1, padding=i, dilation=i, groups=group_ch)
            )
        # self.convs.append(ImagePool(group_ch))
        self.activate = nn.GELU()
        self.norm = nn.BatchNorm2d(ch)
        self.groups = groups

    def forward(self, x):
        features = x.chunk(self.groups, dim=1)
        outs = []
        for i in range(len(features)):
            outs.append(self.convs[i](features[i]))
        net = torch.cat(outs, dim=1)
        net = self.norm(net)
        net = self.activate(net)
        return net

class FFTPrompt(nn.Module):
    def __init__(self, rate=0.25, prompt_type="highpass") -> None:
        super(FFTPrompt, self).__init__()
        assert prompt_type in ["highpass", "lowpass"], "The prompt type must in " \
        "['highpass', 'lowpass'], but got {}".format(prompt_type)
        self.rate = rate
        self.prompt_type = prompt_type
    
    def forward(self, x):
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft)
        h, w = x.shape[2:]
        radio = int((h*w*self.rate)**.5 // 2)
        mask = torch.zeros_like(x)
        c_h, c_w = h // 2, w // 2
        mask[:, :, c_h-radio:c_h+radio, c_w-radio:c_w+radio] = 0
        if self.prompt_type == "highpass":
            fft = fft*(1-mask)
        else:
            fft = fft * mask
        real, imag = fft.real, fft.imag
        shift = torch.fft.fftshift(torch.complex(real, imag))
        inv = torch.fft.ifft2(shift)
        inv = inv.real
        return torch.abs(inv)

class PromptGen(nn.Module):
    def __init__(self, blk, reduction=4, cls_token=False) -> None:
        super(PromptGen, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        prompt_dim = dim // reduction
        self.prompt_learn = nn.Sequential(
            nn.Conv2d(dim, prompt_dim, 1, 1),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            # nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, groups=prompt_dim, bias=False),
            # LayerNorm2d(prompt_dim),
            # nn.GELU(),
            MSConv2d(prompt_dim, groups=4),
            nn.Conv2d(prompt_dim, dim, 1, 1),
            LayerNorm2d(dim),
            nn.Sigmoid()
        )
        self.cls_token = cls_token
    
    def forward(self, x):
        if self.cls_token:
            prompt = self.prompt_learn(x[:,1:])
        else:
            prompt = self.prompt_learn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        promped = x + prompt
        net = self.block(promped)
        return net

class PromptSAM(nn.Module):
    def __init__(self, model_name, checkpoint, num_classes=12, reduction=4, upsample_times=2, groups=4, 
                 prompt_input=False, prompt_type="fft", fft_type="highpass", freq_num=0.25) -> None:
        super(PromptSAM, self).__init__()
        #load same from the pretrained model
        self.sam = sam_model_registry[model_name](checkpoint=checkpoint)
        del self.sam.prompt_encoder
        del self.sam.mask_decoder
        out_dim = self.sam.image_encoder.neck[0].out_channels
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        self.img_size = self.sam.image_encoder.img_size
        blocks = []
        for block in self.sam.image_encoder.blocks:
            blocks.append(
                PromptGen(block, reduction=reduction)
            )
        self.sam.image_encoder.blocks = nn.Sequential(
            *blocks
        )
        self.up_conv = nn.ModuleDict()
        self.up_times = upsample_times
        dim = out_dim
        for i in range(upsample_times):
            self.up_conv["up_{}".format(i+1)] = nn.Sequential(
                    nn.Conv2d(dim, dim // 2, 1, 1, 0),
                    # nn.ConvTranspose2d(dim, dim//2, 2, 2),
                    LayerNorm2d(dim // 2),
                    nn.GELU()
                )
            dim = dim // 2
        self.ms_conv = MSConv2d(dim, groups=groups)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )
        
        if prompt_input:
            if prompt_type == "fft":
                self.prompt_input = FFTPrompt(rate=freq_num, prompt_type=fft_type)
        else:
            self.prompt_input = nn.Identity()

    def upscale(self, x, times=2):
        for i in range(times):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            x = self.up_conv["up_{}".format(i+1)](x)
        return x

    def forward(self, x):
        out = self.sam.image_encoder(x)
        out = self.upscale(out, self.up_times)
        out = self.ms_conv(out)
        seg_out = self.decoder(out)
        seg_out = F.interpolate(seg_out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        return seg_out

DINO_VIT_RESITRY = {
    "vit_b": vit_base,
    "vit_s": vit_small,
    "vit_l": vit_large,
    "vit_g": vit_giant2
}

DINO_CFG = {
    "vit_l":  {
              "patch_size": 14,
              "drop_path_rate": 0.4,
            #   "ffn_layer": "swiglufused",
              "block_chunks": 0,
              "img_size": 518,
              "init_values": 1e-5
        
            },
    "vit_b": {
              "patch_size": 14,
              "drop_path_rate": 0.4,
            #   "ffn_layer": "swiglufused",
              "block_chunks": 0,
              "img_size": 518,
              "init_values": 1e-5
    },
    "vit_g": {
        "patch_size": 14,
        "drop_path_rate": 0.4,
        #   "ffn_layer": "swiglufused",
        "block_chunks": 0,
        "img_size": 518,
        "init_values": 1e-5
    }
}

class PromptDiNo(nn.Module):
    def __init__(self, name, checkpoint=None, reduction=4, num_classes=1) -> None:
        super().__init__()
        cfg = DINO_CFG[name]
        self.encoder = DINO_VIT_RESITRY[name](**cfg)
        self.reset_backbone(checkpoint)
        for param in self.encoder.parameters():
            param.requires_grad = False
        dim = self.encoder.norm.normalized_shape[0]
        # blks = []
        # for blk in self.encoder.blocks:
        #      blks.append(PromptGen(blk, reduction=reduction))
        # self.encoder.blocks = nn.Sequential(*blks)
        self.patch_size = cfg["patch_size"]
        self.img_size = cfg['img_size']
        # self.fc_cls = nn.Linear(1024, num_classes)
        self.out_conv = nn.Conv2d(dim, num_classes, 1, 1, 0)

    
    def reset_backbone(self, chekpoint=None):
        if chekpoint is None:
            return
        state = torch.load(chekpoint, map_location="cpu")
        self.encoder.load_state_dict(state)
    
    def forward(self, x):
        # # 邱老师实现
        # featrues = self.encoder.forward_features(x)
        # feature = featrues["x_norm_patchtokens"]
        # cls_token = featrues["x_norm_clstoken"]
        #
        # _, _, dim = feature.shape
        # feature = feature.reshape(-1, self.img_size // self.patch_size, self.img_size // self.patch_size, dim).permute(0, 3, 1, 2)
        # cls = self.fc_cls(cls_token)
        # out = self.out_conv(feature) * cls.unsqueeze(-1).unsqueeze(-1)
        # out = torch.nn.functional.interpolate(out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        # return out
        # 尝试lin.1
        featrues = self.encoder.forward_features(x)
        feature = featrues["x_norm_patchtokens"]
        cls_token = featrues["x_norm_clstoken"].unsqueeze(1).unsqueeze(1)
        bs, _, dim = feature.shape
        # feature = torch.cat([feature, cls_token], dim=1)

        feature = feature.reshape(bs, self.img_size // self.patch_size, self.img_size // self.patch_size, dim)
        feature = torch.cat([feature, cls_token.expand_as(feature)], dim=-1).permute(0, 3, 1, 2)
        # cls = self.fc_cls(cls_token)
        feature = torch.nn.functional.interpolate(feature, scale_factor=4, mode="bilinear", align_corners=True)
        out = self.out_conv(feature)
        # out = torch.nn.functional.interpolate(out, scale_factor=4, mode="bilinear", align_corners=True)
        return out

if __name__ == "__main__":
    with torch.no_grad():
        # model = PromptSAM("vit_b", "ckpts/sam_vit_b_01ec64.pth").half().cuda()
        x = torch.randn(1, 3, 518, 518).half().cuda()
        cfg = {
              "patch_size": 14,
              "drop_path_rate": 0.4,
            #   "ffn_layer": "swiglufused",
              "block_chunks": 0,
              "img_size": 512,
              "init_values": 1e-5
        }
        model = PromptDiNo("vit_l", "ckpts/dinov2_vitl14_pretrain.pth", 4).half().cuda()

        out = model(x)
        print(out.shape)