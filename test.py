import torch
from taming.models.vqgan import VQModel
from my_utils import loadData
from torchvision import utils as vutils
import os

cpkt = torch.load(
    "ckpts/vqgan_imagenet_f16_1024/ckpts/last.ckpt",
    map_location = torch.device("mps")
)

ddconfig = {
    "double_z": False,
    "z_channels": 256,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [ 1,1,2,2,4],  # num_down = len(ch_mult)-1
    "num_res_blocks": 2,
    "attn_resolutions": [16],
    "dropout": 0.0
}

""" ddconfig = {
    "double_z": False,
    "z_channels": 256,
    "resolution": 256,
    "in_channels": 1,
    "out_ch": 1,
    "ch": 128,
    "ch_mult": [ 1,1,2,2,4],  # num_down = len(ch_mult)-1
    "num_res_blocks": 2,
    "attn_resolutions": [16],
    "dropout": 0.0
} """

lossconfig = {
    "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
    "params": {
        "disc_conditional": False,
        "disc_in_channels": 3,
        "disc_start": 250001,
        "disc_weight": 0.8,
        "codebook_weight": 1.0
    }
}

""" lossconfig = {
    "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
    "params": {
        "disc_conditional": False,
        "disc_in_channels": 1,
        "disc_start": 50001,
        "disc_weight": 0.75,
        "codebook_weight": 1.0
    }   
} """
      

vqgan = VQModel(
    ddconfig = ddconfig,
    lossconfig = lossconfig,
    n_embed = 1024,
    embed_dim = 256
)
vqgan.load_state_dict(cpkt["state_dict"])
vqgan.eval()

dataloader = loadData()
for i, imgs in enumerate(dataloader):
    img, emotions = imgs
    decoded_imgs, diff = vqgan(img)
    real_fake_imgs = torch.cat((
        img[:4],
        decoded_imgs.add(1).mul(0.5)[:4]
    ))
    vutils.save_image(
        real_fake_imgs,
        os.path.join("results", f"result_{i}.jpg"),
        nrow = 4
    )