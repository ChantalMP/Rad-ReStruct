import numpy as np
import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config, create_transform
from torchvision.transforms import transforms


class ImageEncoderEfficientNet(nn.Module):
    def __init__(self, args):
        super(ImageEncoderEfficientNet, self).__init__()

        self.args = args
        self.model = timm.create_model('tf_efficientnet_b5', pretrained=True)
        config = resolve_data_config({}, model=self.model)
        self.transforms = create_transform(**config)
        if 'radrestruct' in args.data_dir:
            self.transforms.transforms[0] = transforms.Resize((488, 488))

        self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.relu = nn.ReLU()
        self.rescale_conv = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.rescale_pool = nn.AvgPool2d(kernel_size=2, stride=1)

        # transforms
        self.img_tfm = transforms.Compose(self.transforms.transforms[:-1])
        self.norm_tfm = self.transforms.transforms[-1]
        self.resize_size = self.img_tfm.transforms[1].size  # size of CenterCrop

    def forward(self, img, mode='train'):
        x = self.model(img)
        x = self.rescale_conv(x)
        x = self.rescale_pool(x)

        return x.flatten(2).permute(0, 2, 1)


def adapt_position_encoding(model, patch_size=32, after=384, suffix='visual.positional_embedding'):
    keys = [k for k in model if k.endswith(suffix)]
    assert len(keys) == 1
    key = keys[0]
    origin_pos_embed = model[key]
    origin_dim2 = False
    if len(origin_pos_embed.shape) == 2:
        origin_dim2 = True
        origin_pos_embed = origin_pos_embed.unsqueeze(0)
    grid_before = int(np.sqrt(origin_pos_embed.shape[1] - 1))
    before = int(grid_before * patch_size)
    assert (before % patch_size) == 0
    grid_after = after // patch_size
    assert (after % patch_size) == 0
    embed_dim = origin_pos_embed.shape[-1]

    pos_embed = origin_pos_embed[0, 1:, :].reshape((grid_before, grid_before, embed_dim))
    new_size = (grid_after, grid_after)
    pos_embed = torch.nn.functional.interpolate(pos_embed.permute((2, 0, 1)).unsqueeze(0), size=new_size,
                                                mode='bicubic')
    pos_embed = pos_embed.squeeze(0).permute((1, 2, 0)).reshape((-1, embed_dim))
    pos_embed = torch.cat((origin_pos_embed[0, 0:1, :], pos_embed), dim=0).unsqueeze(0)
    assert pos_embed.shape == (1, grid_after * grid_after + 1, embed_dim)
    if origin_dim2:
        assert pos_embed.shape[0] == 1
        pos_embed = pos_embed.squeeze(0)
    model[key] = pos_embed
    return model
