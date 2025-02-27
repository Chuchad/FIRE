import torch.nn as nn
from torch import Tensor
from typing import Literal
import math
import torch, torchvision, diffusers
from typing import *
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents, 
)


def get_resnet_model(mode: Literal["rgb", "ours"], norm_layer: Literal["batch", "instance"], pretrained: bool) -> nn.Module:

    if norm_layer == "batch":
        norm = nn.BatchNorm2d
    elif norm_layer == "instance":
        norm = nn.InstanceNorm2d
    else:
        raise AssertionError("Unknown norm layer")

    if norm == nn.InstanceNorm2d:
        model = torchvision.models.resnet50(num_classes=1000, pretrained=False, norm_layer=norm)
        model.load_state_dict(torchvision.models.ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True), strict=False)
    else:
        model = torchvision.models.resnet50(num_classes=1000, weights="IMAGENET1K_V2", norm_layer=norm)

    if mode == "ours":
        model.conv1.weight = nn.Parameter(torch.cat([model.conv1.weight * 0.25] * 4, dim=1))
        model.conv1.in_channels = 12

    model.fc = nn.Linear(2048, 1)
    torch.nn.init.normal_(model.fc.weight.data, 0.0, 0.02)

    return model

def get_frq_resnet_model(mode: Literal["rgb", "ours", "frq"], norm_layer: Literal["batch", "instance"], pretrained: bool) -> nn.Module:

    if norm_layer == "batch":
        norm = nn.BatchNorm2d
    elif norm_layer == "instance":
        norm = nn.InstanceNorm2d
    else:
        raise AssertionError("Unknown norm layer")

    if norm == nn.InstanceNorm2d:
        model = torchvision.models.resnet50(num_classes=1000, pretrained=False, norm_layer=norm)
        model.load_state_dict(torchvision.models.ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True), strict=False)
    else:
        model = torchvision.models.resnet50(num_classes=1000, weights="IMAGENET1K_V2", norm_layer=norm)

    if mode == "frq":
        model.conv1.weight = nn.Parameter(torch.cat([model.conv1.weight * 0.25] * 2, dim=1))
        model.conv1.in_channels = 6

    model.fc = nn.Linear(2048, 1)
    torch.nn.init.normal_(model.fc.weight.data, 0.0, 0.02)

    return model

class ESPCN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            upscale_factor: int,
    ) -> None:
        super(ESPCN, self).__init__()
        hidden_channels = channels // 2
        out_channels = int(out_channels * (upscale_factor ** 2))
        self.bn = nn.BatchNorm2d(in_channels)
        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # Sub-pixel convolution layer
        self.sub_pixel_0 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )

        self.sub_pixel_1 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )

        # Initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                    nn.init.zeros_(module.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.bn(x)
        x = self.feature_maps(x)
        x_h = self.sub_pixel_0(x)
        x_l = self.sub_pixel_1(x)

        return x_h, x_l

def create_vae():
    # create SD pipeline as usual
    pipe = diffusers.StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                             torch_dtype=torch.float16, safety_checker=None)
    vae = pipe.vae
    del pipe
    torch.cuda.empty_cache()

    return vae

class FIRE_model(nn.Module):
    def __init__(self, mode="frq", norm_layer="instance", pretrained=True, radiuslow=40, radiushigh=120):
        super(FIRE_model, self).__init__()
        # initialize VAE in the DM
        self.vae = create_vae()
        self.decode_dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype

        for param in self.vae.parameters():
            param.requires_grad = False

        # backend classifier
        self.resnet = get_frq_resnet_model(mode=mode, norm_layer=norm_layer, pretrained=pretrained)
        # FMRE module
        self.fft_filter_module = fft_filter(radiuslow=radiuslow, radiushigh=radiushigh, rows=256, cols=256)
        self.vae = self.vae.to("cuda")
        self.resnet = self.resnet.to("cuda")
        self.fft_filter_module = self.fft_filter_module.to("cuda")

    def forward(self, x):
        middle_freq_image, middle_filtered_image, mask_mid_frq, mask_mid_filterd = self.fft_filter_module(x)

        latents_x = retrieve_latents(self.vae.encode(x.to(self.decode_dtype)))
        reconstructions_x = self.vae.decode(
                            latents_x.to(self.decode_dtype),return_dict=False
                        )[0]

        latents_middle_filtered = retrieve_latents(self.vae.encode(middle_filtered_image.to(self.decode_dtype)))
        reconstructions_middle_filtered = self.vae.decode(
                            latents_middle_filtered.to(self.decode_dtype),return_dict=False
                        )[0]

        # o = self.resnet(torch.cat(reconstructions_filterd))
        raw_reconstructions_delta = torch.abs(reconstructions_x - x)
        filtered_reconstructions_delta = torch.abs(reconstructions_middle_filtered - middle_filtered_image)
        o = self.resnet(torch.cat([raw_reconstructions_delta, filtered_reconstructions_delta], dim=1))

        return o, middle_freq_image, raw_reconstructions_delta, mask_mid_frq, mask_mid_filterd


class fft_filter(nn.Module):
    def __init__(self, radiuslow=35, radiushigh=120, rows=256, cols=256):
        super(fft_filter, self).__init__()
        self.radiuslow = radiuslow
        self.radiushigh = radiushigh
        self.rows = rows
        self.cols = cols
        # preset masks M_{mid} and M_{mid_c}
        i_mask, r_i_mask = self.init_mask()
        i_mask, r_i_mask = nn.Parameter(i_mask, requires_grad=False), nn.Parameter(r_i_mask, requires_grad=False)
        self.register_buffer('i_mask', i_mask)
        self.register_buffer('r_i_mask', r_i_mask)
        # # encode and decode frequency mask
        self.mask_autoencoder = ESPCN(in_channels=3, out_channels=1, channels=64, upscale_factor=1)
    
    def init_mask(self):
        mask = torch.ones((1, self.rows, self.cols), dtype=torch.float32, requires_grad=False)
        crow, ccol = self.rows // 2 , self.cols // 2
        center = [crow, ccol]
        x, y = torch.meshgrid(torch.arange(self.rows), torch.arange(self.cols), indexing='ij')
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 < self.radiuslow*self.radiuslow
        mask[:, mask_area] = 0
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 >= self.radiushigh*self.radiushigh
        mask[:, mask_area] = 0
        
        return mask, 1-mask
    

    # implement FMRE
    def middle_pass_filter(self, image):
        freq_image = torch.fft.fftn(image * 255, dim=(-2, -1))
        freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))
        mask_mid_frq, mask_mid_filterd = self.mask_autoencoder((20*torch.log(torch.abs(freq_image)+1e-7))/255)
        mask_mid_frq = mask_mid_frq.to(freq_image.dtype)

        middle_freq = freq_image * mask_mid_frq
        middle_freq = torch.fft.ifftshift(middle_freq, dim=(-2, -1))
        masked_image_array = torch.fft.ifftn(middle_freq, dim=(-2, -1))
        z = torch.abs(masked_image_array)
        _min = torch.min(z)
        _max = torch.max(z)
        z = (z)/(_max-_min)
        middle_freq_image = z

        mask_mid_filterd = mask_mid_filterd.to(freq_image.dtype)
        middle_filtered = freq_image * mask_mid_filterd
        middle_filtered = torch.fft.ifftshift(middle_filtered, dim=(-2, -1))
        middle_filtered_array = torch.fft.ifftn(middle_filtered, dim=(-2, -1))
        z = torch.abs(middle_filtered_array)
        _min = torch.min(z)
        _max = torch.max(z)
        z = (z)/(_max-_min)
        middle_filtered_image = z

        return middle_freq_image, middle_filtered_image, mask_mid_frq.to(torch.float32), mask_mid_filterd.to(torch.float32)
    

    def forward(self, image):
        middle_freq_image, middle_filtered_image, mask_mid_frq, mask_mid_filterd = self.middle_pass_filter(image)

        return middle_freq_image, middle_filtered_image, mask_mid_frq, mask_mid_filterd


if __name__ == "__main__":
    # model = ESPCN(3, 1, 64, 2)
    # fft_filter = fft_filter()
    model = FIRE_model()
    x = torch.randn(4, 3, 256, 256)
    y = model(x)