import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=None)
parser.add_argument("--gpu", type=str, default="0,")
parser.add_argument("--port", type=str, default="12355")
args = parser.parse_args()
import os
# os.environ["http_proxy"] = "http://127.0.0.1:8899"
# os.environ["https_proxy"] = "http://127.0.0.1:8899"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms


def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
class ReconDataset(Dataset):
    def __init__(self,
                 datapath,
                 reg_datapath=None,
                 caption=None,
                 reg_caption=None,
                 size=512,
                 interpolation="bicubic",
                 flip_p=0.0,
                 aug=True,
                 style=False,
                 caption_target=None,
                 repeat=0.,
                 transform=None,
                 rank=None,
                 worldsize=None
                 ):

        self.aug = aug
        self.repeat = repeat
        self.image_paths1 = []
        for root, dirs, files in os.walk(datapath, followlinks=True):
            for file in files:
                if isimage(file):
                    self.image_paths1.append(os.path.join(root, file))
        self.image_paths1.sort()
        self.image_paths1 = self.image_paths1[rank::worldsize]
        self._length1 = len(self.image_paths1)

        self.labels = {
            "natural": 0,
            "generated": 1,
        }

        self.size = size
        self.interpolation = {# "linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.caption = caption
        self.transform = transform

    def __len__(self):
        return self._length1

    def __getitem__(self, i):
        def construct_example(path):
            example = {}
            image = Image.open(path)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]

            img = img[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]

            image = Image.fromarray(img)
            if self.transform is not None:
                image = self.transform(image)
            
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)
            input_image1 = np.array(image).astype(np.uint8)
            # input_image1 = (input_image1 / 127.5 - 1.0).astype(np.float32)
            # mask = np.ones((self.size // 8, self.size // 8))

            example["image"] = input_image1
            example["mask"] = torch.empty(0)
            example["path"] = path

            return example

        if isinstance(i, slice):
            return [construct_example(path) for path in tqdm(self.image_paths1[i])]
        
        else:
            return construct_example(self.image_paths1[i])


def construct_dataset(datapath, rank, worldsize, size, transform, batch_size=1):
    dataset = ReconDataset(datapath, size=size, transform=transform, rank=rank, worldsize=worldsize)
    # partitioned_dataset = Partition(dataset, rank, worldsize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

# def init_diffusion():
#     captioner_id = "Salesforce/blip-image-captioning-base"
#     processor = BlipProcessor.from_pretrained(captioner_id)
#     model = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float16, low_cpu_mem_usage=False).to("cuda")

#     sd_model_ckpt = "CompVis/stable-diffusion-v1-4"
#     pipeline = StableDiffusionPix2PixZeroPipeline.from_pretrained(
#         sd_model_ckpt,
#         caption_generator=model,
#         caption_processor=processor,
#         torch_dtype=torch.float16,
#         safety_checker=None,
#     ).to("cuda")
#     pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
#     pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
#     # pipeline.enable_model_cpu_offload()
#     return pipeline

def run_diffusion(batch, pipeline, generator):
    captions =[]
    images = []
    latent_images = []
    topil = ToPILImage()
    for i in range(len(batch["image"])):
        image = topil(batch['image'][i].numpy())
        images.append(image)
        # caption = pipeline.generate_caption(image)
        # captions.append(caption)
    # inv_latents = pipeline.invert(captions, image=images, generator=generator).latents
    # inv_images = pipeline.vae.decode(inv_latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    
    # inv_images = inv_images.detach()
    # inv_images = pipeline.image_processor.postprocess(inv_images, output_type="pil")
    # all_embeds = []
    # for each in captions:
    #     source_embeds = pipeline.get_embeds(each, batch_size=1)
    #     all_embeds.append(source_embeds)
    # source_embeds = torch.cat(all_embeds, dim=0)
    # target_embeds = source_embeds
        
    # rec_images = pipeline(
    #                 captions,
    #                 source_embeds=source_embeds,
    #                 target_embeds=target_embeds,
    #                 num_inference_steps=50,
    #                 cross_attention_guidance_amount=0.0,
    #                 generator=generator,
    #                 latents=inv_latents,
    #                 negative_prompt=captions,
    #                 ).images
    for i, each in enumerate(batch["path"]):
        raw_dir = os.path.dirname(each).replace("DiffusionForensics", "data/raw")
        # inv_dir = raw_dir.replace("raw", "inv")
        # rec_dir = raw_dir.replace("raw", "rec")
        os.makedirs(raw_dir, exist_ok=True)
        # os.makedirs(inv_dir, exist_ok=True)
        # os.makedirs(rec_dir, exist_ok=True)
        images[i].save(os.path.join(raw_dir, os.path.basename(each)))
        # inv_images[i].save(os.path.join(inv_dir, os.path.basename(each)))
        # rec_images[i].save(os.path.join(rec_dir, os.path.basename(each)))

def main(rank, worldsize, path=None):
    setup(rank, worldsize)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.2),
    ])
    dataloader = construct_dataset(path, rank=rank, worldsize=worldsize, size=256, transform=transform, batch_size=8)
    # generator = torch.manual_seed(0)
    # pipeline = init_diffusion()
    generator = None
    pipeline = None
    for batch in tqdm(dataloader, total=len(dataloader)):
        run_diffusion(batch, pipeline, generator)

    print("Done!")
    cleanup()

def run_ddp(world_size, path=None):
    mp.spawn(main, args=(world_size, path), nprocs=world_size, join=True)

if __name__ == "__main__":
    world_size = 3
    run_ddp(world_size, args.path)
