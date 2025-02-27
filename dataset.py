import os
import torch, glob
import torchvision.transforms
from torch.utils.data import Dataset
from typing import Literal
from utils.augment import weak_augment, strong_augment
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class InversionDataset(Dataset):
    def __init__(self, real_dir: str, fake_dir: str, mode: Literal["ours", "rgb"], resize: bool, do_weak_aug: bool, do_strong_aug: bool):

        if mode == "ours":
            self.use_original = True
            self.use_inverted = True
            self.use_reconstructed = True
            self.use_residual = True
            self.use_frq = False

        elif mode == "rgb":
            self.use_original = True
            self.use_inverted = False
            self.use_reconstructed = False
            self.use_residual = False
            self.use_frq = False

        elif mode == "noise":
            self.use_original = False
            self.use_inverted = True
            self.use_reconstructed = False
            self.use_residual = False
            self.use_frq = False

        elif mode == "res":
            self.use_original = False
            self.use_inverted = False
            self.use_reconstructed = False
            self.use_residual = True
            self.use_frq = False

        elif mode == "frq":
            self.use_original = False
            self.use_inverted = False
            self.use_reconstructed = False
            self.use_residual = True
            self.use_frq = True

        elif mode == "fire":
            self.use_original = True
            self.use_inverted = False
            self.use_reconstructed = False
            self.use_residual = False
            self.use_frq = False
        
        # load images from DiffusionForensics
        if "lsun" in real_dir:
            self.real_original_images = sorted(glob.glob(os.path.join("data/DiffusionForensics", real_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_inverted_images = sorted(glob.glob(os.path.join("data/inv", real_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_reconstructed_images = sorted(glob.glob(os.path.join("data/rec", real_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_residual_images = sorted(glob.glob(os.path.join("data/res", real_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_frequency_images = sorted(glob.glob(os.path.join("data/frq", real_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
        
        elif "imagenet" in real_dir:
            self.real_original_images = sorted(glob.glob(os.path.join("data/DiffusionForensics", real_dir, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_inverted_images = sorted(glob.glob(os.path.join("data/inv", real_dir, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_reconstructed_images = sorted(glob.glob(os.path.join("data/rec", real_dir, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_residual_images = sorted(glob.glob(os.path.join("data/res", real_dir, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_frequency_images = sorted(glob.glob(os.path.join("data/frq", real_dir, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))

        # load images from self-collected dataset
        else:
            self.real_original_images = sorted(glob.glob(os.path.join("data/fake-inversion", real_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_inverted_images = sorted(glob.glob(os.path.join("fake-inversion/inv", real_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_reconstructed_images = sorted(glob.glob(os.path.join("fake-inversion/rec", real_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_residual_images = sorted(glob.glob(os.path.join("fake-inversion/res", real_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.real_frequency_images = sorted(glob.glob(os.path.join("fake-inversion/frq", real_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
        
        if "lsun" in real_dir:
            self.fake_original_images = sorted(glob.glob(os.path.join("data/DiffusionForensics", fake_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_inverted_images = sorted(glob.glob(os.path.join("data/inv", fake_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_reconstructed_images = sorted(glob.glob(os.path.join("data/rec", fake_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_residual_images = sorted(glob.glob(os.path.join("data/res", fake_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_frequency_images = sorted(glob.glob(os.path.join("data/frq", fake_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
        elif "imagenet" in real_dir:
            self.fake_original_images = sorted(glob.glob(os.path.join("data/DiffusionForensics", fake_dir, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_inverted_images = sorted(glob.glob(os.path.join("data/inv", fake_dir, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_reconstructed_images = sorted(glob.glob(os.path.join("data/rec", fake_dir, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_residual_images = sorted(glob.glob(os.path.join("data/res", fake_dir, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_frequency_images = sorted(glob.glob(os.path.join("data/frq", fake_dir, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
        else:
            self.fake_original_images = sorted(glob.glob(os.path.join("data/fake-inversion", fake_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_inverted_images = sorted(glob.glob(os.path.join("fake-inversion/inv", fake_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_reconstructed_images = sorted(glob.glob(os.path.join("fake-inversion/rec", fake_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_residual_images = sorted(glob.glob(os.path.join("fake-inversion/res", fake_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))
            # self.fake_frequency_images = sorted(glob.glob(os.path.join("fake-inversion/frq", fake_dir, "*.[jJp][pPEn]*[gG]"), recursive=True))

        self.original_images = self.real_original_images + self.fake_original_images
        # self.inverted_images = self.real_inverted_images + self.fake_inverted_images
        # self.reconstructed_images = self.real_reconstructed_images + self.fake_reconstructed_images
        # self.residual_images = self.real_residual_images + self.fake_residual_images
        # self.frequency_images = self.real_frequency_images + self.fake_frequency_images

        # if not (len(self.original_images) == len(self.inverted_images) == len(self.reconstructed_images) == len(self.residual_images)):
        #     raise AssertionError("There are an inconsistent number of images in the original, inverted, reconstructed, and residual directories.")
        
        self.labels = [0] * len(self.real_original_images) + [1] * len(self.fake_original_images)
        self.labels = torch.tensor(self.labels)

        transform_list = []
        if resize:
            transform_list.append(torchvision.transforms.Resize(256, antialias=True))

        if do_weak_aug:
            transform_list.append(torchvision.transforms.Lambda(weak_augment))
        elif do_strong_aug:
            transform_list.append(torchvision.transforms.Lambda(strong_augment))

        transform_list += [
            torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.transform = torchvision.transforms.Compose(transform_list)


    def __len__(self):
        return len(self.original_images)


    def __getitem__(self, idx):

        composite_image = []

        if self.use_original:
            # original_image = read_image(self.original_images[idx]) / 255.
            try:
                original_image = Image.open(self.original_images[idx])
                if not original_image.mode == "RGB":
                    original_image = original_image.convert("RGB")
            except:
                print(f"load image {self.original_images[idx]} failed, skipping...")
                return self[idx+1]
            original_image = self.transform(original_image)
            composite_image.append(original_image)


        # if self.use_inverted:
        #     # inverted_image = read_image(self.inverted_images[idx]) / 255.
        #     try:
        #         inverted_image = Image.open(self.inverted_images[idx])
        #     except:
        #         print(f"load image {self.inverted_image[idx]} failed, skipping...")
        #         return self[idx+1]
        #     inverted_image = self.transform(inverted_image)
        #     composite_image.append(inverted_image)


        # if self.use_reconstructed:
        #     # reconstructed_image = read_image(self.reconstructed_images[idx]) / 255.
        #     try:
        #         reconstructed_image = Image.open(self.reconstructed_images[idx])
        #     except:
        #         print(f"load image {self.reconstructed_images[idx]} failed, skipping...")
        #         return self[idx+1]
        #     reconstructed_image = self.transform(reconstructed_image)
        #     composite_image.append(reconstructed_image)


        # if self.use_residual:
        #     # residual_image = read_image(self.residual_images[idx]) / 255.
        #     try:
        #         residual_image = Image.open(self.residual_images[idx])
        #     except:
        #         print(f"load image {self.residual_images[idx]} failed, skipping...")
        #         return self[idx+1]
        #     residual_image = self.transform(residual_image)
        #     composite_image.append(residual_image)

        # if self.use_frq:
        #     # residual_image = read_image(self.residual_images[idx]) / 255.
        #     try:
        #         frequency_image = Image.open(self.frequency_images[idx])
        #     except:
        #         print(f"load image {self.frequency_images[idx]} failed, skipping...")
        #         return self[idx+1]
        #     frequency_image = self.transform(frequency_image)
        #     composite_image.append(frequency_image)

        composite_image = torch.cat(composite_image, dim=-3)

        # assert composite_image.shape == (12, 224, 224)

        label = self.labels[idx]

        return composite_image, label