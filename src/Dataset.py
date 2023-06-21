import glob
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import RandomCrop, Resize, ToTensor


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=False) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.input_images = sorted(glob.glob(img_dir + "/*.png"))
        self.output_images = sorted(glob.glob(img_dir + "/*.jpg"))
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # input_image = read_image(self.input_images[idx])

        input_image = self.totensor(Image.open(self.input_images[idx]).convert("RGB"))
        output_image = self.totensor(Image.open(self.output_images[idx]).convert("RGB"))
        if self.transform:
            resize = Resize((286, 286))
            input_image, output_image = resize(input_image), resize(output_image)
            random_crop = RandomCrop((256, 256))
            #Crop both input and output images with same random crop
            cropped = random_crop(torch.cat((input_image, output_image),axis=0))
            input_image = cropped[0:3,:,:]
            output_image = cropped[3:,:,:]
        #return input_image.float(), output_image.float()
        return input_image, output_image
