import torch

from src.Dataset import ImageDataset
from src.generator import Decoder, Discriminator, Encoder, Generator, Pix2Pix

dataset = ImageDataset("../data/base", transform=True)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=2
)

gen = Generator(in_channels=3, out_channels=3)
disc = Discriminator(in_channels=6)
pix2pix = Pix2Pix(gen, disc)
pix2pix.to('cuda')

pix2pix.train_model(200, dataloader, device='cuda')
