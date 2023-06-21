import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("/data/logs")


class Encoder(nn.Module):
    def __init__(
        self, in_channels=3, channels_first=64, channels_last=512, n_layers=8
    ) -> None:
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        in_chans = min(in_channels, channels_last)
        out_chans = min(channels_last, channels_first)

        for i in range(0, n_layers):
            # First and last layers don't have batchnorm
            self.enc_blocks.append(
                self.get_encoder_block(
                    in_chans, out_chans, not (i == 0 or i == n_layers - 1)
                )
            )
            in_chans = min(channels_last, out_chans)
            out_chans = min(channels_last, in_chans * 2)

    def forward(self, x):
        for enc in self.enc_blocks:
            x = enc(x)
        return x

    def get_encoder_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
            )


class Decoder(nn.Module):
    def __init__(self, out_channels) -> None:
        super().__init__()
        # Decoder layers input will have skip connections from encoder
        self.dec_blocks = nn.ModuleList(
            [
                self.get_decoder_block(512, 512),
                self.get_decoder_block(1024, 512),
                self.get_decoder_block(1024, 512),
                self.get_decoder_block(1024, 512),
                self.get_decoder_block(1024, 256),
                self.get_decoder_block(512, 128),
                self.get_decoder_block(256, 64),
            ]
        )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1), nn.Tanh()
        )

    def get_decoder_block(self, in_channels, out_channels, dropout=True):
        if dropout:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(p=0.5),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

    def forward(self, encoder_outputs):
        N = len(encoder_outputs)

        # first layer
        out = self.dec_blocks[0](encoder_outputs[-1])

        for i, dec in enumerate(self.dec_blocks[1:], start=1):
            index = N - 1 - i
            out = torch.concat((out, encoder_outputs[index]), axis=1)
            out = dec(out)

        out = torch.concat((out, encoder_outputs[0]), axis=1)

        return self.final_layer(out)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(out_channels=out_channels)

    def forward(self, inp):
        out = [self.encoder.enc_blocks[0](inp)]
        for en in self.encoder.enc_blocks[1:]:
            out.append(en(out[-1]))

        decoded = self.decoder(out)

        return decoded


class Discriminator(nn.Module):
    def __init__(
        self, in_channels=3, channels_first=64, max_channels=512, n_layers=4
    ) -> None:
        super().__init__()

        self.disc_blocks = nn.ModuleList()
        in_chans = min(in_channels, max_channels)
        out_chans = min(max_channels, channels_first)

        for i in range(0, n_layers):
            self.disc_blocks.append(
                self.get_discriminator_block(in_chans, out_chans, not i == 0)
            )
            in_chans = min(max_channels, out_chans)
            out_chans = min(max_channels, in_chans * 2)

        self.final_layer = nn.Conv2d(out_chans, 1, 4, stride=2, padding=1)

    def get_discriminator_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.2),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
            )

    def forward(self, x):
        for disc in self.disc_blocks:
            x = disc(x)

        return self.final_layer(x)


class Pix2Pix(nn.Module):
    def __init__(self, generator, discriminator) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.L1Loss = torch.nn.L1Loss()

        self.optimizer_gen = torch.optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )

        self.optim_gen_schedul = torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer_gen, lr_lambda=lambda epoch: 0.98
        )
        self.optim_disc_schedul = torch.optim.lr_scheduler.MultiplicativeLR(
            self.optimizer_disc, lr_lambda=lambda epoch: 0.98
        )

        self.generator.apply(self.init_weights)
        self.discriminator.apply(self.init_weights)

    def init_weights(self, m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input_image):
        img_gen = self.generator(input_image)
        return img_gen

    def set_requires_grad(self, net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def train_model(self, epochs, train_dataloader, device='cpu'):
        i = 0
        for epoch in range(epochs):
            disc_loss = 0
            gen_l1_loss = 0
            gen_log_loss = 0
            n_batch = 0
            for batch in train_dataloader:
                gen_out = self.forward(batch[0].to(device))

                self.set_requires_grad(self.discriminator, True)

                self.optimizer_disc.zero_grad()
                disc_real_out = self.discriminator(
                    torch.concat((batch[0].to(device), batch[1].to(device)), axis=1)
                )
                disc_fake_out = self.discriminator(
                    torch.concat((batch[0].to(device), gen_out.detach().to(device)), axis=1)
                )

                loss_disc = self.loss_fn(
                    disc_real_out, torch.ones(1).expand_as(disc_real_out).to(device)
                ) + self.loss_fn(disc_fake_out, torch.zeros(1).expand_as(disc_fake_out).to(device))
                loss_disc = loss_disc * 0.5
                loss_disc.backward()
                self.optimizer_disc.step()

                self.set_requires_grad(self.discriminator, False)

                self.optimizer_gen.zero_grad()
                disc_fake_out = self.discriminator(
                    torch.concat((batch[0].to(device), gen_out), axis=1)
                )

                loss_gen_part1 = self.loss_fn(
                    disc_fake_out, torch.ones(1).expand_as(disc_fake_out).to(device)
                )
                loss_gen_part2 = self.L1Loss(gen_out, batch[1].to(device)) * 100

                loss_gen = loss_gen_part1 + loss_gen_part2

                loss_gen.backward()
                self.optimizer_gen.step()
                print(epoch, loss_gen.detach(), loss_disc.detach())
                disc_loss = disc_loss + loss_disc.detach()

                gen_l1_loss = gen_l1_loss + loss_gen_part2.detach()
                gen_log_loss = gen_log_loss + loss_gen_part1.detach()

                n_batch = n_batch + 1

            writer.add_scalar("gen_l1_loss", gen_l1_loss / n_batch, epoch)
            writer.add_scalar("gen_log_loss", gen_log_loss / n_batch, epoch)
            writer.add_scalar("disc_loss", disc_loss / n_batch, epoch)

            self.optim_gen_schedul.step()
            self.optim_disc_schedul.step()
            if epoch % 10 == 0:
                torch.save(self.state_dict(), f"/data/model_{epoch}.torch")

    def loss_disc(self, disc_fake_out, disc_real_out):
        gan_loss = self.loss_fn(disc_fake_out, 0) + self.loss_fn(disc_real_out, 1)
        return gan_loss


class Pix2PixLightining(nn.Module):
    def __init__(self, generator, discriminator) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=1e-5)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=1e-5)

    def forward(self, input_image):
        img_gen = self.generator(input_image)
        return img_gen

    def set_requires_grad(self, net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def train(self, epochs, train_dataloader):
        for epoch in range(epochs):
            for batch in train_dataloader:
                gen_out = self.forward(batch[0])

                self.set_requires_grad(self.discriminator, True)
                self.set_requires_grad(self.generator, False)

                self.optimizer_disc.zero_grad()

                disc_real_out = self.discriminator(
                    torch.concat((batch[0], batch[1]), axis=1)
                )
                disc_fake_out = self.discriminator(
                    torch.concat((batch[0], gen_out.detach()), axis=1)
                )

                loss_disc = self.loss_fn(
                    disc_real_out, torch.ones(1).expand_as(disc_real_out)
                ) + self.loss_fn(disc_fake_out, torch.zeros(1).expand_as(disc_fake_out))

                loss_disc.backward()
                self.optimizer_disc.step()

                self.set_requires_grad(self.discriminator, False)
                self.set_requires_grad(self.generator, True)

                self.optimizer_gen.zero_grad()
                disc_fake_out = self.discriminator(
                    torch.concat((batch[0], gen_out), axis=1)
                )
                loss_gen = self.loss_fn(
                    disc_fake_out, torch.ones(1).expand_as(disc_fake_out)
                )
                loss_gen.backward()
                self.optimizer_gen.step()

    def loss_disc(self, disc_fake_out, disc_real_out):
        gan_loss = self.loss_fn(disc_fake_out, 0) + self.loss_fn(disc_real_out, 1)
        return gan_loss
