import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(ConvVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=4, stride=2, padding=1
            ),  # Input: 3x40x40, Output: 32x20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: 64x10x10
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128x5x5
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(128 * 5 * 5, latent_dim)  # mean μ
        self.fc_logvar = nn.Linear(128 * 5 * 5, latent_dim)  # log variance log(σ²)

        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, 128 * 5 * 5)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # Output: 64x10x10
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # Output: 32x20x20
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=4, stride=2, padding=1
            ),  # Output: 3x40x40
            nn.Sigmoid(),  # Assuming input is normalized between 0 and 1
        )

    def encode(self, x):
        """Encode the input: x -> (μ, log(σ²))"""
        h = self.encoder(x)  # hidden representation
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparamenterize(self, mu, logvar):
        """Reparameterization trick: (μ, log(σ²)) -> z"""
        std = torch.exp(0.5 * logvar)  # standard deviation
        eps = torch.randn_like(std)  # random noise ε ~ N(0,1)
        z = mu + eps * std
        return z

    def decode(self, z):
        """Decode the latent representation: z -> x̂"""
        h = self.fc_decoder(z)  # hidden representation
        h = h.view(-1, 128, 5, 5)  # reshape to (batch_size, 128, 5, 5)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparamenterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def train_vae(model, dataloader, optimizer, num_epochs=10, beta=1.0):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_idx, (data, _) in enumerate(dataloader):
            optimizer.zero_grad()

            x_recon, mu, logvar = model(data)

            recon_loss = F.binary_cross_entropy(x_recon, data, reduction="sum")

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            total_loss = recon_loss + beta * kl_loss

            loss = recon_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        if epoch % 10 == 0:
            average_loss = total_loss / len(dataloader.dataset)
            average_recon_loss = total_recon_loss / len(dataloader.dataset)
            average_kl_loss = total_kl_loss / len(dataloader.dataset)
            print(
                f"Epoch [{epoch}/{num_epochs}], Loss: {average_loss:.4f}, Recon Loss: {average_recon_loss:.4f}, KL Loss: {average_kl_loss:.4f}"
            )

    return model


if __name__ == "__main__":
    train_data_path = "./data/train"
    transform_train = transforms.Compose(
        [
            transforms.Resize((40, 40)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.ImageFolder(train_data_path, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = ConvVAE(latent_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_vae(model, train_loader, optimizer, num_epochs=500)
