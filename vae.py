import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class VAE(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, latent_dim=64):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # mean μ
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # log variance log(σ²)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
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
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparamenterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def train_vae(
    model, dataloader, optimizer, num_epochs=10, learning_rate=1e-3, beta=1.0
):
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = datasets.ImageFolder(train_data_path, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # input_dim = width x height
    model = VAE(input_dim=1600, hidden_dim=512, latent_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_vae(model, train_loader, optimizer)
