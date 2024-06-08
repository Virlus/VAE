from torchvision.utils import save_image
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        x = self.decoder(z)
        return x

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_pred = self.decoder(z).reshape(-1, 1, 28, 28)
        return x_pred, mu, logvar


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 2
input_dim = 784
ckpt_path = "results_2/best.pt"

model = VAE(input_dim = input_dim, latent_dim = latent_dim).to(device)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt)

z = ((torch.rand(40, 2) * 10) - 5).to(device)

for i in range(40):
    z_i = z[i]
    img_i =  model.decode(z_i).view(-1, 1, 28, 28)
    save_image(img_i, f"latent_2_eval/{round(z_i[0].detach().cpu().item(), 2)}_{round(z_i[1].detach().cpu().item(), 2)}.png")

img = model.decode(z).view(-1, 1, 28, 28)
save_image(img, f"latent_2_eval/all.png")