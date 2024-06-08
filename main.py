import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from torch.utils.tensorboard import SummaryWriter

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
    
def loss_function(x_pred, x, mu, logvar):
    bce_loss = F.binary_cross_entropy(x_pred, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kld_loss

def evaluate(model, data_loader, batch_size, latent_dim, device, save_dir, epoch, best_loss, tb_writer):
    loss = 0
    for idx, (img, target) in enumerate(data_loader):
        img = img.to(device)
        x_hat, mu, var = model(img)
        rec_loss = loss_function(x_hat, img, mu, var)
        loss += rec_loss.item()
    loss /= len(data_loader.dataset)
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), save_dir + '/best.pt')
    tb_writer.add_scalar('Loss/test_loss', loss, epoch)
    
    # Generation
    z = torch.randn(batch_size, latent_dim).to(device)
    samples = model.decode(z).view(batch_size, 1, 28, 28)
    save_image(samples, save_dir + f'/epoch_{epoch}.png')
    
    return loss

# Download MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

# MNist Data Loader
batch_size=50
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 1
save_dir = 'results_' + str(latent_dim)
epochs = 100
seed = 1
lr = 0.001
input_dim = 784
eval_interval = 5

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_loss = float('inf')
writer = SummaryWriter('logs/'+str(latent_dim))


# iterator over train set
for epoch in range(epochs):
    total_loss = 0
    for idx, (img, target) in enumerate(train_loader):
        img = img.to(device)
        x_hat, mu, var = model(img)
        rec_loss = loss_function(x_hat, img, mu, var)
        optimizer.zero_grad()
        rec_loss.backward()
        optimizer.step()
        total_loss += rec_loss.item()
    total_loss /= len(train_loader.dataset)
    writer.add_scalar('Loss/train_loss', total_loss, epoch)
    if epoch % eval_interval == 0:
        best_loss = evaluate(model, test_loader, batch_size, latent_dim, device, save_dir, epoch, best_loss, writer)
    
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Best Loss: {best_loss:.4f}')
