import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torchvision.utils import save_image

d_depth = 32
g_depth = 32
lr = 0.0002
bs = 1
epochs = 10
img_size = 64
img_channels = 1
seed_channels = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(seed_channels, g_depth * 16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(g_depth * 16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(g_depth * 16, g_depth * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(g_depth * 8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(g_depth * 8, g_depth * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(g_depth * 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(g_depth * 4, g_depth * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(g_depth * 2),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(g_depth * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  #Tanh activation
        )

    def forward(self, x):
        return self.model(x)

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.model = nn.Sequential(

            nn.Conv2d(img_channels, d_depth, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d_depth, d_depth * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_depth * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d_depth * 2, d_depth * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_depth * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d_depth * 4, d_depth * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d_depth * 8),
            nn.LeakyReLU(0.2),
        
            nn.Conv2d(d_depth * 8, 1, kernel_size=4, stride=2, padding=0),  #We avoid using a FC layer 
            nn.Sigmoid(), #Sigmoid activation
        )

    def forward(self, x):
        return self.model(x)

#Transforms
all_transforms = transforms.Compose([
  transforms.Grayscale(num_output_channels=1),  #Convert to grayscale 
  transforms.Resize((img_size, img_size)),
  transforms.ToTensor(),
  transforms.Normalize((0.5),(0.5)),  #This should be changed to the mean and std of your dataset                                    
])

#Dataset
root_dir = 'MMBCReco/src/GAN/trainGAN'
dataset = datasets.ImageFolder(root = root_dir, transform = all_transforms)

#Dataloader
loader = DataLoader(dataset, batch_size=bs, shuffle=True)

#Initialize the networks
G = GNet().to(device)  #Generator
D = DNet().to(device) #Discriminator
G.train()
D.train()

#Loss and Optimizers
criterion = nn.BCELoss()
GOptim = optim.Adam(G.parameters(), lr=lr, betas=(0.7,0.999))
DOptim = optim.Adam(D.parameters(), lr=lr, betas=(0.7,0.999))

#Set a sample seed that will allow us to visualize results as training progresses
sample_seed = torch.randn(bs, seed_channels, 1, 1).to(device)

#Training
for epoch in range(epochs):
    loop = tqdm(enumerate(loader), total = len(loader), leave=False)
    for batch, (data, targets) in loop:

        data = data.to(device)
        bs = data.shape[0]

        D.zero_grad()
        targets = (torch.ones(bs)*0.95).to(device)
        results = D(data).reshape(-1)
        D_loss_1 = criterion(results, targets)
        seeds = torch.randn(bs, seed_channels, 1, 1).to(device)
        gen_images = G(seeds)
        targets = (torch.ones(bs)*0.05).to(device)
        results = D(gen_images.detach()).reshape(-1)
        D_loss_0 = criterion(results, targets)
        lossD = D_loss_1 + D_loss_0
        lossD.backward()
        DOptim.step() 

        G.zero_grad()
        targets = torch.ones(bs).to(device)
        results = D(gen_images).reshape(-1)
        lossG = criterion(results, targets)
        lossG.backward()
        GOptim.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"% (epoch, epochs, batch, len(loader), lossD, lossG))

        batches_done = epoch * len(loader) + batch

        if batches_done % 100 == 0:  #Show samples at every 100th batch
            with torch.no_grad():
                fake = G(sample_seed)
                samples_fake = torchvision.utils.make_grid(fake, normalize=True)
                save_image(fake.data[:25], "MMBCReco/src/images/%d.png" % batches_done, nrow=5, normalize=True)