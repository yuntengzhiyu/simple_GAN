import torchvision
import torch
from torch import nn, optim
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

#定义两个网络 G, D
class Generator(nn.Module):
    def __init__(self, in_features ,out_features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,out_features),
            nn.Sigmoid(),
        )
    def forward(self,x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.disc(x)

z_dim = 64
real_dim = 784
disc = Discriminator(real_dim)
gen = Generator(z_dim, real_dim)


#下载mnist
train_data = torchvision.datasets.MNIST('data/mnist',train=True, download=True, transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

#定义优化器
optim_disc = optim.Adam(disc.parameters(), lr=0.0002,betas=(0.9, 0.999))
optim_gen = optim.Adam(gen.parameters(), lr=0.0002,betas=(0.9, 0.999))
#定义损失函数
criterion = nn.BCELoss(reduction='mean')


epoch_num = 200
for epoch in range(epoch_num):
    for idx, (x,_)in enumerate(dataloader):
        x = x.reshape(-1,784)
        batch_size = x.shape[0]
        #计算判别器的损失
        D_real = disc(x)
        loss_real = criterion(D_real, torch.ones_like(D_real))
        #假设此时G已经是固定的
        noise = torch.randn(batch_size,z_dim)
        gen_x = gen(noise)
        assert gen_x.shape == (batch_size, real_dim)
        D_fake = disc(gen_x.detach())
        loss_fake = criterion(D_fake, torch.zeros_like(D_fake))
        loss_disc = loss_real + loss_fake

        loss_disc.backward()
        optim_disc.step()
        disc.zero_grad()

        #计算生成器的损失
        loss_gen = criterion(disc(gen_x),torch.ones_like(D_fake))
        loss_gen.backward()
        optim_gen.step()
        gen.zero_grad()

        if idx % 100 == 0:
            print(f'Epoch: {epoch}, batch: {idx}')
    with torch.no_grad():
        noise = torch.randn(64,z_dim)
        fake_img = gen(noise).reshape(-1,1,28,28)
        grid = torchvision.utils.make_grid(fake_img, nrow=8)
        plt.imshow(grid.permute(1,2,0).numpy())
        plt.show()

