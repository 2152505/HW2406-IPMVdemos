import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, input_size)
        self.fc = nn.Sequential(
            nn.Linear(input_size * 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        return self.fc(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, input_size)
        self.fc = nn.Sequential(
            nn.Linear(input_size * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        x = torch.cat([img, c], 1)
        return self.fc(x)

# 超参数设置
num_classes = 10  # 对于MNIST数据集，有10个类别
input_size = 100
hidden_dim = 256
output_size = 784  # 28x28
num_epochs = 10
batch_size = 32
learning_rate = 0.0002

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 创建网络
generator = Generator(input_size, hidden_dim, output_size, num_classes)
discriminator = Discriminator(output_size, hidden_dim, num_classes)

# 损失函数和优化器
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 训练网络
for epoch in range(num_epochs):
    for batch_idx, (real_images, labels) in enumerate(train_loader):
        real_images = real_images.view(-1, 28*28)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        labels = labels % num_classes  # 确保标签在0-9之间

        # 训练判别器
        discriminator.zero_grad()
        outputs = discriminator(real_images, labels)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        noise = torch.randn(batch_size, input_size)
        fake_images = generator(noise, labels)
        outputs = discriminator(fake_images.detach(), labels)
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        d_optimizer.step()

        # 训练生成器
        generator.zero_grad()
        outputs = discriminator(fake_images, labels)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss_real.item()+d_loss_fake.item():.4f}, g_loss: {g_loss.item():.4f}')

# 生成指定数字的图像
def generate_digit(generator, digit):
    z = torch.randn(1, input_size)
    labels = torch.LongTensor([digit]).to(z.device)  # 指定生成的数字
    fake_images = generator(z, labels).view(-1, 28, 28).data.numpy()
    plt.imshow(fake_images[0], cmap='gray')
    plt.show()

# 示例：生成数字1的图像
generate_digit(generator, 1)