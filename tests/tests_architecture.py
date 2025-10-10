import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, latent_space_dimension: int = 20):
        super(Net, self).__init__()
        self.latent_space_dimension = latent_space_dimension
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, latent_space_dimension, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(int(16 * latent_space_dimension), 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, int(16 * self.latent_space_dimension))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
