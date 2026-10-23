"Here implementing both CNN for MNIST and auto-encoder structure"

from torch import nn
import torch.nn.functional as F

class CNNMNIST(nn.Module):
    def __init__(self,args):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=25)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

# "For AutoEncoder"
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1,(1,28,28)),
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


# For multi-classification

import torch.nn as nn
import torch.nn.functional as F

class SharedEncoder(nn.Module):
    def __init__(self, args=None):
        super(SharedEncoder, self).__init__()

        in_channels = args.num_channels if args is not None else 12

        # Choosing the latent feature size produced by encoder as 64
        # Task head will receive this
        self.output_dim = 128
        encoder_norm = getattr(args, "encoder_norm", "none")
        if encoder_norm not in {"none", "group"}:
            raise ValueError("encoder_norm must be none or group.")

        def activation(channels):
            if encoder_norm == "group":
                # No running statistics or affine parameters to transmit/noise.
                # Keep convolution state_dict keys and latent width unchanged.
                return nn.Sequential(
                    nn.GroupNorm(8, channels, affine=False),
                    nn.ReLU(),
                )
            return nn.ReLU()

        self.features = nn.Sequential(
            # Input: [batch_size, 12, 120,120]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            activation(32),
            nn.MaxPool2d(2),
            # Output: [batch_size, 32, 60, 60]

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            activation(64),
            nn.MaxPool2d(2),
            # Output [batch_size, 64, 30,30]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation(128),
            nn.MaxPool2d(2),
            # Output [batch_size, 128, 15,15]

        )
        # Converting [64,15,15] directly to [64,1,1] by global pooling
        # Each 15 dimension is replaced by one avg value

        self.global_pool =nn.AdaptiveAvgPool2d((1,1))
        # Output [batch_size, 128, 1,1]

        self.has_printed_shape = False

    def forward(self,x):
        x=self.features(x)

        x=self.global_pool(x)

        x=x.view(x.size(0),-1)

        if not self.has_printed_shape:
            print(f"Encoder latent shape: {x.shape}")
            self.has_printed_shape = True

        return x

class TaskHead(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(TaskHead, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,output_dim)
        )

    def forward(self,x_hat):
        return self.fc(x_hat)
