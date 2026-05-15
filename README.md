# Image Reconstruction with Deep Autoencoder in PyTorch (MNIST)

## Overview

This project demonstrates how to build and train a **Deep Autoencoder** using PyTorch for **image reconstruction** on the MNIST handwritten digit dataset.

The autoencoder learns how to compress 28×28 grayscale digit images into a compact latent representation and then reconstruct them back into their original form. This type of neural network is widely used in:

* Image compression
* Noise reduction (denoising)
* Feature extraction
* Dimensionality reduction
* Anomaly detection
* Representation learning

The implementation uses a **fully connected deep neural network** architecture with separate encoder and decoder components.

---

# Project Structure

```bash
.
├── data/
│   └── MNIST/
├── MNIST_Images/
│   ├── linear_ae_image0.png
│   ├── linear_ae_image50.png
│   └── linear_ae_image95.png
├── MNIST_reconstruction.png
├── deep_ae_mnist_loss.png
├── train.py
└── README.md
```

---

# Features

* Deep fully connected autoencoder
* GPU acceleration with CUDA support
* Automatic MNIST dataset downloading
* Reconstruction image visualization
* Training loss monitoring
* Latent-space compression
* Reconstruction result saving
* PyTorch DataLoader integration

---

# Required Libraries

Install the required dependencies before running the project.

## Install Dependencies

```bash
pip install torch torchvision matplotlib pillow
```

---

# Import Libraries

```python
import os
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
```

---

# Hyperparameters

```python
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 128
```

| Hyperparameter | Value |
| -------------- | ----- |
| Epochs         | 100   |
| Learning Rate  | 0.001 |
| Batch Size     | 128   |

---

# Data Preprocessing

The input images are converted into tensors and normalized.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

## Why Normalization?

Normalization helps:

* Stabilize training
* Improve gradient flow
* Speed up convergence

The pixel values are scaled approximately to:

x_{normalized}=\frac{x-0.5}{0.5}

which maps image values roughly into the range:

[-1,1]

---

# MNIST Dataset

The project uses the MNIST handwritten digit dataset.

## Dataset Information

| Property        | Value         |
| --------------- | ------------- |
| Training Images | 60,000        |
| Test Images     | 10,000        |
| Image Size      | 28×28         |
| Channels        | 1 (Grayscale) |
| Classes         | 10 Digits     |

---

# Loading Dataset

```python
trainset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

testset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
```

---

# DataLoader

```python
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
```

The `DataLoader`:

* Loads mini-batches efficiently
* Shuffles training samples
* Improves GPU utilization

---

# CUDA / Device Selection

```python
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
```

The model automatically uses:

* GPU if CUDA is available
* CPU otherwise

---

# Autoencoder Architecture

The network consists of:

1. Encoder
2. Bottleneck (latent space)
3. Decoder

---

# Encoder

The encoder compresses the image gradually:

```text
784 → 256 → 128 → 64 → 32 → 16
```

Where:

28\times28=784

The latent representation size is:

16

---

# Decoder

The decoder reconstructs the original image:

```text
16 → 32 → 64 → 128 → 256 → 784
```

---

# Autoencoder Model

```python
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 128)
        self.enc3 = nn.Linear(128, 64)
        self.enc4 = nn.Linear(64, 32)
        self.enc5 = nn.Linear(32, 16)

        # Decoder
        self.dec1 = nn.Linear(16, 32)
        self.dec2 = nn.Linear(32, 64)
        self.dec3 = nn.Linear(64, 128)
        self.dec4 = nn.Linear(128, 256)
        self.dec5 = nn.Linear(256, 784)

    def forward(self, x):

        # Encoder
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        # Decoder
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))

        return x
```

---

# Model Compression Concept

The encoder attempts to learn a compressed feature representation:

784\rightarrow16

Compression ratio:

\frac{16}{784}\approx0.0204

This means the latent vector contains only about **2%** of the original input dimensionality.

---

# Loss Function

The reconstruction loss is computed using Mean Squared Error (MSE).

```python
criterion = nn.MSELoss()
```

MSE formula:

MSE=\frac{1}{N}\sum_{i=1}^{N}(x_i-\hat{x}_i)^2

Where:

* (x_i) = original image pixels
* (\hat{x}_i) = reconstructed pixels

---

# Optimizer

```python
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
```

The project uses the Adam optimizer for:

* Faster convergence
* Adaptive learning rates
* Stable optimization

---

# Training Process

The training loop performs:

1. Forward propagation
2. Reconstruction loss computation
3. Backpropagation
4. Parameter updates

```python
outputs = net(img)
loss = criterion(outputs, img)

loss.backward()
optimizer.step()
```

---

# Training Results

The training loss gradually decreases over epochs:

```text
Epoch 1   → Loss: 0.924
Epoch 50  → Loss: 0.882
Epoch 100 → Loss: 0.881
```

This indicates the autoencoder successfully learns image reconstruction.

---

# Visualizing Training Loss

```python
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('deep_ae_mnist_loss.png')
```

The resulting graph shows:

* Smooth convergence
* Stable learning
* Reduced reconstruction error

---

# Reconstruction Visualization

The reconstructed images are periodically saved during training.

```python
save_decoded_image(outputs.cpu().data, epoch)
```

Saved examples:

* `linear_ae_image0.png`
* `linear_ae_image50.png`
* `linear_ae_image95.png`

These images demonstrate how reconstruction quality improves over time.

---

# Testing Reconstruction

```python
test_image_reconstruction(net, testloader)
```

The final reconstructed output is saved as:

```text
MNIST_reconstruction.png
```

---

# Reconstruction Pipeline

The complete autoencoder workflow:

```text
Input Image
      ↓
Flatten (28×28 → 784)
      ↓
Encoder
      ↓
Latent Representation (16)
      ↓
Decoder
      ↓
Reconstructed Image
```

---

# Results

The autoencoder successfully:

* Learns compressed image representations
* Preserves digit structure
* Reconstructs recognizable handwritten digits

As training progresses:

* Digits become clearer
* Noise decreases
* Reconstruction accuracy improves

---

# Advantages of Autoencoders

## Dimensionality Reduction

Autoencoders can reduce high-dimensional inputs into compact feature vectors.

## Unsupervised Learning

Labels are not required.

## Feature Learning

The model automatically discovers important visual features.

## Image Compression

The latent space acts as compressed image storage.

---

# Limitations

* Fully connected layers ignore spatial information
* Reconstruction quality is lower than convolutional autoencoders
* MSE may produce blurry outputs
* Deep linear networks can be parameter-heavy

---

# Possible Improvements

## Convolutional Autoencoder

Replace linear layers with convolutional layers for better spatial feature extraction.

## Better Activation Functions

Use:

* LeakyReLU
* GELU
* ELU

## Sigmoid Output Layer

Instead of ReLU at the final decoder layer:

```python
x = torch.sigmoid(self.dec5(x))
```

## Batch Normalization

Improve convergence stability.

## Dropout

Reduce overfitting.

## Denoising Autoencoder

Train the network to reconstruct clean images from noisy inputs.

## Variational Autoencoder (VAE)

Learn probabilistic latent representations.

---

# Example Future Architecture

```text
Conv2D → Conv2D → Bottleneck → ConvTranspose2D → Output
```

---

# Applications of Autoencoders

* Medical image reconstruction
* Face generation
* Super-resolution
* Image denoising
* Data compression
* Anomaly detection
* Deep feature extraction

---

# Run the Project

## Training

```bash
python train.py
```

---

# Output Files

| File                       | Description                         |
| -------------------------- | ----------------------------------- |
| `deep_ae_mnist_loss.png`   | Training loss graph                 |
| `MNIST_reconstruction.png` | Final reconstructed images          |
| `MNIST_Images/`            | Intermediate reconstruction outputs |

---

# Example Output

During early epochs:

* Reconstructions appear blurry

During later epochs:

* Digits become sharper
* Shapes become recognizable

---

# Conclusion

This project demonstrates the implementation of a deep fully connected autoencoder using PyTorch for MNIST image reconstruction.

The network successfully:

* Compresses image information into a low-dimensional latent space
* Learns meaningful representations
* Reconstructs handwritten digits with reasonable accuracy

This project provides a strong foundation for understanding:

* Representation learning
* Neural compression
* Deep unsupervised learning
* Generative modeling concepts

---
# !!! ReadMe is generated with GPT. Check the important info !!!
