# Neural-Network-and-Optimizer-Comparison
This repository contains benchmarking and comparison of the FFNN, RNN, and CNN on sequential and image data, alongwith using different optimizers for analyzing performance.

1. FFNN vs. RNN on Sequential Data (HAR Dataset)

In this comparison, the UCI Human Activity Recognition (HAR) dataset was used to classify six physical activities using smartphone sensor data. A custom PyTorch dataset class was created to load and preprocess data from nine signal types, including body acceleration, gyroscope, and total acceleration on three axes (x, y, z). Each signal was collected at 128 time steps, and the final input shape was structured as [samples, 128, 9]. All signals were standardized using global z-score normalization, and the learning rate was set to 0.001.

Two models were implemented. The Feedforward Neural Network (FFNN) flattened the input from [128, 9] to a 1D vector of size 1152 and passed it through three linear layers: Linear (1152 → 256) → ReLU → Dropout (0.3) → Linear (256 → 128) → ReLU → Linear (128 → 6). The Recurrent Neural Network (RNN) used an LSTM architecture, where the sequence was fed directly into an LSTM (input size = 9, hidden_size = 64), and the final hidden state was passed into a Linear (64 → 6) layer for classification. Both models were trained using cross-entropy loss and optimized with Adam.

2. FFNN vs. CNN on Image Data (FashionMNIST)

The second comparison used the FashionMNIST dataset, consisting of (28 x 28) grayscale images of clothing items from 10 categories. No resizing was required, and all pixel
values were normalized to fall within a common scale using mean = 0.5 and std = 0.5. The input shape was [batch_size, 1, 28, 28] for both models with a learning rate of 0.001.

The FFNN model flattened the input to a vector of 784 and passed it through the layers: Linear (784 → 256) → ReLU → Linear (256 → 64) → ReLU → Linear (64 → 10). The CNN model used two convolutional layers: Conv2D (1 → 32, kernel = 3, padding = 1) → ReLU → MaxPool (2) followed by Conv2D (32 → 64, kernel = 3, padding = 1) → ReLU → MaxPool (2), which transformed the image into a [64, 7, 7] tensor. This was then flattened and passed through Linear (64 x 7 x 7 → 128) → ReLU → Dropout (0.3) → Linear (128 → 10). Both models used cross-entropy loss and were trained under the same batch size of 64 and 5 epochs.

3. Optimizer Comparison on FFNN (FashionMNIST)

The final part of Milestone 3 reused the FFNN architecture from Part 2 to evaluate how different optimizers affect training behavior. The preprocessing steps remained unchanged: the FashionMNIST images were normalized, flattened, and processed as input vectors of size 784. The same FFNN model (784 → 256 → 64 → 10 with ReLU activations) was trained using three
optimizers: Adam, SGD, and RMSProp. All models were initialized identically and trained for the same number of epochs with a learning rate of 0.001 and a batch size of 64. Early stopping was implemented to halt training if validation loss did not improve over a fixed patience threshold. This controlled setup allowed for a fair comparison of training speed and loss convergence across optimizers.
