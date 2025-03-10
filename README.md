## Deep Learning and Image Understanding IDC

### Overview
This repository contains multiple projects covering fundamental and advanced topics in **Machine Learning** and **Deep Learning**. Each project focuses on implementing different models, exploring optimization techniques, and evaluating performance on diverse datasets. The projects range from classical linear models to complex neural networks, including Convolutional and Recurrent Neural Networks.

### Projects Included

#### 1. **Linear Image Classifier**
- **Objective**: Implement a **linear classifier** using `numpy` to classify images.
- **Key Concepts**: Gradient descent, loss functions, optimization techniques, and performance evaluation.
- **Techniques Covered**: 
  - Implementing a linear classifier with vectorized operations.
  - Training classifiers with gradient descent.
  - Evaluating performance with accuracy and loss metrics.

#### 2. **Neural Networks Classifier**
- **Objective**: Implement a **three-layer multi-class neural network** from scratch.
- **Key Concepts**: Forward and backward propagation, activation functions, and hyperparameter tuning.
- **Techniques Covered**:
  - Implementing a feedforward neural network with multiple layers.
  - Using activation functions such as ReLU and softmax.
  - Optimizing model parameters using gradient descent.

#### 3. **Convolutional Neural Network (CNN) - Classifying CIFAR-10 and Localization as Regression**
- **Objective**: Implement a **Convolutional Neural Network (CNN)** to classify CIFAR-10 images and use **ResNet18** for localization.
- **Key Concepts**: Feature extraction, transfer learning, automatic differentiation, and object localization.
- **Techniques Covered**:
  - Training a CNN using PyTorch.
  - Using **ResNet18** for feature extraction and localization.
  - Evaluating classification and regression-based localization.

#### 4. **Recurrent Neural Networks (RNNs) and Image Captioning**
- **Objective**: Implement an **RNN** to process sequential data and build an **image captioning model** using a pretrained network.
- **Key Concepts**: Sequence modeling, backpropagation through time (BPTT), and multimodal learning.
- **Techniques Covered**:
  - Manually implementing RNN forward and backward passes.
  - Using a pretrained **ResNet152 + LSTM** model for image captioning.
  - Applying transfer learning for multimodal tasks.

### Libraries and Dependencies
Each project is implemented using a combination of the following libraries:
- `numpy` – For numerical computations.
- `matplotlib` – For data visualization.
- `torch` – For building and training neural networks.
- `torchvision` – For working with pretrained models and image datasets.
- `scipy` – For advanced mathematical functions.

### Summary
This repository provides hands-on experience across a range of **Machine Learning and Deep Learning** techniques. By implementing models from **linear classifiers** to **CNNs** and **RNNs**, these projects serve as a strong foundation for more advanced **AI research and real-world applications**.
