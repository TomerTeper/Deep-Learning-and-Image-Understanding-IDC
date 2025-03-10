## README: Convolutional Neural Network (CNN) - Classifying CIFAR-10 and Localization as Regression

### Overview
This project involves implementing a **Convolutional Neural Network (CNN)** for image classification on the **CIFAR-10** dataset using **PyTorch**. Unlike previous exercises, where both forward and backward passes were manually implemented, this task leverages **automatic differentiation** for computing gradients. Additionally, a **Localization as Regression** task is introduced using a **pretrained ResNet18** model to classify and localize images of cats and dogs.

### Project Structure
- **Data Preprocessing**: Loading and normalizing CIFAR-10 images.
- **CNN Architecture**: Implementing a multi-layer CNN for feature extraction and classification.
- **Automatic Differentiation**: Using PyTorch’s `autograd` to compute gradients efficiently.
- **Loss Function and Optimization**: Implementing cross-entropy loss and training with SGD or Adam optimizer.
- **Model Training**: Training the CNN on CIFAR-10 with proper evaluation metrics.
- **GPU/CPU Compatibility**: Ensuring the model works on both CPU and GPU.
- **Localization as Regression**: Using a **pretrained ResNet18 model** to classify and localize images of cats and dogs.
- **Transfer Learning**: Leveraging pre-trained ResNet18 features to enhance classification performance on a new dataset with minimal training data.

### Key Topics Covered
- **Convolutional Neural Networks**: Implementing feature extractors with convolutional layers.
- **Activation Functions**: Using ReLU for non-linearity and Softmax for classification.
- **Batch Normalization and Dropout**: Improving generalization and preventing overfitting.
- **Loss Functions**: Using cross-entropy loss for multi-class classification.
- **Optimization Techniques**: Training the network using optimizers like Adam or SGD.
- **Automatic Differentiation**: Using PyTorch’s `autograd` for efficient gradient computation.
- **Performance Evaluation**: Measuring accuracy and loss during training and testing.
- **GPU/CPU Compatibility**: Ensuring the model can switch between GPU and CPU execution.
- **Transfer Learning with ResNet18**: Utilizing pre-trained ResNet18 for feature extraction.
- **Localization as Regression**: Predicting bounding boxes for object localization using regression techniques.

### Libraries and Dependencies
This project requires the following Python libraries:
- `torch` – For building and training the CNN model.
- `torchvision` – For accessing the CIFAR-10 dataset and common image transformations.
- `numpy` – For numerical computations.
- `matplotlib` – For visualizing training progress.

### Summary
This project provides hands-on experience in implementing a CNN for image classification using **PyTorch**. It introduces concepts such as convolutional layers, automatic differentiation, and deep learning optimizations. Additionally, it explores **transfer learning** using a **pretrained ResNet18 model** to classify and localize cats and dogs with minimal training data. By combining CNN classification with object localization, this exercise builds a strong foundation for advanced computer vision tasks.
