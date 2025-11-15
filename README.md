# 🧠 PyTorch Deep Learning Professional Path

## Course Overview

This repository documents my structured, project-based journey from foundational principles to professional deployment skills in Deep Learning, utilizing the **PyTorch** framework. This self-paced curriculum is designed to mirror the rigor and practical application focus of a professional certificate (similar to those offered by DeepLearning.AI).

The primary goal is to build a robust portfolio of hands-on projects, ensuring proficiency in core concepts, advanced architectures, and modern MLOps practices.

---

## 🎯 Phase 1: Foundation (Completed)

This phase established the core building blocks of PyTorch and the fundamental concepts of neural network training. All projects and exercises within this phase are complete and available in the `/phase1-foundation` directory.

### Core Concepts Covered:

* **Tensors & Device Management:** Understanding PyTorch's primary data structure and efficiently moving data/models between CPU and GPU (`.to(device)`).
* **Autograd:** Mastering automatic differentiation, the computational graph, and gradient calculation (`.backward()`).
* **Model Building:** Defining complex, trainable network architectures using `torch.nn.Module` and `nn.Linear`.
* **Training Essentials:** Implementing the core training loop: loss calculation (`nn.CrossEntropyLoss`), optimization (`torch.optim.Adam`), zeroing gradients (`.zero_grad()`), and parameter updates (`.step()`).
* **Data Handling:** Professional management of datasets using `torch.utils.data.Dataset` and efficient batch loading with `torch.utils.data.DataLoader`.

### Capstone Project:

| Project | Description | Status |
| :--- | :--- | :--- |
| **Fashion-MNIST Classifier** | Built, trained, and evaluated a fully-connected Neural Network to classify 10 items from the Fashion-MNIST dataset, achieving an accuracy of **>88%**. | ✅ Complete |

---

## 🚀 Phase 2: Intermediate (Current Focus)

This phase was focused on mastering Convolutional Neural Networks (CNNs) and essential optimization techniques needed for high-performance computer vision tasks. The key success was implementing and utilizing Transfer Learning.

Core Concepts Covered:
* Convolutional Layers (`nn.Conv2d`): Understanding kernels, stride, padding, and feature map shape transformations.

* CNN Architectures: Combining Conv, ReLU, and Pooling (`nn.MaxPool2d`) layers into cohesive network structures (e.g., MiniVGG).

* Data Augmentation: Using `torchvision.transforms` for robust training, including `RandomCrop`, `HorizontalFlip`, and essential `Normalization`.

* Optimization: Integrating Batch Normalization (`nn.BatchNorm2d`) and Dropout (`nn.Dropout`) for training stability and regularization.

* Learning Rate Scheduling: Implementing `StepLR` to dynamically adjust the learning rate during training.

* Transfer Learning: Loading pre-trained models (`ResNet-18`) and utilizing the Feature Extractor (Fine-Tuning) method to achieve high accuracy with minimal data.
* 
### Phase 2 Capstone:

CIFAR-10 Transfer Learning:	Built a high-performance image classifier by fine-tuning the last few layers of a pre-trained ResNet-18 model on the CIFAR-10 dataset, achieving an accuracy of >88%.

---

## 🛠️ Repository Structure
| Directory/File | Type | Purpose |
| :--- | :--- | :--- |
| `README.md` | File | The document you are currently reading. |
| `requirements.txt` |File | Lists all necessary Python packages `(torch, torchvision, numpy)` for running the code. |
| `phase1-foundation/` | Folder | Contains all code and exercises from the foundational phase. |
| ├── `capstone_fashion_mnist.py` | Script | The final project script for the Fashion-MNIST classification (Module 6).|
| └── `exercises.ipynb` | Notebook | Contains solutions and notes for Modules 1 through 5 exercises. |
| `phase2-intermediate/` | Folder | Will house all code and projects related to Intermediate Deep Learning (CNNs, Transfer Learning).


## 💡 Phase 3: Advanced Deep Learning
The final phase shifts from model architecture to production readiness, advanced sequence models, and model explainability, moving towards MLOps and complex AI systems.

### Setup and Running Code

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Alexmarco-gif/pytorch-deep-learning-professional-path.git](https://github.com/Alexmarco-gif/pytorch-deep-learning-professional-path.git)
    cd pytorch-deep-learning-professional-path
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Capstone Project (e.g., Phase 1):**
    ```bash
    python phase1-foundation/capstone_fashion_mnist.py
    ```

---

## 🤝 Next Steps

This repository is actively maintained as a record of my learning path. Contributions (e.g., cleaner code implementations, performance tuning suggestions) are welcome!

**Next Goal:** Complete Phase 2, Module 1: Convolutional Layers.
