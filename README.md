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

The current phase is dedicated to mastering Convolutional Neural Networks (CNNs) and crucial optimization techniques necessary for handling image data and preparing for advanced architectures.

### Upcoming Modules:

1.  **Convolutional Layers:** `nn.Conv2d`, Stride, Padding, and feature extraction.
2.  **Architectures & Pooling:** Implementing basic CNN structures and using `nn.MaxPool2d`.
3.  **Advanced Data Augmentation:** Utilizing `torchvision.transforms` for robust training.
4.  **Optimization:** Implementing Batch Normalization and Learning Rate Scheduling.

### Phase 2 Capstone:

* **Transfer Learning Project:** Classifying a custom image dataset by fine-tuning a pre-trained model (e.g., **ResNet-18**).

---

## 🛠️ Repository Structure
├── phase1-foundation/ │ ├── capstone_fashion_mnist.py # Final project code │ └── exercises.ipynb # Module 1-5 exercise solutions ├── phase2-intermediate/ # (Future Code) ├── requirements.txt └── README.md |

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
