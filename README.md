# 🧠 Deep Learning Classification on DermaMNIST

This project implements and compares Feedforward and Convolutional Neural Networks for image classification using the **DermaMNIST** dataset from [MedMNIST v2](https://medmnist.com/). The goal is to classify skin lesion images into 7 dermatological conditions using deep learning.

---

## 📁 Dataset: DermaMNIST

- **Source**: [MedMNIST v2](https://medmnist.com/)
- **Data Type**: 28x28 grayscale images
- **Classes**:
  - 0: Actinic keratoses and intraepithelial carcinoma  
  - 1: Basal cell carcinoma  
  - 2: Benign keratosis-like lesions  
  - 3: Dermatofibroma  
  - 4: Melanoma  
  - 5: Melanocytic nevi  
  - 6: Vascular lesions  

The dataset is downloaded and preprocessed using the `medmnist` Python library.

---

## 🏗️ Model Architectures

### 1. Feedforward Neural Network (FFN)
- **Input**: Flattened 784-dim vector (28×28)
- **Hidden Layer**: 256 neurons with Sigmoid
- **Output Layer**: 7 neurons with Softmax
- **Loss**: CrossEntropyLoss

### 2. Convolutional Neural Network (CNN)
- **Conv2D**: 32 filters, kernel size 3×3
- **MaxPooling2D**: 2×2
- **Fully Connected Layers**: 128 → 7 output classes
- **Loss**: CrossEntropyLoss

---

## 🏋️ Training Details

- **Framework**: PyTorch  
- **Epochs**: 10  
- **Batch Size**: 64  
- **Optimizer**: Adam  
- **Device**: CUDA (GPU) if available  
- **Transforms**: Grayscale conversion and normalization using `ToTensor()`

---

## 📊 Evaluation and Visualizations

The notebook includes:
- 📈 Accuracy & Loss curves for both FFN and CNN
- 🔁 Confusion matrices
- 🔍 Per-class accuracy bar plots
- ❌ Misclassified sample visualizations
- 🖼️ Sample montages of dataset images

---

## 🔧 Installation

Install the required libraries:

```bash
pip install torch torchvision scikit-learn medmnist v2 matplotlib seaborn pillow
```
## 🚀 How to Run
- Clone this repository:
```bash
git clone https://github.com/yourusername/deep-derma-classification.git
cd deep-derma-classification
```
## ✅ Results Summary

| Model | Final Training Accuracy | Key Observations                    |
| ----- | ----------------------- | ----------------------------------- |
| FFN   | \~81%                   | Lower capacity, overfitting likely  |
| CNN   | \~85%+                  | Better generalization & performance |

CNN outperforms FFN in both accuracy and robustness.

## 📄 Citation
- If you use this work or dataset, cite the MedMNIST paper:
```bash
@article{medmnistv2,
  title={MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification},
  author={Yang, Jiazhen and Shi, Yuwei and Bian, Conghao and et al.},
  journal={Scientific Data},
  year={2023}
}
```
## 👤 Author
- Manjul Mayank
