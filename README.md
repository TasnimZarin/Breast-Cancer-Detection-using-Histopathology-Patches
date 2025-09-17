# 🩺 Breast Cancer Detection using Histopathology Patches

A deep learning project to classify breast histopathology images into **Non-invasive Ductal Carcinoma (IDC–, Normal)** and **Invasive Ductal Carcinoma (IDC+, Cancerous)**.  
We applied **transfer learning** with multiple CNN architectures (**VGG16, ResNet, InceptionV3**) and compared their performance.

---

## 📚 Project Overview
- **Goal:** Early detection of breast cancer through automated classification of histopathology images.  
- **Problem Statement:** Distinguish between IDC– (non-invasive) and IDC+ (invasive) carcinoma using deep learning.  
- **Classes:**  
  - `0` → Non-invasive Ductal Carcinoma (Normal)  
  - `1` → Invasive Ductal Carcinoma (Cancerous)  

---

## 🧾 Dataset
- **Source:** [Breast Histopathology Images (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)  
- **Original Size:** ~277,524 image patches  
  - IDC– (Normal): 198,738 patches  
  - IDC+ (Cancerous): 78,786 patches  
- **Subset for experiments:** Balanced dataset of **4,000 images**  
  - 2,000 IDC–  
  - 2,000 IDC+  
- **Preprocessing:**  
  - Images resized to **224×224×3 (RGB)**  
  - Balanced dataset to remove class bias  
  - Labels encoded: `0 = Normal`, `1 = Cancerous`

---

## 🧠 Models Used
We evaluated **three CNN architectures** using **transfer learning**:

### 🔹 VGG16
- Pretrained on ImageNet, last dense layer modified to output 1 with **Sigmoid activation**.  
- **Accuracy:** ~80%  

### 🔹 ResNet (Residual Network)
- Used skip connections to address the **vanishing gradient problem**.  
- **Accuracy:** ~84%  

### 🔹 InceptionV3 (Best Performer)
- Transfer learning: froze most layers, fine-tuned last 3 layers.  
- Added **Dropout (0.5)** + new dense layers.  
- **EarlyStopping** (patience=5) + ModelCheckpoint to prevent over/underfitting.  
- **Accuracy:** ~87%  

---

## ⚙️ Training Setup
- **Frameworks & Libraries:**  
  - PyTorch  
  - TensorFlow/Keras  
  - Scikit-learn  
  - NumPy, Pandas  
  - Matplotlib, Seaborn  
  - OpenCV, PIL  

- **Optimizer:** Adam (`lr = 1e-3`)  
- **Loss Function:** Binary Cross-Entropy  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  

---

## 📈 Results Comparison
| Model      | Training Accuracy | Testing Accuracy |
|------------|-------------------|------------------|
| **VGG16**  | ~81%              | ~80%             |
| **ResNet** | ~85%              | ~84%             |
| **InceptionV3** | ~87%        | ~87%             |

- InceptionV3 performed best, achieving **87% accuracy**.  
- ResNet performed well with 84%, while VGG16 gave solid baseline performance at 80%.  
- Confusion matrices and classification reports showed InceptionV3 consistently predicted IDC+ and IDC– with higher precision/recall.  

---

## ✅ Key Features
- Used **transfer learning** with VGG16, ResNet, and InceptionV3.  
- Balanced dataset to prevent class bias.  
- Implemented **Dropout, EarlyStopping, and fine-tuning** for better generalization.  
- Achieved **up to 87% accuracy** (InceptionV3).  
