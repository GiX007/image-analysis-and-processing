# Image Analysis and Processing

This repository provides an introduction to image analysis and processing, combining classical techniques with modern machine learning and deep learning approaches. 

---

## Contents

### 1. `classic_image_processing.ipynb`
Explore core image processing operations using OpenCV:
- Intensity transformations (step function)
- Brightness & color enhancement
- Gamma correction for contrast adjustment
- Image sharpening and spatial enhancement
- Edge & corner detection
- Angle estimation and synthetic image generation
  
Uses images from the `data/` folder. 

### 2. `learning_based_pneumonia_detection.ipynb`
Detect pneumonia in chest X-ray images using both ML and DL techniques:
- **Dataset**: Chest X-Ray Images (Pneumonia)
- **ML models**: KNN, Logistic Regression, SVM, Random Forest, Gradient Boosting and XGBoost
- **DL models**: Custom NN, CNN, and pretrained models like VGG16, ResNet18, Xception and EfficientNet

Key steps:
- Thorough data exploration and preprocessing with format checks, resizing, class distribution analysis, etc
- Training and evaluation with accuracy, precision, recall, confusion matrix
- Model comparison with visual insights and performance metrics

### 3. `pneumonia_detection_with_CNN_and_VGG.ipynb`
A fast, end-to-end pneumonia detection pipeline using a **custom CNN** and **VGG16**, covering exploration and preprocessing of the dataset, training, evaluation, and prediction in a compact, 40-cell format.

The notebooks combine clear explanations, input/output image examples, and practical experiments to help you understand both **foundational image processing** and **modern ML/DL applications**.
