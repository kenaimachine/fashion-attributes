
# Fashion Attributes Classification Challenges

This repository contains the implementation of a deep learning-based solution for **multi-label classification of fashion attributes**. It was developed as part of the **AI6126: Advanced Computer Vision** project. The project employs advanced techniques such as the **EfficientNet-V2-L architecture**, **Convolutional Block Attention Module (CBAM)**, and **Weighted Focal Loss** to address the complexity of fashion attribute classification.

## Project Overview

The goal of this project is to accurately predict multiple attributes of fashion items using a dataset of annotated fashion images. This solution has applications in fashion retail, e-commerce, and content-based image retrieval systems. Key highlights include:

- **Custom Neural Network**: Incorporates attention mechanisms and independent classifiers for multi-label prediction.
- **Advanced Data Preprocessing**: Features robust data augmentation and oversampling to address class imbalance.
- **Custom Loss Function**: Employs a Weighted Focal Loss to improve performance on underrepresented attributes.

---

## Features

1. **Efficient Base Model**: Uses the EfficientNet-V2-L pretrained model for feature extraction.
2. **CBAM Attention**: Enhances the model's ability to focus on important features in both channel and spatial dimensions.
3. **Multi-Label Prediction**: Independent classifier branches for each attribute category enable accurate multi-label predictions.
4. **Class Imbalance Handling**: Oversampling of minority classes and weighted loss functions.
5. **Flexible and Modular**: Easily customizable for different datasets and attribute categories.

---

## File Structure

- `Fashion_attributes_classifier.ipynb`: Jupyter Notebook with the main implementation.
- `prediction.txt`: Sample predictions made by the model.
- `best_model.pth`: Pretrained model weights
- `Readme.txt`: Brief description of the project.
- `Project1_Report.docx`: Comprehensive project report.
- `Screen_shot_codalab.png`: Screenshot of the evaluation results on CodaLab.

---

## Dataset and Preprocessing

The dataset contains 6,000 images, each annotated with attributes from six categories. Preprocessing involves:
- Resizing images to 224x224 pixels.
- Data augmentation techniques: random cropping, flipping, color jittering, and Gaussian blurring.
- Oversampling minority classes to mitigate class imbalance.

---

## Model Architecture

### **AttributeModel**
The `AttributeModel` consists of:
1. **EfficientNet-V2-L**: Base feature extractor pretrained on ImageNet.
2. **CBAM**: Channel and spatial attention modules to refine feature maps.
3. **Independent Classifiers**: Separate classifiers for each attribute category with Adaptive Average Pooling, Dense layers, and Dropout for regularization.

### **Loss Function**
The **Weighted Focal Loss** is used to address class imbalance by assigning higher weights to underrepresented attributes. This enhances the model's focus on challenging examples.

---

## Training and Evaluation

### Training Details
- Optimizer: Adam
- Learning Rate: 0.0001 with cosine annealing scheduler and warm restarts
- Batch Size: 32
- Number of Epochs: 15-30
- Hardware: Apple MacBook Pro M3 Max (64 GB RAM, 18-core GPU)

### Metrics
- Accuracy
- Average Class Accuracy
- Loss (Training and Validation)

Training curves for these metrics are included in the report to visualize the model's performance.

---

## Results

The model demonstrates strong performance in predicting multiple fashion attributes. Detailed evaluation metrics and insights are included in the `Project1_Report.docx`.

---

## Usage Instructions

### Prerequisites
- Python 3.8+
- Libraries: PyTorch, NumPy, Matplotlib, scikit-learn

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/kenaimachine/fashion-attributes.git
   cd fashion-attributes-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Fashion_attributes_classifier.ipynb
   ```
4. Download the pretrained model (`best_model.pth`)

---

## Limitations and Future Work

- Explore state-of-the-art architectures like Vision Transformers.
- Experiment with additional data augmentation techniques (e.g., MixUp, CutOut).
- Fine-tune hyperparameters using cross-validation or grid search.

---

## References

1. Tan, Mingxing, and Quoc V. Le. "EfficientNetV2: Smaller Models and Faster Training." arXiv (2021).
2. Woo, Sanghyun, et al. "CBAM: Convolutional Block Attention Module." ECCV (2018).
3. Lin, Tsung-Yi, et al. "Focal Loss for Dense Object Detection." ICCV (2017).

---

## License

This project is licensed under the MIT License.

---

