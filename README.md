Certainly! Here's a structured README file based on the methodology and details you provided for your tuberculosis detection project using CNN models:

---

# Tuberculosis Detection using CNN Models

This project focuses on detecting tuberculosis (TB) from chest X-ray images using Convolutional Neural Network (CNN) architectures including UNET, ResNet, DenseNet, and VGG16. The models are trained and evaluated using the "Chest X-ray Masks and Labels" dataset from Kaggle, which includes:

- **Image**: Paths to X-ray images
- **Mask**: Paths to corresponding lung masks
- **Gender**: Patient gender information
- **Age**: Patient age data
- **Diagnosis**: Binary classification (normal or tuberculosis) with detailed TB subtypes.

## Methodology

### Data Collection and Preprocessing

#### Dataset Description
The dataset provides chest X-ray images with corresponding lung masks, facilitating precise TB detection research.

#### Data Preprocessing
1. **Image and Mask Reading**: Utilized OpenCV to read X-ray images and their masks, isolating lung areas crucial for TB detection.
   
2. **Image Resizing**: Standardized all images to 256x256 pixels for uniform input size to CNN models.
   
3. **Pixel Normalization**: Scaled pixel values from 0-255 to 0-1 for efficient model training and convergence.
   
4. **Data Augmentation**: Implemented real-time augmentation (rotation, zoom, flips, shifts) to enhance dataset variability and model generalization.

### Model Architectures

#### UNET
- Adapted for binary TB classification from chest X-rays.
- Utilized encoder-decoder structure with skip connections for precise localization and high performance.

#### ResNet
- Employed ResNet50 for feature extraction and classification.
- Pre-trained on ImageNet for robust feature learning, fine-tuned for TB detection.

#### DenseNet
- Leveraged DenseNet169 for feature reuse and efficient gradient flow.
- Integrated custom segmentation head for TB classification from chest X-rays.

#### VGG16
- Used VGG16 with modifications for TB detection.
- Pre-trained on ImageNet and fine-tuned with additional layers for improved classification.

### Training and Evaluation

- **Loss Function**: Binary cross-entropy (UNET, ResNet, VGG16) and dice loss (DenseNet) for model optimization.
- **Optimizer**: Adam optimizer with adaptive learning rate.
- **Metrics**: Evaluated models using accuracy, precision, recall, F1 score, and ROC-AUC.

### Implementation Details

- **Training Strategy**: Employed callbacks (ModelCheckpoint, ReduceLROnPlateau) to optimize training efficiency and model performance.
- **Performance Evaluation**: Conducted comprehensive evaluation on validation set to assess model accuracy and robustness.




---

## Results of Each CNN Model

### U-Net
- **Validation Loss:** 0.0803
- **Validation Accuracy:** 0.9707
- **Pixel Accuracy:** 0.9715
- **F1 Score:** 0.9424
- **Precision:** 0.9497
- **Recall:** 0.9352
- **Jaccard Index (IoU):** 0.8910
- **Dice Coefficient:** 0.9424

The U-Net model demonstrated exceptional performance in segmentation tasks, achieving high accuracy, F1 score, and pixel accuracy. It excelled in accurately identifying TB from chest X-ray images, with robust precision and recall.

### ResNet
- **Precision:** 0.8485
- **Recall:** 0.8684
- **F1 Score:** 0.8583
- **Loss:** 0.1420
- **Dice Coefficient:** 0.8466
- **Accuracy:** 0.9256
- **Validation Accuracy:** 0.9256

ResNet performed well in classification tasks, showing balanced precision and recall. It effectively distinguished between normal and TB cases, as evidenced by its confusion matrix results and overall accuracy.

### DenseNet169
- **F1 Score:** 0.8132
- **Accuracy:** 0.9012
- **Validation F1 Score:** 0.8094
- **Validation Accuracy:** 0.9014
- **Precision:** 0.5266
- **Recall:** 0.9548

DenseNet169 achieved moderate performance with high recall but lower precision, indicating good sensitivity but with more false positives. It demonstrated stable learning and performance consistency between training and validation sets.

### VGG16
- **Validation Loss:** 0.3708
- **Validation Accuracy:** 0.8416
- **Accuracy:** 0.8416
- **Precision:** 0.8500
- **Recall:** 0.8793
- **F1 Score:** 0.8644

VGG16 exhibited respectable performance but was outperformed by other models. Its simpler architecture potentially limited its ability to capture detailed features in lung X-rays.

### Comparative Analysis

The performance of each model varied significantly across different metrics, reflecting their unique strengths and weaknesses in tuberculosis detection. U-Net excelled in segmentation tasks with superior accuracy and precision in identifying TB from chest X-ray images. ResNet showed strong capabilities in classification, distinguishing between normal and TB cases effectively. DenseNet169 demonstrated good sensitivity but lower specificity compared to U-Net and ResNet. VGG16, while competent, did not match the performance levels of the other architectures due to its simpler design.

