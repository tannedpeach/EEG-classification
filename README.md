# EEG Seizure Type Classification

A machine learning system for classifying seven different types of epileptic seizures from EEG (electroencephalography) signals using deep learning and traditional machine learning approaches.

## Overview

This project implements automated seizure type classification using the TUH Seizure Corpus dataset. It leverages transfer learning with ResNet50V2 (pretrained on ImageNet) adapted for EEG signal analysis, as well as traditional machine learning models for comparison.

## Seizure Types Classified

The system classifies seven distinct seizure types:
- **TNSZ** - Tonic Seizures
- **SPSZ** - Simple Partial Seizures
- **ABSZ** - Absence Seizures
- **TCSZ** - Tonic-Clonic Seizures
- **CPSZ** - Complex Partial Seizures
- **GNSZ** - Generalized Non-Specific Seizures
- **FNSZ** - Focal Non-Specific Seizures

## Features

- **Transfer Learning**: Utilizes ResNet50V2 architecture adapted for EEG signal classification
- **Multiple ML Algorithms**: Implements K-Nearest Neighbors, Decision Trees, and Random Forest classifiers
- **K-Fold Cross-Validation**: Patient-wise 3-fold cross-validation to ensure robust model evaluation
- **Class Imbalance Handling**: Incorporates oversampling and undersampling techniques with class weighting
- **Comprehensive Evaluation**: Generates confusion matrices, precision, recall, F1-scores, and AUC metrics
- **TensorBoard Integration**: Real-time training visualization and performance monitoring
- **FFT Preprocessing**: Uses Fast Fourier Transform for frequency domain feature extraction

## Architecture

### Deep Learning Model (CNN)
- **Base Model**: ResNet50V2 (pretrained on ImageNet)
- **Input Shape**: 156 × 900 × 3 (EEG windows × columns × channels)
- **Preprocessing**: EEG signals are stacked into 3 channels to match ImageNet input format
- **Output**: 7-class softmax classifier
- **Regularization**: Dropout (0.2) and Global Average Pooling

### Traditional ML Models
- K-Nearest Neighbors (KNN) with hyperparameter tuning
- Decision Tree Classifier
- Random Forest Classifier

## Dataset

The project uses the TUH Seizure Corpus with FFT-transformed EEG data:
- **Window Length**: 1 second
- **Window Step**: 0.5 seconds
- **Sampling Frequency**: 250 Hz
- **FFT Range**: 1-24 Hz
- **Features**: Time-frequency correlation features

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EEG-classification.git
cd EEG-classification

# Install dependencies
pip install tensorflow tensorflow-addons
pip install scikit-learn numpy pandas matplotlib seaborn
pip install dill pickle5
```

## Usage

### Training CNN Model

```bash
python cnn.py
```

The script will:
1. Load preprocessed EEG data and cross-validation splits
2. Calculate class weights to handle imbalanced data
3. Train the model using K-fold cross-validation
4. Generate confusion matrices and performance metrics
5. Save the trained model weights

### Training Traditional ML Models

```bash
python model.py -c <cross_val_file> -d <data_directory>
```

Example:
```bash
python model.py -c ./data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl \
                -d "/path/to/fft_seizures_data"
```

### Visualization

```bash
# View training progress with TensorBoard
tensorboard --logdir=logs/fit

# Visualize results
python visualizer.py
```

## Model Performance

The model outputs the following metrics per fold:
- Categorical Accuracy
- Precision (per class)
- Recall (per class)
- AUC (Area Under Curve)
- F1-Score (macro and weighted)
- Confusion Matrix

Results are logged to TensorBoard for visualization and comparison.

## Project Structure

```
EEG-classification/
├── cnn.py                    # Main CNN training script with ResNet50V2
├── cnn_params.py             # Model hyperparameters and constants
├── model.py                  # Traditional ML models (KNN, Decision Tree, Random Forest)
├── training_subspace.py      # Training subspace creation utility
├── util.py                   # Data loading and preprocessing utilities
├── tensorflow_utils.py       # TensorFlow helper functions
├── visualizer.py             # Visualization tools
├── testing.py                # Model evaluation script
└── README.md                 # This file
```

## Key Technical Details

- **Batch Size**: 2 (adjustable based on GPU memory)
- **Epochs**: Configurable with early stopping based on validation AUC
- **Optimizer**: Adam with learning rate of 0.1
- **Loss Function**: Categorical Cross-Entropy
- **Data Augmentation**: Oversampling minority classes and undersampling majority classes

## Acknowledgments

This project is based on research in automated seizure classification and utilizes the TUH Seizure Corpus dataset. The implementation draws inspiration from:
- [IBM Seizure Type Classification TUH](https://github.com/IBM/seizure-type-classification-tuh)
- Transfer learning techniques from the TensorFlow/Keras ecosystem

## Future Improvements

- [ ] Implement attention mechanisms for better feature extraction
- [ ] Experiment with other pretrained architectures (EfficientNet, Vision Transformers)
- [ ] Add real-time seizure detection capabilities
- [ ] Develop a web interface for clinical use
- [ ] Expand to additional seizure types and datasets

## License

This project is available for academic and research purposes.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This project requires access to the TUH Seizure Corpus dataset and preprocessed FFT data. Please ensure you have the necessary permissions and data before running the code.
