# Bacteria Classification Model

This project aims to classify bacteria images into three categories: **cocci**, **bacilli**, and **spirilla**. The model uses a Convolutional Neural Network (CNN) trained on labeled images to achieve high accuracy in distinguishing these bacterial shapes.

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Predictions](#predictions)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [Usage](#usage)

## Dataset

The dataset is organized into three sets:
- **Train**: Contains images in subfolders labeled as `cocci`, `bacilli`, and `spirilla`.
- **Validation**: Also structured into subfolders by class, used for evaluating model performance during training.
- **Test**: Contains unlabeled images for which the model will predict labels.

Each image class is represented by:
- `0` for cocci
- `1` for bacilli
- `2` for spirilla

### File Structure

The project folder structure is as follows:
```
bacteria_classification/
│
├── split_dataset/
│   ├── train/
│   │   ├── cocci/
│   │   ├── bacilli/
│   │   └── spirilla/
│   ├── validation/
│   │   ├── cocci/
│   │   ├── bacilli/
│   │   └── spirilla/
│   └── test/  # Contains unlabeled images
└── test_filenames.txt  # List of filenames in the test set
```

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:
1. **Convolutional Layers**: Three convolutional layers with ReLU activation and MaxPooling.
2. **Flatten Layer**: Converts the 2D matrix to a vector for input into fully connected layers.
3. **Dense Layers**: One fully connected (dense) layer followed by a Dropout layer for regularization.
4. **Output Layer**: A softmax layer with three units for classification.

## Training

The model is trained using images in the `train` folder, with validation using images in the `validation` folder. During training:
- **Loss Function**: Categorical Cross-Entropy (for multiclass classification)
- **Optimizer**: Adam
- **Batch Size**: 32
- **Image Size**: 128x128 pixels
- **Epochs**: 10 (can be adjusted)

## Predictions

The trained model is used to classify images in the `test` folder. Predictions are saved in `predictions.csv` in the following format:

| filename          | label |
|-------------------|-------|
| `image1.jpg`      | 0     |
| `image2.jpg`      | 1     |
| `image3.jpg`      | 2     |

where `label` corresponds to the predicted class (`0` for cocci, `1` for bacilli, and `2` for spirilla).

## Evaluation

The model's performance is evaluated using the validation set, and the following metrics are calculated:
- **Accuracy**: Proportion of correctly classified images.
- **Classification Report**: Precision, recall, and F1-score for each class.

## Requirements

To run this project, you’ll need the following libraries:
- `tensorflow`
- `numpy`
- `pandas`
- `tqdm`

Install the required packages:
```bash
pip install tensorflow numpy pandas tqdm
```

## Usage

1. **Prepare the Dataset**: Ensure the dataset is organized as described above.
2. **Run the Training Script**: Train the model using the training and validation sets.
3. **Generate Predictions**: The model will output predictions on the test set and save them in `predictions.csv`.

### Example Command

To run the entire script:
```python
python bacteria_classification.py
```

### Output

- **Model File**: The trained model is saved as `model.h5`.
- **Training Labels**: A `training_labels.txt` file with labels for the training images.
- **Predictions**: The test set predictions are saved in `predictions.csv`.

## Author

This model and script were developed by Caden Picard.

