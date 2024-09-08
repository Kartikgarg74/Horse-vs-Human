# Horse vs Human Image Classification

This repository contains a TensorFlow-based project that classifies images as either **horses** or **humans**. The model is trained on a dataset of images of horses and humans, and it uses a Convolutional Neural Network (CNN) for classification.

## Features

- **Dataset**: The dataset consists of horse and human images, which are split into training and validation sets.
- **Model Architecture**: The project utilizes a Convolutional Neural Network (CNN) for image classification.
- **Data Preprocessing**: Images are resized and normalized before being fed into the model.
- **Training and Validation**: The model is trained on a set of labeled images and validated on a separate dataset.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Other libraries: OpenCV, NumPy, Matplotlib

To install TensorFlow, use:

```bash
pip install tensorflow
```

### Dataset

The training and validation datasets are downloaded directly from Google Cloud Storage.

- Training Set: Contains images of horses and humans.
- Validation Set: Used for model validation during training.

### How to Run

1. Clone this repository:

```bash
git clone https://github.com/Kartikgarg74/horse-vs-human.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook horse_vs_human.ipynb
```

4. Follow the steps in the notebook to train the model and evaluate its performance.

### Project Workflow

1. **Download Dataset**: The training and validation datasets are automatically downloaded using the `wget` command.
2. **Unzip Files**: The downloaded `.zip` files are unzipped and extracted into the appropriate directories.
3. **Model Training**: A CNN is trained on the dataset with appropriate layers for feature extraction and classification.
4. **Evaluation**: The model is evaluated using the validation set to measure its accuracy.

## Model Architecture

- **Convolutional Layers**: Used for feature extraction from the images.
- **Max Pooling Layers**: For down-sampling the feature maps.
- **Fully Connected Layers**: For classification.
- **Output Layer**: A sigmoid activation function is used to classify between horses and humans.

## Future Improvements

- **Model Tuning**: Hyperparameter tuning to improve model accuracy.
- **Transfer Learning**: Implementing a pre-trained model to boost performance.
- **Data Augmentation**: Increasing dataset size using augmentation techniques.
