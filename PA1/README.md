# Pneumonia Classification from Chest X-Rays

This project involves the development of a deep learning model to classify chest X-ray images into two categories: Normal and Pneumonia. We explore two approaches: training a model from scratch and fine-tuning a pre-trained ResNet-18 model.

## Dataset

The dataset comprises chest X-ray images categorized into Normal and Pneumonia, sourced from [Kaggle's Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- Matplotlib
- Google Colab (for training with GPU support)

## Usage

1. Clone this repository to your local machine or Google Colab environment.
2. Mount your Google Drive to access the dataset (if using Colab). If using Jupyter Notebook please make sure you modify the dataset location in the script.
3. Run the Jupyter Notebook or Python script to train the model and evaluate its performance.

## Approach

### Task 1.1: Training from Scratch

We train a modified ResNet-18 model from scratch, adapting the final layer to output two classes. The model is trained over 20 epochs with a learning rate of 0.001 and momentum of 0.9.

### Task 1.2: Fine-tuning a Pre-trained Model

We fine-tune a pre-trained ResNet-18 model, replacing the final layer and training it on our dataset. This approach leverages transfer learning to improve performance.

## Results

The models' performance is evaluated based on training/validation loss, overall test accuracy, and class-specific accuracy. We also visualize misclassified images to gain insights into the model's predictions.

## Future Work

Further improvements could include more extensive hyperparameter tuning, exploring more complex architectures, and employing advanced data augmentation techniques.

