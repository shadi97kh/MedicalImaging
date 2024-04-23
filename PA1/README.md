# U-Net for MRI Image Segmentation

This project implements a U-Net convolutional neural network to segment MRI brain images into different tumor regions. This document provides instructions on how to set up and run the project, along with details on the project structure and usage.

## Project Description

The U-Net model is used to segment MRI brain images to identify different regions of interest, such as necrotic and non-enhancing tumor core, peritumoral edema, and GD-enhancing tumor. This model is implemented using TensorFlow and trained on a dataset of MRI images.

## Installation

### Prerequisites

- Python 3.8 or above
- pip package manager
- Access to a GPU is recommended for training the model efficiently

### Libraries

Install the required Python libraries using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn medpy nibabel SimpleITK
