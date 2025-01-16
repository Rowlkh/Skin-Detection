# Skin Detection using Image Processing and Classification

## Overview

This project implements a skin detection system using various image processing techniques and machine learning models. It aims to classify image pixels as skin or non-skin, which has numerous applications, including medical diagnosis, augmented reality, and image editing.

## Project Objectives

- Implement a comprehensive pipeline for skin detection.
- Evaluate the performance of different algorithms in classifying skin areas.
- Visualize the results of each stage to better understand the image processing workflow.

## Folder Structure

The project is organized into the following files:

- **image_acquisition.py**: Handles loading and resizing images from specified directories.
- **preprocessing.py**: Contains functions for image enhancement and restoration.
- **segmentation.py**: Implements algorithms for segmenting skin areas from images.
- **ml_models.py**: Includes machine learning models for feature extraction and classification.
- **main.py**: The main script that runs the entire pipeline and executes all steps.

## Setup and Installation

### Requirements

Make sure you have the following Python libraries installed:

- OpenCV
- NumPy
- scikit-learn
- Matplotlib

You can install them using `pip`:

```bash
pip install opencv-python numpy scikit-learn matplotlib
```

### Running the Project

To run the project, simply execute the `main.py` script:

```bash
python main.py
```

This will load the images, apply preprocessing, perform segmentation, extract features, and then train the machine learning models to classify skin regions.

## Methodology

### Image Acquisition

Images are acquired from two directories: one containing skin images and the other containing non-skin images. The `load_images_from_folder` function is used to load and resize images to 800x600 pixels. The function also ensures that grayscale images are converted to RGB format.

### Preprocessing

- **Gaussian Blur**: Reduces image noise to enhance skin detection accuracy.
- **Gamma Correction**: Enhances image brightness and contrast for better skin detection.

### Segmentation

- **Color Space Conversion**: Converts the image from BGR to HSV color space, which is more effective for identifying skin tones.
- **Skin Tone Range**: Defines a range for typical skin tones in HSV values.
- **Morphological Operations**: Removes noise and fills small holes in the detected skin areas.
- **Mask Application**: A binary mask is created where skin areas are white, and non-skin areas are black.

### Feature Extraction

Features are extracted from the segmented images, including:

- **Color Space Conversion**: Converts images to HSV color space.
- **Histogram Calculation**: Computes histograms for each HSV channel.
- **Mean Intensity**: Measures overall brightness.
- **Canny Edge Detection**: Detects edges in the image.

### Model Training and Evaluation

The models (Logistic Regression and Random Forest) are trained using extracted features:

- **Data Splitting**: The dataset is split into 70% training and 30% testing.
- **Model Training**: Both models are trained using the training set.
- **Model Evaluation**: Performance is assessed using a confusion matrix, ROC curve, and classification report.

### Testing and Results

The models are evaluated using test images. Each test image undergoes the following steps:

1. **Preprocessing**
2. **Segmentation**
3. **Feature Extraction**
4. **Classification**

The results of the classification are visualized for both models (Random Forest and Logistic Regression).

## Results

- **Random Forest Classifier**: Shows how well the Random Forest model performed on the test set.
- **Logistic Regression**: Displays the performance of the Logistic Regression model on the same test set.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

You can simply copy and paste this into your **README.md** file in your GitHub repository.
