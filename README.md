# AgTeach ML Backend - Plant Disease Detection Model
This repository contains code and instructions for building a Convolutional Neural Network (CNN) model to detect various plant diseases using the PlantVillage dataset. The project involves dataset preprocessing, model training, validation, and deployment into a full-stack application using React.js for the frontend and Flask for the backend.


Experience [here](https://agteach.site/agai)


## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training and Evaluation](#training-and-evaluation)
6. [Model Integration](#model-integration)
7. [Deployment](#deployment)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Results](#results)
11. [Future Improvements](#future-improvements)
12. [References](#references)


**Project Structure**
```bash
├── metadata 
   └── disease_list.json
├── model 
   └── plant_disease_cnn_x.keras
├── README.md
└── requirements.txt
├── app.py
```

---

## Overview

This project aims to predict plant diseases using images. A custom CNN model was built and trained using TensorFlow/Keras. The dataset used is the [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data) dataset, which contains various classes representing different diseases and healthy plant conditions.

The trained model is integrated with a web application to allow users to upload plant images and receive disease predictions in real time.

## Dataset

The dataset used is the **PlantVillage dataset**, containing labeled images of leaves in various health conditions (healthy or diseased). There are 38 classes, including common diseases like **Tomato___Bacterial_spot**, **Potato___Late_blight**, and **Apple___Cedar_apple_rust**.

- **Dataset structure**: 
  - 38 folders, each representing a different plant disease class or a healthy plant.
  - Images are in `.jpg` format.

- **Data distribution**:
  - Some classes have more images than others (e.g., `Tomato___Bacterial_spot` has over 2000 images, while `Apple___Cedar_apple_rust` has 275 images).
  
### Splitting the Dataset
The dataset was split into **train**, **validation**, and **test** sets:
- Train: 80%
- Validation: 10%
- Test: 10%

To balance classes with very few images, data augmentation techniques were applied.

## Preprocessing

The preprocessing pipeline includes:
1. **Image resizing**: All images were resized to 150x150 pixels.
2. **Normalization**: Pixel values were normalized to a [0, 1] range.
3. **Augmentation**: Data augmentation techniques such as rotation, zoom, and horizontal flipping were used to improve the model's generalization.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2]
)
```

## Model Architecture

The CNN architecture consists of four convolutional layers followed by max-pooling layers and a fully connected layer at the end. A softmax activation function is used in the output layer to classify among 38 classes.

```python
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Regularization to avoid overfitting
    layers.Dense(38, activation='softmax')  # 38 classes in the dataset
])
```

The model is compiled using `Adam` optimizer and categorical cross-entropy as the loss function.

## Training and Evaluation

The model was trained for 20 epochs using the following code:

```python
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)
```


### Metrics:
- **Loss**: Categorical cross-entropy
- **Accuracy**: Used as the primary evaluation metric

## Model Integration

The trained CNN model was integrated into a full-stack web application using:
- **React.js**: Frontend for uploading images and displaying predictions.
- **Flask**: Backend API for handling image uploads and making predictions using the trained TensorFlow model.
- **TensorFlow.js**: For running the trained model in a browser if required.

Model was saved as follows:
```python
model.save('./plant_disease_prediction_model.keras')
```

## Deployment

The web application was deployed on an AWS EC2 instance using Nginx as the reverse proxy. The TensorFlow model is loaded via the Flask API running on the backend to serve predictions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/plant-disease-detection.git
    ```
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app locally:
    ```bash
    python app.py
    ```

## Usage

- Users can upload images of plant leaves through the web interface.
- The CNN model will process the image and return a predicted disease label.

## Results

After 20 epochs of training, the model achieved an accuracy of 93% on the validation set.

## Future Improvements

1. **Class balancing**: Some classes have fewer images, which could affect the model's performance. Techniques such as oversampling or generating synthetic images can help balance the dataset.
2. **Fine-tuning**: Consider experimenting with transfer learning using pre-trained models like ResNet or EfficientNet.


## References

- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data)
- TensorFlow/Keras documentation

