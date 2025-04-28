# Handwritten-Text-Classifier

A simple handwritten text classifier using PyTorch.

## Requirements

- Python 3.9+
- torch
- torchvision
- matplotlib
- Pillow

Install dependencies with:

```
pip install torch torchvision matplotlib Pillow
```

## Training

Trains a simple neural network on the MNIST dataset.

- Input: 28x28 grayscale images
- Architecture: 1 hidden layer with 128 neurons
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Only 1 epoch

Saves the trained model as `model.pth`.

Run training with:

```
python buildClassifier.py
```

## Prediction

Uses the saved model to predict a digit from a custom image.

- Loads and preprocesses the image (invert, crop, center, resize).
- Displays the processed image.
- Prints the predicted digit.

Run prediction with:

```
python classiferTest.py
```

Make sure your input image is white digits on a black background for best results.

