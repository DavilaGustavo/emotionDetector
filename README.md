<p align="center">
  <img src="https://github.com/user-attachments/assets/a40867a1-5482-4c2f-9552-d6e77e10ca10">
</p>

# Facial Emotion Prediction (CNN)

This repository contains a project to predict facial emotions using a Convolutional Neural Network (CNN) with the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) (download necessary if trying to train the model, place it in `documentation/src/testTrainModel/data/fer2013`). The code is written in Python and leverages popular machine learning libraries. The goal is to create an efficient model to classify facial expressions into different emotional categories.

## Implemented Features

1. ### Data Preprocessing
   - Loading the FER2013 dataset and preparing the data, including image normalization and conversion.

2. ### Model Training
   - Configuration and training of a CNN to classify facial expressions. The model is trained using various regularization techniques, such as normalization and dropout, to improve generalization.

3. ### Model Evaluation
   - Evaluation of the model using metrics like accuracy and performance visualization through a confusion matrix. Qualitative analysis of incorrect predictions to better understand the model's errors.

4. ### Model Saving and Loading
   - Saving the model architecture in JSON format and weights in Keras format. Loading the saved model for future predictions and analysis.

5. ### Emotion Prediction
   - Using the trained model to detect and classify facial emotions in videos and images, displaying the results.

## Tools Used
- **Python**: Main programming language.
- **Python Libraries**: TensorFlow, Keras, NumPy, Matplotlib, scikit-learn, OpenCV.
- **Development Environment**: Any Python IDE such as Visual Studio Code.

## Main Project Structure

- **`videoDetection.py`**: Main script to load the model and apply emotion detection to the video emotionMovies.
- **`emotionMovies.mp4 && expressionResultVideo.mp4`**: Video files representing before and after being processed by the model.
- **`testTrainModel/`**: Directory containing the model construction process, the saved model, and its corresponding weights.
  - `modelTrain.py`: Script responsible for training the model.
  - `modelTest.py`: Script responsible for testing the model.
  - `model_expression.json`: File containing the model architecture.
  - `model_expression.keras_`: File containing the trained model weights.

## How to Use

1. **Clone the repository**:
    - git clone <repository URL>

2. **Install the required dependencies**:
    - pip install tensorflow numpy matplotlib scikit-learn opencv-python

3. **Ensure the model files are in the testTrainModel/ directory**:
  - model_expression.json
  - model_expression.keras

4. **Run the main script**:
  - python videoDetection.py
    - The script will load the model, predict emotions for each frame in the video, and produce the final output file.

5. **Changes**:
  - Feel free to modify the code to adjust model parameters or experiment with different datasets.

## Contributions

Contributions are welcome! Feel free to submit pull requests with improvements, bug fixes, or new features. For discussions and suggestions, please open an issue.