import cv2
import numpy as np
from tensorflow.keras.models import model_from_json, load_model
from sklearn.metrics import confusion_matrix

# Load the model
model_json_file = "model_expressions.json"
model_weights_file = "model_expressions.keras"

# Load the test data
x_test = np.load('mod_xtest.npy')
y_test = np.load('mod_ytest.npy')

# Check the accuracy using the test data
true_labels = []
pred_labels = []
with open(model_json_file, 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_weights_file)
y_pred = loaded_model.predict(x_test)
y_pred_list = y_pred.tolist()
y_test_list = y_test.tolist()
correct_predictions = 0

for i in range(len(y_test)):
    predicted_prob = max(y_pred_list[i])
    true_prob = max(y_test_list[i])
    pred_labels.append(y_pred_list[i].index(predicted_prob))
    true_labels.append(y_test_list[i].index(true_prob))
    if y_pred_list[i].index(predicted_prob) == y_test_list[i].index(true_prob):
        correct_predictions += 1

accuracy = (correct_predictions / len(y_test)) * 100
np.save('true_labels_mod01', true_labels)
np.save('pred_labels_mod01', pred_labels)
print("Test set accuracy: " + str(accuracy) + "%")

# Confusion matrix
y_true = np.load('true_labels_mod01.npy')
y_pred = np.load('pred_labels_mod01.npy')
conf_matrix = confusion_matrix(y_true, y_pred)
expressions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
title = 'Confusion Matrix'
print(conf_matrix)

# Import an image for testing
image = cv2.imread("data/test.png")

# Load the model
model = load_model(model_weights_file)
scores = model.evaluate(np.array(x_test), np.array(y_test), batch_size=16)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))

# Display the expression
expressions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
original_image = image.copy()
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
detected_faces = face_cascade.detectMultiScale(gray_image, 1.1, 3)

for (x, y, w, h) in detected_faces:
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_gray = gray_image[y:y + h, x:x + w]
    roi_gray = roi_gray.astype("float") / 255.0
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, 
                  norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
    prediction = model.predict(cropped_img)[0]
    cv2.putText(original_image, expressions[int(np.argmax(prediction))], (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
# Display the result
cv2.imshow('Result', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()