import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model and the video file
model = load_model("testTrainModel/model_expressions.keras")
video_file = "emotionMovies.mp4"
cap = cv2.VideoCapture(video_file)

connected, video = cap.read()
print(video.shape)  # Display video dimensions

resize_video = True  # True to prevent exceeding the maximum width
max_width = 600  # Set the maximum width of the saved video

# Resize the video if necessary
if resize_video and video.shape[1] > max_width:
    aspect_ratio = video.shape[1] / video.shape[0]  # Save the aspect ratio to avoid inconsistencies
    video_width = max_width
    video_height = int(video_width / aspect_ratio)  # Calculate height based on aspect ratio and width
else:
    video_width = video.shape[1]
    video_height = video.shape[0]

# Output file
output_file = 'expressionResultVideo.mp4'
fps = 24

# Codec definition
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID', 'MP4V', 'MJPG', 'X264'...

output_video = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))

# Variable to detect only one face, False to detect all (False will not display the chart in the top left corner)
single_face = True

# Haarcascade algorithm for face detection
haarcascade_face_path = 'testTrainModel/data/haarcascade_frontalface_alt.xml'

# Settings for font and text
small_font, medium_font = 0.32, 0.7
font = cv2.FONT_HERSHEY_SIMPLEX
expressions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Function to return the color corresponding to each emotion
def get_color(emotion_index):
    switcher = {
        0: (0, 0, 255),     # Angry
        1: (0, 255, 0),     # Disgust
        2: (204, 50, 153),  # Fear
        3: (0, 255, 255),   # Happy
        4: (255, 191, 0),   # Sad
        5: (80, 127, 255),  # Surprise
        6: (128, 128, 128)  # Neutral
    }
    return switcher.get(emotion_index, (0, 0, 0))  # Black if an unknown emotion is detected

while cv2.waitKey(1) < 0:
    connected, frame = cap.read()

    if not connected:
        break

    start_time = time.time()  # Record start time

    # Resize each frame if necessary
    if resize_video and frame.shape[1] > max_width:
        frame = cv2.resize(frame, (video_width, video_height))

    # Parameters for face detection
    face_cascade = cv2.CascadeClassifier(haarcascade_face_path)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    # If more than one face is detected
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # If single_face is True, return only the largest face
            if single_face and len(faces) > 1:
                max_face = faces[0]
                for face in faces:
                    # Replace with a larger face if found
                    if face[2] * face[3] > max_face[2] * max_face[3]:
                        max_face = face
                # Update variables with the largest face
                face = max_face
                (x, y, w, h) = max_face

            # Region of Interest (ROI) = region where the face is located
            roi = gray_frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))    # Resize to the format used by the model (48x48)
            roi = roi.astype("float") / 255.0  # Normalize
            roi = img_to_array(roi)            # Convert to array
            roi = np.expand_dims(roi, axis=0)  # Change the array shape

            # Make a prediction
            result = model.predict(roi)[0]
            print(result)

            if result is not None:
                if single_face:
                    for (index, (emotion, prob)) in enumerate(zip(expressions, result)):
                        text = "{}: {:.2f}%".format(emotion, prob * 100)  # Emotion and its probability percentage
                        bar_length = int(prob * 150)                       # Generate a bar representing the probability
                        left_margin = 7                                     # Spacing to avoid starting at the edge
                        emotion_color = get_color(index)                   # Save the color corresponding to the emotion
                        if bar_length <= left_margin:
                            bar_length = left_margin + 1  # 1 pixel width to avoid growing leftwards

                        # Generate prediction chart
                        cv2.rectangle(frame, (left_margin, (index * 18) + 7), (150, (index * 18) + 18), (230, 230, 230), -1)
                        cv2.rectangle(frame, (left_margin, (index * 18) + 7), (bar_length, (index * 18) + 18), emotion_color, -1)
                        cv2.putText(frame, text, (15, (index * 18) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)

                # Get the emotion with the highest probability
                final_result = np.argmax(result)

                color = get_color(final_result)

                # Draw a rectangle around the face(s) with the emotion name
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h + 10), color, 3)
                cv2.putText(frame, expressions[final_result], (x, y - 10), font, medium_font, color, 2, cv2.LINE_AA)

    # Show time taken to process the previous frame
    cv2.putText(frame, " Frame processed in {:.2f} seconds".format(time.time() - start_time), (20, video_height - 29), font, small_font, (250, 250, 250), 0, lineType=cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Result', frame)

    # Write the frame to the output file
    output_video.write(frame)

    # Wait to maintain the frame rate
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()