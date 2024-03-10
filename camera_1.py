import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('model_1.h5')

# Define the emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'contempt'}

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to predict emotion from the input face image
def predict_emotion(face_img):
    # Preprocess the face image
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = face_img / 255.0

    # Predict the emotion
    emotions = model.predict(face_img)
    predicted_label = np.argmax(emotions)

    return emotion_labels[predicted_label]

# Function to detect faces in the input frame and predict emotions
def detect_faces(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and predict emotions
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        emotion_label = predict_emotion(face_region)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Function to start the camera and perform emotion detection
def start_emotion_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_faces(frame)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == '__main__':
    start_emotion_detection()
