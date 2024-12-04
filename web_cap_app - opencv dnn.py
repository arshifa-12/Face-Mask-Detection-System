import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load the face mask detection model
model = load_model('model/XceptionModel.keras')

# Load DNN model for face detection
modelFile = "model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "model/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Initialize the main window
window = tk.Tk()
window.title("Face Mask Detection")
window.geometry("650x500")

# Create a label in the GUI to show the camera feed
lmain = tk.Label(window)
lmain.pack()

# Constants
IMG_SIZE = 128

# Initialize camera
cap = cv2.VideoCapture(0)

def get_faces_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    faces = []
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX-startX, endY-startY))
    return faces

def show_frame():
    ret, frame = cap.read()
    if ret:
        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the image
        faces = get_faces_dnn(rgb_image)
        
        # Process each face found
        for (x, y, w, h) in faces:
            face_img = rgb_image[y:y+h, x:x+w]
            resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, IMG_SIZE, IMG_SIZE, 3))
            result = model.predict(reshaped)
            label = np.round(result).astype(int)[0][0]
            
            # Draw rectangle around the face and put label
            cv2.rectangle(rgb_image, (x, y), (x+w, y+h), (0, 255, 0) if label == 0 else (255, 0, 0), 2)
            cv2.putText(rgb_image, "Mask!" if label == 0 else "No Mask", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if label == 0 else (255, 0, 0), 2)
        
        # Convert the image to PIL format
        im_pil = Image.fromarray(rgb_image)
        
        # Convert the PIL image to ImageTk format
        imgtk = ImageTk.PhotoImage(image=im_pil)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)  # Repeat after 10 milliseconds

# Start the GUI and frame update
show_frame()
window.mainloop()

# Cleanup when closing the window
cap.release()
cv2.destroyAllWindows()
