import cv2
import streamlit as st
from PIL import Image
import numpy as np
import time

# Function to detect faces in an image using Haar cascade classifier
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return faces

# Streamlit web application
def main():
    st.title("Face Detection Web App")
    st.write("Upload an image or capture photos from webcam to detect faces")


    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Detection WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Sidebar options
    option = st.sidebar.selectbox("Select an option", ("Upload Image", "Capture from Webcam"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Convert the uploaded image to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Display the uploaded image
            st.image(image, channels="BGR", caption="Uploaded Image")

            # Detect faces in the uploaded image
            faces = detect_faces(image)
            if len(faces) > 0:
                st.write(f"Number of faces detected: {len(faces)}")
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                st.image(image, channels="BGR", caption="Image with Face Detection")
            else:
                st.write("No faces detected in the uploaded image")

    elif option == "Capture from Webcam":
        st.warning("Please grant webcam access to the application.")
        stop = st.button("Stop")  # Button to stop capturing photos
        cam = cv2.VideoCapture(0)
        frame_number = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            frame_number += 1
            # Display the webcam frame
            st.image(frame, channels="BGR", caption="Webcam")
            # Detect faces in the webcam frame
            faces = detect_faces(frame)
            if len(faces) > 0:
                st.write(f"Number of faces detected: {len(faces)}")
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                st.image(frame, channels="BGR", caption="Webcam with Face Detection")
            else:
                st.write("No faces detected in the webcam frame")
            if stop:
                break
            if frame_number % 300 == 0:  # Capture interval fixed at 10 seconds (10 * 30 frames)
                time.sleep(1)

        cam.release()

if __name__ == "__main__":
    main()
