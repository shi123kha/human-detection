import os
import time
import cv2
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import PersonDetector



class StreamLiteApp:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.person_detector = self.executor.submit(PersonDetector).result()


    def run(self):
        st.title("Person Detection using YOLO")
        st.write("Upload an image or select webcam to detect persons")

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                self.temp_file = self.save_uploaded_file(uploaded_file)
                img = self.load_and_process_image(self.temp_file)
                detector.detect_persons(img)
            except Exception as e:
                st.write(f"Error processing image: {str(e)}")

        if st.button("Use Webcam"):
            self.start_webcam()

        if self.cap is not None:
            with st.spinner("Processing..."):
                while True:
                    if self.stop:
                        self.cap.release()
                        self.cap = None
                        self.stop = False
                        st.write("Webcam stopped.")
                        break

                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    self.frame_number += 1
                    img = self.load_and_process_image(frame)
                    detector.detect_persons(img)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame, channels="RGB", use_column_width=True)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    if self.stop:
                        break
                    if self.frame_number % 300 == 0:  # Capture interval fixed at 10 seconds (10 * 30 frames)
                        time.sleep(1)

            st.write("Finished processing.")

    def save_uploaded_file(self, uploaded_file):
        if not os.path.exists("temp"):
            os.makedirs("temp")
        temp_file = os.path.join("temp", uploaded_file.name)
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        return temp_file

    def load_and_process_image(self, file_path):
        img = cv2.imread(file_path)
        height, width, _ = img.shape
        max_dim = max(height, width)
        if max_dim > 800:
            scale_factor = 800 / max_dim
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        return img

    def start_webcam(self):
        try:
            self.cap = cv2.VideoCapture(0)
            self.frame_number = 0
        except Exception as e:
            st.write(f"Error starting webcam: {str(e)}")
        else:
            self.stop = False
            st.write("Webcam started.")

    def stop_webcam(self):
        self.stop = True


@st.cache(allow_output_mutation=True)
def get_detector():
    return StreamLiteApp()

detector = get_detector()
detector.run()

