# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 17:22:32 2023

@author: rajku
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 16:43:52 2023

@author: rajku
"""

!pip install OpenCV
import streamlit as st
import numpy as np
import cv2 
from PIL import Image
from ultralytics import YOLO

# Load the model
model = YOLO(r"D:\DS Crs\Live Project\Project 119\final\success\Naga\best.pt")

# Function to display results
class_names = ['32mm']  # List of class names

def display_results(results):
    for result in results:
        orig_img = result.orig_img  # Get the original PIL image

        # Convert PIL image to OpenCV format (BGR)
        img_cv = np.array(orig_img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
       
        # Draw bounding boxes and labels on the image
        for box in result.boxes.xyxy:
            x_min, y_min, x_max, y_max = box.tolist()
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            
            # Get class label based on index in class_names list
            class_index = 0  # Replace with the appropriate index
            class_label = class_names[class_index]
            
            img_cv = cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box
            img_cv = cv2.putText(img_cv, class_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label

        # Convert OpenCV image back to PIL format (RGB)
        img_pil = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_pil)

        # Display the image
        st.image(img_pil, caption='Detected Image', use_column_width=True)


        # Get and display the detected object count
        object_count = len(result.boxes.data)
        st.write("Detected_Count:","label", object_count, class_label)





# Streamlit app
st.title("Steel_Rod_Counting")
upload = st.file_uploader(label="Upload Image:", type=['png', 'jpg', 'jpeg'])

if upload:
    img = Image.open(upload)
    results = model([img])  # Run inference on the uploaded image
    display_results(results)  # Display the results

# ... Rest of the Streamlit app code ...
