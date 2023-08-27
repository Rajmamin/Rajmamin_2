# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 09:29:13 2023

@author: rajku
"""
import streamlit as st
import pandas as pd
import numpy as np


# Set Streamlit app title
st.title("Steel Rod Counting")

# Upload the video file
video_file = st.file_uploader("Upload a video", type=["mp4"])

# Check if a video file is uploaded
if video_file is not None:
    # Display the video
    st.video(video_file)
    st.success("Video uploaded successfully!")
else:
    st.warning("Please upload a video file.")
    
import os
HOME = os.getcwd()
print(HOME)

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image

Image(filename='D:\\DS Crs\\Live Project\\Project 119\\final\\success\\confusion_matrix.png', width=416)

Image(filename='D:\\DS Crs\\Live Project\\Project 119\\final\\success\\results.png', width=416)

Image(filename='D:\\DS Crs\\Live Project\\Project 119\\final\\success\\val_batch0_pred.jpg', width=416)

Image(filename='D:\\DS Crs\\Live Project\\Project 119\\final\\success\\val_batch1_pred.jpg', width=416)

Image(filename='D:\\DS Crs\\Live Project\\Project 119\\final\\success\\val_batch2_pred.jpg', width=416)

!yolo task= detect mode= predict model= "D:\DS Crs\Live Project\Project 119\final\success\best.pt" conf=0.25 source= video_file
import glob
from IPython.display import Image, display

for image_path in glob.glob('runs/detect/predict/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")
