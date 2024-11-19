import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import cv2
import random

# Load YOLO model
model = YOLO('best.pt')  # Update with the correct path to your model

# Custom CSS for styling
st.markdown("""
    <style>
        .header {
            font-size: 1.5em; 
            font-weight: bold; 
            text-align: center; 
            color: #4a00e0; 
            margin-bottom: 10px;
        }
        .metrics-container {
            display: flex; 
            justify-content: space-between; 
            padding: 15px; 
            border: 1px solid #ddd; 
            border-radius: 10px; 
            margin-bottom: 20px;
            background-color: #f7f7f7;
        }
        .metric-bar {
            margin: 5px 0;
        }
        .buttons-container {
            display: flex; 
            justify-content: space-around; 
            margin-bottom: 20px;
        }
        .button {
            flex: 1;
            min-width: 250px;  /* Ensures rectangular buttons */
            margin: 0 5px;
            padding: 10px;
            text-align: center;
            background-color: white;
            border: 1px solid #d3d3d3;
            border-radius: 15px;
            color: #9370DB;
            font-weight: bold;
            cursor: pointer;
            text-decoration: none;
            white-space: nowrap;  /* Prevent text from wrapping */
        }
        .button:hover {
            background-color: #f3f3f3;
        }
        .title-left {
            text-align: left;
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Main Page
st.markdown("<div class='header'>Road assessment for self-driving cars</div>", unsafe_allow_html=True)
st.markdown("**Model Type**: YOLOv8 Object Detection (Fast)<br>", unsafe_allow_html=True)

# Metrics Section with Colored Progress Bars
st.markdown("""
    <div class='metrics-container'>
        <div class='metric'>
            <div><strong>mAP</strong>: 50.8%</div>
            <div class='metric-bar'>
                <div style="background-color: #4CAF50; width: 50.8%; height: 10px; border-radius: 5px;"></div>
            </div>
        </div>
        <div class='metric'>
            <div><strong>Precision</strong>: 63.7%</div>
            <div class='metric-bar'>
                <div style="background-color: #2196F3; width: 63.7%; height: 10px; border-radius: 5px;"></div>
            </div>
        </div>
        <div class='metric'>
            <div><strong>Recall</strong>: 45.7%</div>
            <div class='metric-bar'>
                <div style="background-color: #FF5722; width: 45.7%; height: 10px; border-radius: 5px;"></div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Buttons Section (with tabs)
st.markdown("<br>", unsafe_allow_html=True)  # This adds a one-line space.
tabs = st.tabs(["Detailed Model Evaluation", "Performance By Class", "Visualize Model"])

with tabs[0]:
 # Detailed Model Evaluation Section with Images
    st.subheader("Detailed Model Evaluation")

    # Display the first metric image with caption
    image1_path = "./confusion_matrix.png"  # Replace with the path to your first image
    st.image(image1_path, caption="Our Confusion Matrix shows your model's incorrect predictions broken down by class.", width=300)

    # Add a line break between images (optional)
    st.write("")

    # Display the second metric image with caption
    image2_path = "./vector_analysis.png"  # Replace with the path to your second image
    st.image(image2_path, caption="Vector Analysis visualizes your models images by their similarity and helps identify where your model is struggling.", width=300)


with tabs[1]:
    st.subheader("Performance By Class")
    st.write("Display performance metrics for individual classes. For example:")
    st.write("""
        - Class 1: Precision 80%, Recall 75%
        - Class 2: Precision 90%, Recall 85%
    """)

with tabs[2]:
    st.subheader("Visualize Model")
    image3_path = "./graph.jpg"  # Replace with the path to your second image
    st.image(image3_path, width=900)


# Add a line space between buttons and "Deploy Your Model"
st.markdown("<br>", unsafe_allow_html=True)  # This adds a one-line space.

# Define function for generating a random speed recommendation
def generate_random_speed():
    return random.randint(10, 30)  

# File upload
st.markdown("## Upload Image or Video")
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg', 'mp4', 'mov'], accept_multiple_files=False)
st.write("Supported files: .png, .jpg, .jpeg for images, .mp4, .mov for videos")

if uploaded_file is not None:
    content_container = st.container()
    col1, col2 = content_container.columns([2, 1])

    # Process image if an image file is uploaded
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        image_np = np.array(image)  # Convert the image to a NumPy array

        with col1:
            st.markdown("<h3 style='text-align:center; color:green;'>Uploaded Image</h3>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)

            # Run YOLO model to get annotations
            results = model(image_np)  # Pass the NumPy array instead of PIL image
            annotated_image = results[0].plot()  # Generate the annotated image

            # Display annotated image
            st.markdown("<h3 style='text-align:center; color:green;'>Annotated Image</h3>", unsafe_allow_html=True)
            st.image(annotated_image, use_column_width=True)

            # Generate a random speed recommendation
            random_speed = generate_random_speed()
            st.markdown(f"<h4 style='text-align:center; color:orange;'>Recommended Speed: {random_speed} km/h</h4>", unsafe_allow_html=True)

    # Process video if a video file is uploaded
    else:
        video_bytes = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(video_bytes)
            temp_video_file_path = temp_video_file.name

        st.markdown("<h3 style='text-align:center; color:green;'>Uploaded Video</h3>", unsafe_allow_html=True)
        st.video(video_bytes)

        # Annotate video
        st.markdown("<h4 style='color:purple;'>Annotating Video...</h4>", unsafe_allow_html=True)

        # Open video file with OpenCV
        cap = cv2.VideoCapture(temp_video_file_path)
        if not cap.isOpened():
            st.error("Error: Could not open video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a VideoWriter object to save the annotated video
        annotated_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference on the frame
            results = model(frame)
            annotated_frame = results[0].plot()  # Annotate the frame
            out.write(annotated_frame)  # Write the annotated frame to the output video
            frame_count += 1

        # Release resources
        cap.release()
        out.release()

        # Provide download link for the annotated video
        if frame_count > 0:
            st.markdown("<h3 style='text-align:center; color:green;'>Annotated Video Ready for Download</h3>", unsafe_allow_html=True)
            with open(annotated_video_path, "rb") as f:
                st.download_button(
                    label="Download Annotated Video",
                    data=f,
                    file_name="annotated_video.mp4",
                    mime="video/mp4"
                )

        # Generate a random speed recommendation
        random_speed = generate_random_speed()
        st.markdown(f"<h4 style='text-align:center; color:orange;'>Recommended Speed: {random_speed} km/h</h4>", unsafe_allow_html=True)
