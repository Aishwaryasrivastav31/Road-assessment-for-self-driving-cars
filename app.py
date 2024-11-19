import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO('best.pt')  # Ensure 'best.pt' is in the same directory or provide the full path
    return model

model = load_model()

# Main Interface Header
st.markdown("<h1 style='text-align:center; color:purple;'>Interactive YOLO Model Visualization</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar performance metrics in a single line
st.sidebar.markdown("""
<div style="display: flex; align-items: center; justify-content: space-around; padding: 10px;">
    <div style="text-align: center;">
        <div style="font-size: 20px; color: blue;">🔬</div>
        <p style="margin: 0; color: blue;">Precision</p>
        <div style="font-size: 16px; color: blue;"><strong>0.85</strong></div>
    </div>
    <div style="text-align: center;">
        <div style="font-size: 20px; color: green;">🧪</div>
        <p style="margin: 0; color: green;">Recall</p>
        <div style="font-size: 16px; color: green;"><strong>0.83</strong></div>
    </div>
    <div style="text-align: center;">
        <div style="font-size: 20px; color: purple;">📊</div>
        <p style="margin: 0; color: purple;">mAP</p>
        <div style="font-size: 16px; color: purple;"><strong>0.82</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Option to visualize model performance graph
if st.sidebar.button("Show Model Performance Graph"):
    st.sidebar.markdown("<h4 style='color:purple;'>Model Performance Visualization</h4>", unsafe_allow_html=True)

    # Sample data for visualization (replace with actual data if available)
    metrics = ['Precision', 'Recall', 'mAP']
    values = [0.85, 0.83, 0.82]

    fig, ax = plt.subplots()
    ax.bar(metrics, values, color=['blue', 'green', 'purple'])
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Metrics")
    ax.set_ylabel("Score")
    st.sidebar.pyplot(fig)

# Interactive file uploader
st.markdown("## Upload Image or Video")
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg', 'mp4', 'mov'], accept_multiple_files=False)
st.write("Supported files: .png, .jpg, .jpeg for images, .mp4, .mov for videos")

# Display uploaded file and options for annotation
if uploaded_file is not None:
    col1, col2 = st.columns([2, 1])

    # Process image if an image file is uploaded
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)

        with col1:
            st.markdown("<h3 style='text-align:center; color:green;'>Uploaded Image</h3>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)

            # Run YOLO model to get annotations
            results = model(image)
            annotated_image = results[0].plot()  # Generate the annotated image

            # Display annotated image
            st.markdown("<h3 style='text-align:center; color:green;'>Annotated Image</h3>", unsafe_allow_html=True)
            st.image(annotated_image, use_column_width=True)

    # Process video if a video file is uploaded
    else:
        video_bytes = uploaded_file.read()

        with col1:
            st.markdown("<h3 style='text-align:center; color:green;'>Uploaded Video</h3>", unsafe_allow_html=True)
            st.video(video_bytes)

    # Options section
    with col2:
        st.markdown("<h3 style='color:blue;'>Options</h3>", unsafe_allow_html=True)
        if st.button("Visualize Model"):
            st.success("Model visualized on the uploaded content", icon="✅")
else:
    st.warning("Please upload an image or video to continue.")
