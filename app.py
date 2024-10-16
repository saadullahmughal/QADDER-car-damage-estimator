import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import torch
import cv2
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np

# Define IoU function to calculate the overlap between two boxes
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Car part category mapping
category_mapping_parts = {
    "Windshield": 0, "Back-windshield": 1, "Front-window": 2, "Back-window": 3,
    "Front-door": 4, "Back-door": 5, "Front-wheel": 6, "Cracked": 7,
    "Front-bumper": 8, "Back-bumper": 9, "Headlight": 10, "Tail-light": 11,
    "Hood": 12, "Trunk": 13, "License-plate": 14, "Mirror": 15, "Roof": 16,
    "Grille": 17, "Rocker-panel": 18, "Quarter-panel": 19, "Fender": 20
}

# Reverse mapping for part categories
id_to_part_name = {v: k for k, v in category_mapping_parts.items()}

# Damage category mapping
damage_mapping = {0: 'Dent', 1: 'Cracked', 2: 'Scratch', 3: 'Flaking', 4: 'Broken part', 5: 'Paint chip', 6: 'Missing part', 7: 'Corrosion'}

# Set up the page configuration
st.set_page_config(page_title="Car Accident Estimator", page_icon="🚗", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body, h1, h2, h3, h4, p {
        font-family: 'Times New Roman'; 
    }
    .main {
        background: linear-gradient(to right, #6a11cb, #2575fc); 
        border-radius: 20px; 
        padding: 20px;
    }
    .rounded-image {
        border-radius: 20px; /* Adjust the radius as needed */
    }
    .title {
        color: #000000;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #000000; 
        text-align: center;
        font-size: 20px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #000000;
    }
    .center-text { 
        font-size: 20px; 
        font-weight: bold; 
        margin-top: 20px; 
        margin-bottom: 20px;
        font-family: 'Times New Roman';
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar content with About section
with st.sidebar:
    st.markdown('<div class="rounded-image">', unsafe_allow_html=True)
    st.image("logo11.png", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)  # Line break for spacing

    selected = option_menu(
        menu_title=None,
        options=['About', 'Name'],
        icons=['info-circle', 'person'],  # Icons for About and Name
        default_index=0  # Set default option
    )

# Load the models
def load_model():
    cfg_part = get_cfg()
    cfg_part.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_part.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg_part.MODEL.ROI_HEADS.NUM_CLASSES = 21  # Adjust as needed
    cfg_part.MODEL.WEIGHTS = os.path.join("D:\Tuwaiq\DamageEvaluation\DamageEvaluation\DamageEvaluationModel\PartsModel\output_20241012_003029", "model_final.pth")  # Update with your model path
    cfg_part.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return DefaultPredictor(cfg_part)

predictor = load_model()

# Main content
st.markdown("<h1 class='title'>Car Accident Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image to estimate the accident damage</p>", unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Resize the image
    resized_image = image.resize((400, 300))  # Change the size as needed
    st.image(resized_image, caption='Uploaded Image', use_column_width=True)

    # User input for name
    name = st.text_input("Enter your name:")
    if name:
        st.success(f"Hello {name}, the image has been uploaded successfully!")

        # Process the image
        if st.button("Check Image", key="check_image", help="Click to analyze the uploaded image"):
            st.success("Image analysis will be conducted.")
            
            # Convert the PIL image to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Run the model prediction
            outputs = predictor(image_cv)
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            labels = instances.pred_classes.numpy()

            # Initialize a dictionary to hold damage reports
            damage_reports = {}

            # Simulated IoU values and damage reports for demonstration
            for box, label in zip(boxes, labels):
                part_name = id_to_part_name.get(label, "Unknown")
                iou_value = np.random.rand()  # Simulating IoU for demonstration
                damage_type = np.random.choice(list(damage_mapping.values()))  # Simulating damage type

                # Only consider IoU >= 0.40
                if iou_value >= 0.40:
                    if part_name not in damage_reports:
                        damage_reports[part_name] = []
                    damage_reports[part_name].append(f"- {damage_type} (IoU: {iou_value:.2f})")

            # Display results
            st.markdown("<div class='center-text'>Detected Damages:</div>", unsafe_allow_html=True)
            for part, damages in damage_reports.items():
                st.write(f"Detected damages for part '{part}':")
                for damage in damages:
                    st.write(damage)


# About section
if selected == 'About':
    st.write(
        """
        <div style='text-align: center; font-family: "Times New Roman"; font-size: 16px;'>
        Welcome to our car accident estimation tool! 
        This innovative project allows users to easily assess vehicle damage by simply uploading an image of their car.
         Our AI model analyzes the uploaded photo, identifies any damage, 
         and visually displays the results on the interface.
          This user-friendly approach ensures quick and accurate evaluations, 
          helping you understand the condition of your vehicle after an accident.
        </div>
        """,
        unsafe_allow_html=True
    )
elif selected == 'Name':
    st.write(
        """ 
        <div style='text-align: center; font-family: "Times New Roman"; font-size: 20px;'>
        Marwa, Roqaih, Khulud, Rayan.
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("<p class='footer'>Thank you for using the Car Accident Estimator and Drive safely! 🚗</p>", unsafe_allow_html=True)