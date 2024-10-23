import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import torch
import cv2
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np
import pandas as pd
from InvoiceGenerator.pdf import SimpleInvoice
from InvoiceGenerator.api import Invoice, Item, Client, Provider, Creator
import random
import io

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
        border-radius: 20px; 
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

st.image("bnr.png", use_column_width=True)  

# Sidebar content with Estimator, About, and Name sections
with st.sidebar:
    st.markdown('<div class="rounded-image">', unsafe_allow_html=True)
    st.image("logo.png", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=['Estimator', 'About', 'Name'],
        icons=['calculator', 'info-circle', 'person'],  
        default_index=0  
    )


# IoU function to calculate the overlap between two boxes
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

# Load models for part and damage detection
def load_model(model_path, threshold, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    return DefaultPredictor(cfg)

# Load the models
predictor_part = load_model("D:/Tuwaiq/Models/DamageEvaluation/DamageEvaluationModel/PartsModel/output_20241012_003029/model_final.pth", 0.5, 21)
predictor_damage = load_model("D:/Tuwaiq/Models/DamageEvaluation/DamageEvaluationModel/DamageModel/output_20241010_002720/model_final.pth", 0.2, 8)

# Part category mapping
category_mapping_parts = {
    "Windshield": 0,
    "Back-windshield": 1,
    "Front-window": 2,
    "Back-window": 3,
    "Front-door": 4,
    "Back-door": 5,
    "Front-wheel": 6,
    "Cracked": 7,
    "Front-bumper": 8,
    "Back-bumper": 9,
    "Headlight": 10,
    "Tail-light": 11,
    "Hood": 12,
    "Trunk": 13,
    "License-plate": 14,
    "Mirror": 15,
    "Roof": 16,
    "Grille": 17,
    "Rocker-panel": 18,
    "Quarter-panel": 19,
    "Fender": 20
}

# Reverse mapping for part categories
id_to_part_name = {v: k for k, v in category_mapping_parts.items()}

# Damage category mapping
damage_mapping = {
    0: 'Dent',
    1: 'Cracked',
    2: 'Scratch',
    3: 'Flaking',
    4: 'Broken part',
    5: 'Paint chip',
    6: 'Missing part',
    7: 'Corrosion'
}

# Load car parts price dataset from CSV
df_prices = pd.read_csv('prices.csv')

# Invoice generation function
# Thanks Riyadh!!!
def invoice_gen(invoice_data: list[dict], invoice_details: dict) -> Invoice:
    """
    invoice_data: list[dict]
        List of invoice items, each containing Brand, Model, Year, Part, Damage, and Price
    invoice_details: dict
        Dictionary containing details about the provider, including bank info and creator name

    Returns:
        Invoice object with client, provider, and creator details.
    """
    os.environ["INVOICE_LANG"] = "en"
    client = Client(invoice_details["client_name"])
    provider = Provider(
        invoice_details["bank_name"],
        bank_account=invoice_details["bank_account"],
        bank_code=invoice_details["bank_code"],
    )
    creator = Creator(invoice_details["creator_name"])
    invoice: Invoice = Invoice(client, provider, creator)
    invoice.currency = "SAR"
    invoice.currency_locale = "ar_SA.UTF-8"
    invoice.use_tax = True
    invoice.title = invoice_details["title"]
    invoice.number = str(random.randint(100000, 999999))  # random invoice number

    for item in invoice_data:
        description = f"Brand: {item['Brand']}, Model: {item['Model']}, Year: {item['Year']}, Part: {item['Part']}"
        invoice.add_item(Item(1, float(item["Price"]), description=description, tax="15"))

    return invoice



# Step 2: Upload an image of your car
if selected == 'Estimator':
    # st.markdown("<h1 class='title'> QADDER </h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload an image to estimate the accident damage</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image of your car:", type=["jpg", "jpeg", "png"])
    brands = df_prices['Brand'].unique().tolist()
    selected_brand = st.selectbox("Select your car brand:", brands)
    filtered_models = df_prices[df_prices['Brand'] == selected_brand]['Model'].unique().tolist()
    selected_model = st.selectbox("Select your car model:", filtered_models)
    filtered_years = df_prices[(df_prices['Brand'] == selected_brand) & (df_prices['Model'] == selected_model)]['Year'].unique().tolist()
    selected_year = st.selectbox("Select your car year:", filtered_years)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        image = np.array(image)

        # Step 3: Run part detection
        st.write("Detecting car parts...")
        output_part = predictor_part(image)
        instances_part = output_part["instances"].to("cpu")
        boxes_part = instances_part.pred_boxes
        labels_part = instances_part.pred_classes

        # Step 4: Run damage detection
        st.write("Detecting damages...")
        output_damage = predictor_damage(image)
        instances_damage = output_damage["instances"].to("cpu")
        boxes_damage = instances_damage.pred_boxes
        labels_damage = instances_damage.pred_classes

        # Step 5: Map damages to car parts
        detected_damages = {}
        fallback_part_name = "Unknown Part"

        for i, box_part in enumerate(boxes_part):
            part_id = labels_part[i].item()
            part_name = id_to_part_name.get(part_id, fallback_part_name)
            part_damages = []
            for j, box_damage in enumerate(boxes_damage):
                iou = compute_iou(box_part.tolist(), box_damage.tolist())
                if iou >= 0.50:  # Adjust IoU threshold as needed
                    damage_name = damage_mapping.get(labels_damage[j].item(), "Unknown Damage")
                    part_damages.append(damage_name)
            detected_damages[part_name] = part_damages

        # Step 6: Display detected damages and allow user confirmation
        st.write("Detected damages:")
        confirmed_damages = {}
        for part, damages in detected_damages.items():
            if damages:
                selected_damages = st.multiselect(f"Confirm damages for {part}:", damages, default=damages)
                confirmed_damages[part] = selected_damages

        # Step 7: Generate report
        if st.button("Generate Report"):
            st.write("### Damage Report")
            total_price = 0
            invoice_data = []
            for part, damages in confirmed_damages.items():
                if damages:
                    part_price = df_prices[(df_prices['Brand'] == selected_brand) & 
                                        (df_prices['Model'] == selected_model) & 
                                        (df_prices['Year'] == selected_year) & 
                                        (df_prices['Part'] == part)]['Price']
                    if not part_price.empty:
                        price = part_price.iloc[0]
                        total_price += price
                        st.write(f"- {part}: {', '.join(damages)} (Price: {price} SAR)")
                        invoice_data.append({
                            "Brand": selected_brand,
                            "Model": selected_model,
                            "Year": selected_year,
                            "Part": part,
                            "Damage": ', '.join(damages),
                            "Price": price
                        })
                    else:
                        st.write(f"- {part}: {', '.join(damages)} (Price: Not available)")
            st.write(f"### Total Estimated Cost: {total_price} SAR")
            st.write("Please verify the detected damages and contact a professional for an accurate assessment.")

            # Generate PDF invoice
            invoice_details = {
                "title": "Car Damage Invoice",
                "client_name": "Client Company",
                "bank_name": "Bank Name",
                "bank_account": "12345678",
                "bank_code": "111",
                "creator_name": "Creator Name",
            }

            invoice = invoice_gen(invoice_data, invoice_details)
            pdf_io = io.BytesIO()  # Create a BytesIO object to store the PDF in memory
            pdf = SimpleInvoice(invoice)
            pdf.gen(pdf_io, generate_qr_code=True)  # Generate PDF into BytesIO object
            pdf_io.seek(0)  # Reset the buffer position to the start

            # Display the PDF as a downloadable file
            st.download_button("Download Invoice", pdf_io, file_name="invoice.pdf", mime="application/pdf")

# About section
if selected == 'About':
    st.write(
        """
        <div style='text-align: center; font-family: "Times New Roman"; font-size: 16px;'>
        Welcome to our QAADER! 
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
st.markdown("<p class='footer'>Thank you for using the QDDER and Drive safely! 🚗</p>", unsafe_allow_html=True)