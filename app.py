import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import os
import plotly.express as px
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import tempfile


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Fake Logo Detection", page_icon="🔍", layout="wide")


# -------------------------------
# Theme Toggle (Light/Dark Mode)
# -------------------------------
theme = st.sidebar.title("🌗 Theme")
option= st.sidebar.radio("Select a theme", ["Light", "Dark"])
chart_template = "plotly_dark" if option == "Dark" else "plotly_white"


if option == "Dark":
    st.markdown("""
        <style>
            body, .stApp {
                background-color: #121212;
                color: #f5f5f5;
            }
            .card {
                background: #1e1e1e;
                color: #f5f5f5;
            }
            .banner {
                background: linear-gradient(90deg, #0d47a1, #1565c0);
            }
                
            /* Top bar (toolbar/header) */
            header[data-testid="stHeader"] {
                background-color: black !important;
                color: white !important;
            }

            header[data-testid="stHeader"] * {
                color: white !important;
            }

            /* File uploader label */
            .stFileUploader label {
                color: white !important;
                font-weight: 600;
            }

            /* Browse button */
            .stFileUploader div div div button {
                color: black !important;
                background-color: #f5f5f5 !important;
                border-radius: 8px;
            }

            /* All clickable buttons */
            button[kind="secondary"],
            button[kind="primary"],
            .stButton>button {
                color: black !important;
            }

            /* 🔹 Tabs styling in Dark Mode */
            .stTabs [data-baseweb="tab"] {
                color: black !important;   /* Inactive tabs text */
                background-color: #f5f5f5 !important; /* Light background */
                border-radius: 6px;
                margin-right: 6px;
                padding: 6px 12px;
                font-weight: 600;
            }

            /* Active tab */
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                color: white !important;   /* Active tab text */
                background-color: #1565c0 !important; /* Blue active tab */
            }
                
            /* Metric text color */
            [data-testid="stMetricValue"],
            [data-testid="stMetricLabel"] {
                color: white !important;
            }
            
            /* Sidebar styling - keep consistent */
            [data-testid="stSidebar"] {
                background-color: black !important; 
                color: white !important;              
            }

            [data-testid="stSidebar"] label, 
            [data-testid="stSidebar"] span, 
            [data-testid="stSidebar"] p {
                color: white !important;  
            }
                
            /* Hover effect for download button */
            .stDownloadButton > button:hover,
            .stButton > button:hover {
                background-color: #dcdcdc !important;   /* Light grey hover */
                color: black !important;

            }

        </style>
    """, unsafe_allow_html=True)

else:  # Light theme
    st.markdown("""
        <style>
            body, .stApp {
                background-color: #ffffff;
                color: #000000;
            }
            .card {
                background: #ffffff;
                color: #000000;
            }
            .banner {
                background: linear-gradient(90deg, #1565c0, #1e88e5);
            }
            /* File uploader text */
            .stFileUploader label {
                color: black !important;
                font-weight: 600;
            }
            /* Browse button */
            .stFileUploader div div div button {
                color: white !important;
                background-color: #1565c0 !important; /* Blue button */
                border-radius: 8px;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)


# Custom CSS for sleek banner
st.markdown("""
    <style>
        .banner {
            background: linear-gradient(90deg, #1565c0, #1e88e5);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }
""", unsafe_allow_html=True)




# Apply global spacing for all subheaders
st.markdown("""
    <style>
        subheader {
            margin-top: 40px;   /* Adds space before every st.subheader */
        }
    </style>
""", unsafe_allow_html=True)



# -------------------------------
# Model Loader
# -------------------------------
@st.cache_resource
def load_model_for_inference(ckpt_path="D:\\M.Sc. Data Analytics\\SY\\Semester 4\\DL\\CA2\\models\\best_model.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)

    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, len(classes))
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return model, tfm, classes

# -------------------------------
# Grad-CAM
# -------------------------------
def grad_cam(model, image_tensor, target_layer="layer4"):
    gradients, activations = [], []

    def save_gradients(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def save_activations(module, input, output):
        activations.append(output)

    for name, module in model.named_modules():
        if name == target_layer:
            module.register_forward_hook(save_activations)
            module.register_backward_hook(save_gradients)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() + 1e-6)
    return cam, pred_class

def overlay_cam_on_image(pil_img, cam):
    img = np.array(pil_img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap) / 255 + np.float32(img) / 255
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)



def generate_pdf_report(logo_img, pred_label, probs, classes, gradcam_img):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_file.name, pagesize=letter)
    width, height = letter

    # --- Title Banner ---
    c.setFillColor(colors.HexColor("#1f77b4"))
    c.rect(0, height - 80, width, 80, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(180, height - 50, "Fake Logo Detection Report")

    # --- Prediction Box ---
    if pred_label.lower() == "genuine":
        pred_color = colors.green
    else:
        pred_color = colors.red

    c.setFillColor(pred_color)
    c.rect(50, height - 150, width - 100, 50, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(60, height - 135, f"Prediction: {pred_label}")
    confidence = float(np.max(probs)) * 100
    c.drawString(300, height - 135, f"Confidence: {confidence:.2f}%")

    # --- Uploaded Logo ---
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 200, "Uploaded Logo:")
    logo_reader = ImageReader(logo_img)
    c.drawImage(logo_reader, 50, height - 400, width=150, height=150)

    # --- Probabilities Table ---
    data = [["Class", "Probability (%)"]] + [[cls, f"{prob*100:.2f}"] for cls, prob in zip(classes, probs)]
    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 250, height - 400)

    # --- Grad-CAM ---
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawString(50, height - 430, "Grad-CAM Heatmap:")
    gradcam_reader = ImageReader(gradcam_img)
    c.drawImage(gradcam_reader, 50, height - 650, width=200, height=200)

    c.save()
    return tmp_file.name



# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Inference", "Training Dashboard"])



# -------------------------------
# Page 0: Welcome / Splash Page
# -------------------------------

if page == "Welcome":
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 60px;
            border-radius: 12px;
            background: linear-gradient(135deg, #1565c0, #1e88e5);
            color: white;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            animation: fadeIn 2s ease-in;
        ">
            <h1 style="font-size: 48px; font-weight: bold;">🔍 Fake Logo Detection</h1>
            <p style="font-size: 20px; margin-top: 10px;">
                Welcome to the Fake Logo Detection App.<br>
                Upload a logo, run inference and generate reports.<br>
                Switch to <b>Inference</b> or <b>Training Dashboard</b> from the sidebar to get started 🚀
            </p>
        </div>

        <style>
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        </style>
        """,
        unsafe_allow_html=True
    )



# -------------------------------
# Page 1: Inference
# -------------------------------
if page == "Inference":
    st.markdown('<div class="banner"><h2>Fake Logo Detection</h2></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload a logo image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")

        # Load model
        model, tfm, classes = load_model_for_inference("D:\\M.Sc. Data Analytics\\SY\\Semester 4\\DL\\CA2\\models\\best_model.pt")

        # Preprocess
        img_tensor = tfm(pil_img).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(model(img_tensor), dim=1)[0].numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = classes[pred_idx]

        # Tabs for neat layout
        tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Probabilities", "🔥 Grad-CAM"])


        with tab1:
            if pred_label.lower() == "genuine":
                bg_color = "#90EE90"   # light green
                st.markdown(
                    f"""
                    <div style="
                        background-color: {bg_color};
                        color: black;
                        padding: 15px;
                        border-radius: 8px;
                        font-weight: bold;
                        font-size: 18px;
                        text-align: center;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.4);
                    ">
                        ✅ Prediction: Genuine Logo
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # 😃 Bouncing smiley animation
                st.markdown(
                    """
                    <div style="text-align:center; font-size:80px; color:green; animation: bounce 1s infinite;">
                        😊
                    </div>
                    <style>
                    @keyframes bounce {
                        0%, 100% { transform: translateY(0); }
                        50% { transform: translateY(-15px); }
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

            else:
                bg_color = "#FF7F7F"   # light red
                st.markdown(
                    f"""
                    <div style="
                        background-color: {bg_color};
                        color: black;
                        padding: 15px;
                        border-radius: 8px;
                        font-weight: bold;
                        font-size: 18px;
                        text-align: center;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.4);
                    ">
                        ❌ Prediction: Fake Logo
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # 😞 Shaking sad face animation
                st.markdown(
                    """
                    <div style="text-align:center; font-size:80px; color:red; animation: shake 0.8s infinite;">
                        😞
                    </div>
                    <style>
                    @keyframes shake {
                        0% { transform: translateX(0); }
                        25% { transform: translateX(-8px); }
                        50% { transform: translateX(8px); }
                        75% { transform: translateX(-8px); }
                        100% { transform: translateX(0); }
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )


        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            #st.markdown("<p style='color:white;'>Select Probability Visualization:</p>", unsafe_allow_html=True)


            # Dataframe of probabilities
            df_probs = pd.DataFrame({"Class": classes, "Probability": probs})

            # Toggle option
            chart_type = st.radio(
                "Select Probability Visualization:",
                ["Bar Chart", "Circular Gauge"],
                horizontal=True,
            )

            if option == "Dark":
                st.markdown("""
                    <style>
                    div[data-testid="stRadio"] label, 
                    div[data-testid="stRadio"] span, 
                    div[data-testid="stRadio"] p, 
                    div[data-testid="stRadio"] div {
                        color: white !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
               

            if chart_type == "Bar Chart":
                # --- Horizontal bar chart ---
                #fig_bar = px.bar(
                #df_probs, x="Probability", y="Class", orientation="h",
                #color="Probability", text="Probability",
                #color_continuous_scale="Blues",
                #)
                #st.plotly_chart(fig_bar, use_container_width=True)


                fig_bar = px.bar(
                    df_probs, x="Probability", y="Class", orientation="h",
                    color="Probability", text="Probability",
                    color_continuous_scale="Blues" if option == "Light" else "Viridis",
                    template=chart_template
                )

                # Force background + text color to match theme
                fig_bar.update_layout(
                    plot_bgcolor="black" if option == "Dark" else "white",
                    paper_bgcolor="black" if option == "Dark" else "white",
                    font=dict(color="white" if option == "Dark" else "black")
                )

                st.plotly_chart(fig_bar, use_container_width=True)


            
            else:
                # --- Confidence Gauge (circular) ---
                import plotly.graph_objects as go
                confidence = float(np.max(probs)) * 100

                pred_class = classes[pred_idx]

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence,
                    delta={'reference': 50},  # difference from threshold
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "limegreen" if pred_class.lower() == "genuine" else "red"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ffcccc"},
                            {'range': [50, 80], 'color': "#fff3cd"},
                            {'range': [80, 100], 'color': "#d4edda"}
                        ],
                        'threshold': {
                            'line': {'color': "blue", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    },
                    number={'suffix': "%"},
                    title={'text': f"Confidence for {pred_class}"}
                ))

                fig_gauge.update_layout(
                    template="plotly_dark" if option == "Dark" else "plotly_white",
                    paper_bgcolor="black" if option == "Dark" else "white",
                    font=dict(color="white" if option == "Dark" else "black")
                )

                st.plotly_chart(fig_gauge, use_container_width=True)

            # --- Download probabilities as CSV ---
            csv = df_probs.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Probabilities CSV", csv, "probabilities.csv", "text/csv")

            st.markdown('</div>', unsafe_allow_html=True)


        # -------------------------------
        # Class Distribution Overview
        # -------------------------------
        st.subheader("📊 Class Distribution Overview")

        if "session_preds" not in st.session_state:
            st.session_state["session_preds"] = []

        # Save current prediction
        st.session_state["session_preds"].append(pred_label)

        # Convert to DataFrame
        dist_df = pd.DataFrame(st.session_state["session_preds"], columns=["Prediction"])
        counts = dist_df["Prediction"].value_counts().reset_index()
        counts.columns = ["Class", "Count"]

        # Create donut chart
        fig_donut = px.pie(
            counts, values="Count", names="Class",
            hole=0.5, color="Class",
            color_discrete_map={"Genuine": "green", "Fake": "red"},
            title="Distribution of Predictions in This Session"
        )

        fig_donut.update_traces(
            textfont=dict(color="white" if option == "Dark" else "black")
        )

        fig_donut.update_layout(
            template=chart_template,
            font=dict(color="white" if option == "Dark" else "black"),
            paper_bgcolor="black" if option == "Dark" else "white",
            plot_bgcolor="black" if option == "Dark" else "white",
            title_font=dict(color="white" if option == "Dark" else "black"),
            legend_font=dict(color="white" if option == "Dark" else "black"),
        )

        st.plotly_chart(fig_donut, use_container_width=True)



        # Save history of predictions (image + label)
        if "history" not in st.session_state:
            st.session_state["history"] = []

        st.session_state["history"].append({
            "image": pil_img,
            "prediction": pred_label
        })

        # Keep only last N (e.g., 5)
        N = 5
        st.session_state["history"] = st.session_state["history"][-N:]

        # -------------------------------
        # Prediction History Timeline
        # -------------------------------
        
        st.subheader("🕒 Prediction History Timeline")
        if st.session_state["history"]:
            for idx, entry in enumerate(reversed(st.session_state["history"]), 1):
                col1, col2 = st.columns([1,3])
                with col1:
                    st.image(entry["image"], caption=f"Logo {idx}", width=100)
                with col2:
                    if entry["prediction"].lower() == "genuine":
                        st.markdown(f"<p style='color:lime; font-weight:bold;'>✅ Genuine</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='color:red; font-weight:bold;'>❌ Fake</p>", unsafe_allow_html=True)
        else:
            st.info("No history yet. Upload some logos to see the timeline.")




        with tab3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            cam, pred_class = grad_cam(model, img_tensor)
            overlay = overlay_cam_on_image(pil_img, cam)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(pil_img, use_column_width=True)
            with col2:
                st.subheader("Grad-CAM Heatmap")
                st.image(overlay, use_column_width=True)

            # Download Grad-CAM
            buf = BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")
            st.download_button("💾 Download Grad-CAM Heatmap", buf.getvalue(),
                               "gradcam.png", "image/png")
            

            # --- PDF Report Download Button ---
            if st.button("📑 Generate PDF Report"):
                gradcam_buf = BytesIO()
                Image.fromarray(overlay).save(gradcam_buf, format="PNG")

                pdf_path = generate_pdf_report(
                    pil_img, pred_label, probs, classes, gradcam_buf
                )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📥 Download Report",
                        f,
                        file_name="logo_report.pdf",
                        mime="application/pdf"
                    )

            st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Page 2: Training Dashboard
# -------------------------------
elif page == "Training Dashboard":
    st.markdown('<div class="banner"><h2>Training Dashboard</h2></div>', unsafe_allow_html=True)

    log_path = "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\DL\\CA2\\outputs\\training_log.csv"
    roc_path = "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\DL\\CA2\\outputs\\curves\\roc.png"
    pr_path = "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\DL\\CA2\\outputs\\curves\\pr.png"
    reliability_path = "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\DL\\CA2\\outputs\\curves\\reliability.png"
    cm_path = "D:\\M.Sc. Data Analytics\\SY\\Semester 4\\DL\\CA2\\outputs\\confusion_matrix.png"

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)

        # Summary KPIs
        best_acc = df["val_acc"].max()
        final_acc = df["val_acc"].iloc[-1]
        final_loss = df["val_loss"].iloc[-1]

        ##Updated col1,col2,col3
        col1, col2, col3 = st.columns(3)
        
        # Best Accuracy (blue neon glow)
        col1.markdown(
            f"""
            <div style="
                background-color: #111111;
                padding: 15px;
                border-radius: 12px;
                text-align: center;
                color: white;
                font-size: 20px;
                font-weight: bold;
                box-shadow: 0 0 15px #00BFFF, 0 0 30px #00BFFF;
            ">
                🏆 Best Accuracy <br> {best_acc:.3f}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Final Accuracy (green neon glow)
        col2.markdown(
            f"""
            <div style="
                background-color: #111111;
                padding: 15px;
                border-radius: 12px;
                text-align: center;
                color: white;
                font-size: 20px;
                font-weight: bold;
                box-shadow: 0 0 15px #39FF14, 0 0 30px #39FF14;
            ">
                ✅ Final Accuracy <br> {final_acc:.3f}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Final Loss (red neon glow)
        col3.markdown(
            f"""
            <div style="
                background-color: #111111;
                padding: 15px;
                border-radius: 12px;
                text-align: center;
                color: white;
                font-size: 20px;
                font-weight: bold;
                box-shadow: 0 0 15px #FF073A, 0 0 30px #FF073A;
            ">
                📉 Final Loss <br> {final_loss:.3f}
            </div>
            """,
            unsafe_allow_html=True
        )

        
        
        # Interactive accuracy/loss plot
        st.subheader("📊 Training & Validation Metrics")

        def apply_plotly_theme(fig, option):
            """Apply dark/light theme to Plotly figures with custom styling."""
            if option == "Dark":
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="black",
                    plot_bgcolor="black",
                    font=dict(color="white"),  
                    title=dict(font=dict(color="white")),  
                    legend=dict(font=dict(color="white")),
                    xaxis=dict(showline=True, linewidth=2, linecolor="white"),
                    yaxis=dict(showline=True, linewidth=2, linecolor="white"),
                )
            else:
                fig.update_layout(
                    template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="black"),
                title=dict(font=dict(color="black")),
                #margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(showline=True, linewidth=2, linecolor="black"),
                yaxis=dict(showline=True, linewidth=2, linecolor="black"),
                )
            return fig

        fig_acc = px.line(df, x="epoch", y=["train_acc", "val_acc"],
                          labels={"value": "Accuracy", "epoch": "Epoch"},
                          title="Accuracy over Epochs")
        fig_acc = apply_plotly_theme(fig_acc, option)
        st.plotly_chart(fig_acc, use_container_width=True)

        fig_loss = px.line(df, x="epoch", y=["train_loss", "val_loss"],
                           labels={"value": "Loss", "epoch": "Epoch"},
                           title="Loss over Epochs")
        fig_loss = apply_plotly_theme(fig_loss, option)
        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.warning("⚠️ Training log not found. Please train the model first.")

    
    
    # Static plots (ROC, PR, Reliability, Confusion Matrix)
    import base64
    def styled_image(path, caption, option):
        border_color = "white" if option == "Dark" else "black"
        if os.path.exists(path):
            st.markdown(
                f"""
                <div style="border:2px solid {border_color}; border-radius:10px; padding:5px; text-align:center;">
                    <img src="data:image/png;base64,{base64.b64encode(open(path,"rb").read()).decode()}" 
                        style="width:100%; border-radius:10px;">
                    <p style="color:{border_color}; font-weight:bold;">{caption}</p>
                </div>
                """, unsafe_allow_html=True
            )


    # Inject tooltip CSS globally
    st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        margin-left: 5px;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%; 
        left: 0;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        white-space: normal;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

    def tooltip(label, text):
        return f"""
        <div class="tooltip">ℹ️
            <span class="tooltiptext"><b>{label}</b>: {text}</span>
        </div>
        """


    st.subheader("📉 Model Performance Curves")
    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(roc_path):
            styled_image(roc_path, "ROC Curve", option)
            st.markdown(tooltip("ROC Curve", "Shows the trade-off between True Positive Rate and False Positive Rate."), unsafe_allow_html=True)

        if os.path.exists(pr_path):
            styled_image(pr_path, "Precision-Recall Curve", option)
            st.markdown(tooltip("Precision-Recall Curve", "Displays the balance between Precision and Recall, useful for imbalanced datasets."), unsafe_allow_html=True)


    with col2:
        if os.path.exists(reliability_path):
            styled_image(reliability_path, "Reliability Curve", option)
            st.markdown(tooltip("Reliability Curve", "Compares predicted probabilities to actual outcomes for calibration assessment."), unsafe_allow_html=True)

        if os.path.exists(cm_path):
            styled_image(cm_path, "Confusion Matrix", option)
            st.markdown(tooltip("Confusion Matrix", "Summarizes correct vs incorrect classifications in a 2x2 grid."), unsafe_allow_html=True)