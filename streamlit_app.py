import os
import io
import time
import requests
import streamlit as st
import replicate
from PIL import Image
import os
print(os.getenv("REPLICATE_API_TOKEN"))

# -------------------------------------------------
# AUTH
# -------------------------------------------------
# os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Skin Test Reaction AI Reader",
    layout="centered"
)

# -------------------------------------------------
# SIDEBAR (LEFT TAB)
# -------------------------------------------------
with st.sidebar:
    st.markdown("<div style='text-align:center; padding:12px 0;'>", unsafe_allow_html=True)
    st.image("serum_institute_of_india_limited_logo.jpg", width=160)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        **Tuberculin Skin Test (Mantoux Test)**

        This application assists in the visual interpretation of the
        tuberculin skin test, commonly used to screen for exposure to
        *Mycobacterium tuberculosis*.

        The tool provides an **AI-assisted estimate** of whether the
        observed skin reaction is likely **negative or positive**,
        supporting consistent screening.

        ---
        ⚠️ *For screening support only.  
        Not a diagnostic device.*
        """
    )

DISPLAY_HEIGHT = 350

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def resize_for_display(img, max_height=DISPLAY_HEIGHT):
    w, h = img.size
    scale = max_height / h
    return img.resize((int(w * scale), max_height), Image.BICUBIC)

def download_image(url):
    r = requests.get(url)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

# -------------------------------------------------
# INPUT
# -------------------------------------------------
uploaded = st.file_uploader(
    "Upload test image",
    ["jpg", "jpeg", "png"]
)

run = uploaded is not None and st.button("Run analysis")

# -------------------------------------------------
# SINGLE CANVAS (UI KEY)
# -------------------------------------------------
canvas = st.empty()

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "image" not in st.session_state:
    st.session_state.image = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# -------------------------------------------------
# PREVIEW (NO LAG)
# -------------------------------------------------
if uploaded:
    if uploaded.name != st.session_state.last_uploaded_name:
        st.session_state.image = Image.open(uploaded).convert("RGB")
        st.session_state.last_uploaded_name = uploaded.name

    if not run:
        canvas.image(
            resize_for_display(st.session_state.image),
            caption="Original image"
        )

# -------------------------------------------------
# RUN PIPELINE (SEQUENTIAL UI)
# -------------------------------------------------
if run:
    image = st.session_state.image


    crop_deployment = replicate.deployments.get("serum4321/cropmodel")
    explain_deployment = replicate.deployments.get("serum4321/tbsiglip")

    # ---- STATE 1: ORIGINAL ----
    canvas.image(
        resize_for_display(image),
        caption="Original uploaded image"
    )
    progress = st.progress(0)
    progress.progress(15)

    # ---- STATE 2: CROP ----
    with st.spinner("Detecting reaction region..."):
        crop_pred = crop_deployment.predictions.create(
            input={"image": uploaded}
        )
        crop_pred.wait()

    cropped = download_image(crop_pred.output)

    canvas.image(
        resize_for_display(cropped),
        caption="Detected reaction region"
    )
    progress.progress(45)

    # ---- STATE 3: HEATMAP ----
    with st.spinner("Analyzing reaction pattern..."):
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        buf.seek(0)

        explain_pred = explain_deployment.predictions.create(
            input={"image": buf}
        )
        explain_pred.wait()

    explain_out = explain_pred.output
    heatmap_img = download_image(explain_out["heatmap"])

    canvas.image(
        resize_for_display(heatmap_img),
        caption="Model attention heatmap"
    )
    progress.progress(75)

    time.sleep(1.5)

    # -------------------------------------------------
    # FINAL RESULT LOGIC
    # -------------------------------------------------
    metrics = explain_out.get("metrics", {})

    prob = float(metrics.get("probability", 0))


    # --- TRIAGE DECISION ---
    if prob >= 0.60:
        label = "LIKELY Tb POSITIVE"
        advice = "Visible induration detected"
        color = "#2ECC71"

    elif prob >= 0.40:
        label = "MANUAL CHECK REQUIRED"
        advice = "Visual features unclear — palpation recommended"
        color = "#F39C12"

    else:
        label = "NEGATIVE"
        advice = "No visible induration detected"
        color = "#E74C3C"

    # ---- STATE 4: RESULT (SAME CANVAS) ----
    canvas.empty()

    # label = explain_out["metrics"]["label"]
    # prob = explain_out["metrics"]["probability"]
    # color = "#2ECC71" if label == "POSITIVE" else "#E74C3C"

    canvas.markdown(
        f"""
        <div style="
            background:#0f172a;
            border-radius:12px;
            padding:28px;
            text-align:center;
            border:1px solid #1e293b;
        ">
            <h2 style="color:{color}; margin-bottom:10px;">{label}</h2>
            <p style="color:#cbd5e1; font-size:16px;">{advice}</p>
            <p style="color:#64748b; font-size:13px; margin-top:10px;">
            AI screening support — clinical confirmation required
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    progress.progress(100)
