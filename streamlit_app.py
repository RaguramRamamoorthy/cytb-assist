import os
import io
import time
import requests
import streamlit as st
import replicate
from PIL import Image

# -------------------------------------------------
# AUTH
# -------------------------------------------------
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Skin Test Reaction AI Reader",
    layout="centered"
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
# HEADER
# -------------------------------------------------
left, right = st.columns([3, 1])

with left:
    st.markdown(
        """
        <h2>Skin Test <span style="color:#4A6FE3;">Reaction</span> AI Reader</h2>
        <small>AI-assisted screening support</small>
        """,
        unsafe_allow_html=True
    )

with right:
    st.image(
        "serum_institute_of_india_limited_logo.jpg",
        use_container_width=True
    )

    st.markdown(
        """
        <small style="color:#555;">
        This application assists in visual interpretation of the
        <b>Tuberculin Skin Test (Mantoux test)</b>, which is commonly
        used to screen for exposure to
        <i>Mycobacterium tuberculosis</i>.
        <br><br>
        The tool provides an AI-assisted estimate of whether the
        observed skin reaction is <b>likely negative or positive</b>,
        supporting consistent screening and reducing
        inter-observer variability.
        </small>
        """,
        unsafe_allow_html=True
    )
# -------------------------------------------------
# INPUT
# -------------------------------------------------
uploaded = st.file_uploader(
    "Upload skin test image",
    ["jpg", "jpeg", "png"]
)

run = uploaded is not None and st.button("Run analysis")

# -------------------------------------------------
# FIXED LAYOUT ANCHOR (THIS IS THE KEY)
# -------------------------------------------------
image_container = st.container()

with image_container:
    image_slot = st.empty()

# -------------------------------------------------
# SESSION CACHE
# -------------------------------------------------
if "image" not in st.session_state:
    st.session_state.image = None

# -------------------------------------------------
# PREVIEW
# -------------------------------------------------
if uploaded and not run:
    if st.session_state.image is None:
        st.session_state.image = Image.open(uploaded).convert("RGB")

    image_slot.image(
        resize_for_display(st.session_state.image),
        caption="Original image"
    )

# -------------------------------------------------
# RUN PIPELINE
# -------------------------------------------------
if run:
    image = st.session_state.image
    progress = st.progress(0)

    crop_deployment = replicate.deployments.get("serum4321/cropmodel")
    explain_deployment = replicate.deployments.get("serum4321/tbsiglip")

    # STEP 1
    image_slot.image(
        resize_for_display(image),
        caption="Original uploaded image"
    )
    progress.progress(20)
    time.sleep(0.5)

    # STEP 2
    with st.spinner("Detecting reaction region..."):
        crop_pred = crop_deployment.predictions.create(
            input={"image": uploaded}
        )
        crop_pred.wait()

    cropped = download_image(crop_pred.output)

    image_slot.image(
        resize_for_display(cropped),
        caption="Detected reaction region"
    )
    progress.progress(45)
    time.sleep(0.5)

    # STEP 3
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

    image_slot.image(
        resize_for_display(heatmap_img),
        caption="Model attention heatmap"
    )
    progress.progress(70)
    time.sleep(3.0)

    # STEP 4
    image_slot.empty()

    label = explain_out["metrics"]["label"]
    prob = explain_out["metrics"]["probability"]

    color = "#2ECC71" if label == "POSITIVE" else "#E74C3C"

    st.markdown(
        f"""
        <div style="text-align:center; padding-top:30px;">
            <h2 style="color:{color};">{label}</h2>
            <h4>Confidence: {prob:.2%}</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    progress.progress(100)
    st.caption("⚠️ AI-assisted screening support only. Not a diagnostic device.")
