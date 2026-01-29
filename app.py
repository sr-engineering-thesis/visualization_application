import streamlit as st
import numpy as np
import torch
import cv2
import time
import hashlib

from models.fsrcnn import FSRCNN_BIG_PRETRAIN
from models.ninasr_b0 import NinaSR
from models.ninasr_small import NinaSR_small
from models.carn import CARN
from models.interpolations import upscale_image
from utils import *

INTERPOLATION_METHODS = {
    "Bilinear Interpolation": cv2.INTER_LINEAR,
    "Bicubic Interpolation": cv2.INTER_CUBIC,
    "Lanczos4 Interpolation": cv2.INTER_LANCZOS4
}

models = {
    "Bilinear Interpolation": None,
    "Bicubic Interpolation": None,
    "Lanczos4 Interpolation": None,
    "FSRCNN": FSRCNN_BIG_PRETRAIN,
    "NinaSR-B0": NinaSR,
    "NinaSR-small": NinaSR_small
    # "CARN": CARN
}

weights = {
    "FSRCNN": "./weights/fsrcnn_finetuned_0.0036803.pth",
    "NinaSR-B0": "./weights/ninasr_finetuned.pth",
    "NinaSR-small": "./weights/nina_sr_julka_best.pth",
    # "CARN": "./weights/carn_finetuned.pth"
}

def run_model_inference(lr_img, selected_model_name):

    if selected_model_name in INTERPOLATION_METHODS:
        interp_method = INTERPOLATION_METHODS[selected_model_name]
        start_ns = time.monotonic_ns()
        sr_img = upscale_image(lr_img, scale_factor=2, interpolation=interp_method)
        end_ns = time.monotonic_ns()
        elapsed_ms = (end_ns - start_ns) / 1_000_000
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models[selected_model_name](scale=2).to(device)
        model.load_state_dict(torch.load(weights[selected_model_name], map_location=device))
        model.eval()

        lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

        start_ns = time.monotonic_ns()
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        end_ns = time.monotonic_ns()

        elapsed_ms = (end_ns - start_ns) / 1_000_000
        sr_img = postprocess(sr_tensor)

    return sr_img, elapsed_ms, selected_model_name


def display_single_model(hr_img, lr_resized, sr_img, zoom, size):
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="
            text-align: center;
            font-size: 1.0rem;
            color: gray;
            letter-spacing: 0.02em;
        ">
        <b>Left:</b> High-Resolution (HR) ground truth &nbsp;|&nbsp;
        <b>Middle:</b> Low-Resolution (LR) model input &nbsp;|&nbsp;
        <b>Right:</b> Super-Resolution (SR) model output
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    gt_bytes = cv2.imencode('.png', hr_img)[1].tobytes()
    lr_bytes = cv2.imencode('.png', lr_resized)[1].tobytes()
    sr_bytes = cv2.imencode('.png', sr_img)[1].tobytes()

    add_synced_magnifiers(gt_bytes, lr_bytes, sr_bytes, zoom=zoom, size=size)
    

def display_model_comparison(hr_img, lr_resized, sr_images_dict, zoom, size):
    gt_bytes = cv2.imencode('.png', hr_img)[1].tobytes()
    lr_bytes = cv2.imencode('.png', lr_resized)[1].tobytes()
    sr_bytes_list = []
    model_names = []
    for model_name, sr_img in sr_images_dict.items():
        sr_bytes = cv2.imencode('.png', sr_img)[1].tobytes()
        sr_bytes_list.append(sr_bytes)
        model_names.append(model_name)

    add_multi_model_magnifiers(gt_bytes, lr_bytes, sr_bytes_list, model_names, zoom=zoom, size=size)


st.title("Super-Resolution Models Demo")
st.set_page_config(layout="wide")

# ========================= SIDEBAR ====================================
mode = st.sidebar.radio("Mode", ["Single Model", "Model Comparison"])
if mode == "Single Model":
    selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
elif mode == "Model Comparison":
    model_compare = st.sidebar.multiselect("Compare Models", list(models.keys()), default=[list(models.keys())[0]])
uploaded = st.sidebar.file_uploader("Upload a High-Resolution image", type=["png", "jpg", "jpeg"])
zoom = st.sidebar.slider("Magnifiers - Zoom", min_value=1, max_value=5, value=1)
size = st.sidebar.slider("Magnifiers - Size", min_value=50, max_value=350, value=150)

# ========================= MAIN AREA ====================================
if "uploaded_file_hash" not in st.session_state:
    st.session_state.uploaded_file_hash = None

if "hr_img" not in st.session_state:
    st.session_state.hr_img = None
    st.session_state.lr_img = None
    st.session_state.lr_resized = None
    st.session_state.sr_images = {}
    st.session_state.selected_model = None
    st.session_state.model_compare = []
    st.session_state.inference_done = False
    st.session_state.metrics_df = {}

if uploaded:
    uploaded_bytes = uploaded.read()
    uploaded_hash = hashlib.md5(uploaded_bytes).hexdigest()
    uploaded.seek(0) 

    file_bytes = np.frombuffer(uploaded_bytes, np.uint8)
    hr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    lr_img = cv2.resize(
        hr_img, 
        (hr_img.shape[1] // 2, hr_img.shape[0] // 2),
        interpolation=cv2.INTER_CUBIC
    )
    lr_resized = cv2.resize(
        lr_img, 
        (hr_img.shape[1], hr_img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    st.session_state.hr_img = hr_img
    st.session_state.lr_img = lr_img
    st.session_state.lr_resized = lr_resized
    
    if mode == "Single Model":
        need_inference = (
            st.session_state.selected_model != selected_model_name
            or st.session_state.uploaded_file_hash != uploaded_hash
            or not st.session_state.inference_done
        )
        if need_inference:  
            sr_img, elapsed_ms, _ = run_model_inference(st.session_state.lr_img, selected_model_name)
            st.session_state.metrics_df[selected_model_name] = calculate_metrics(
                hr_img, sr_img, inference_time=elapsed_ms, selected_model_name=selected_model_name
            )

            st.session_state.sr_images = {selected_model_name: sr_img}
            st.session_state.selected_model = selected_model_name
            st.session_state.inference_done = True
        else:
            sr_img = st.session_state.sr_images[selected_model_name]

    elif mode == "Model Comparison":
        new_models = [m for m in model_compare if m not in st.session_state.sr_images 
                    or st.session_state.uploaded_file_hash != uploaded_hash]

        metrics_list = []

        for model_name in new_models:
            sr_img, elapsed_ms, _ = run_model_inference(lr_img, model_name)
            st.session_state.sr_images[model_name] = sr_img
            st.session_state.metrics_df[model_name] = calculate_metrics(
                hr_img, sr_img, inference_time=elapsed_ms, selected_model_name=model_name
            )

        for model_name in model_compare:
            if model_name in st.session_state.metrics_df:
                metrics_list.append(st.session_state.metrics_df[model_name])

        if metrics_list:
            metrics_table = pd.concat(metrics_list)
        else:
            metrics_table = pd.DataFrame()


        for model_name in list(st.session_state.sr_images.keys()):
            if model_name not in model_compare:
                st.session_state.sr_images.pop(model_name)

        st.session_state.model_compare = model_compare
        st.session_state.uploaded_file_hash = uploaded_hash
        st.session_state.inference_done = True

    # ========================= DISPLAY RESULTS ====================================
    if st.session_state.inference_done:
        
        # =================== VISUAL COMPARISON ===================
        st.subheader("Visual Comparison")

        st.markdown(
            """
            <div style="
                text-align: justify;
                font-size: 1.0rem;
                color: gray;
                line-height: 1.6;
                max-width: 900px;
                margin: 0 auto;
            ">
            The high-resolution (HR) image is first downscaled to generate the low-resolution (LR) input. 
            This LR image is then upscaled using the selected super-resolution model or interpolation method 
            to produce the super-resolution (SR) output. To facilitate visual analysis, 
            interactive magnifiers are provided. Hover over any image to inspect corresponding magnified regions; 
            the zoomed views are synchronized across all images for precise, side-by-side comparison. 
            The magnifier zoom level and window size can be adjusted using the controls in the sidebar.
            </div>
            """,
            unsafe_allow_html=True
        )

        if mode == "Single Model":
            sr_img = st.session_state.sr_images[selected_model_name]
            lr_resized = st.session_state.lr_resized
            hr_img = st.session_state.hr_img
            display_single_model(hr_img, lr_resized, sr_img, zoom, size)

        elif mode == "Model Comparison":
            sr_images_dict = st.session_state.sr_images
            lr_resized = st.session_state.lr_resized
            hr_img = st.session_state.hr_img
            display_model_comparison(hr_img, lr_resized, sr_images_dict, zoom, size)
        

        # =================== ERROR ANALYSIS ===================
        st.subheader("Error Analysis")

        if mode == "Single Model":
            models_to_plot = [selected_model_name]
        else:
            models_to_plot = st.session_state.model_compare

        for model_name in models_to_plot:
            sr_img = st.session_state.sr_images[model_name]

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**{model_name} - Heatmap**")
                diff_map = compute_global_diff_maps(hr_img, [sr_img])[0]
                diff_bytes = cv2.imencode('.png', diff_map)[1].tobytes()
                st.image(diff_bytes, width="stretch")

            with col2:
                st.write(f"**{model_name} - Histogram**")
                fig, stats = plot_absolute_pixel_error_histogram(hr_img, sr_img)
                st.pyplot(fig)
                st.markdown(
                    f"""
                    <div style="text-align: center; font-size: 0.85rem; color: gray; margin-top: 5px;">
                    <b>Per-pixel absolute error statistics (|HR − SR|)</b><br>
                    Min: {stats['min']}, Max: {stats['max']} &nbsp;|&nbsp;
                    Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}, Std: {stats['std']:.3f}<br>
                    P95: {stats['p95']:.2f}, P99: {stats['p99']:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(""" 
                        <div style=" text-align: justify; font-size: 0.95rem; color: gray; line-height: 1.5; max-width: 400px; margin: 0 auto; ">
                         The heatmap shows absolute per-pixel differences between the High-Resolution and Super-Resolution images. 
                        Brighter regions indicate larger errors, highlighting areas where the SR model deviates most from the ground truth. 
                        </div> """, unsafe_allow_html=True 
                        )
        with col2:
            st.markdown( """ 
                        <div style=" text-align: justify; font-size: 0.95rem; color: gray; line-height: 1.5; max-width: 400px; margin: 0 auto; ">
                         Histogram of absolute per-pixel intensity differences between High-Resolution and Super-Resolution images. 
                        The plot shows the magnitude of errors and their frequency, providing insight into the overall reconstruction quality. 
                        </div> """, unsafe_allow_html=True 
                        )

        # =================== METRICS TABLE ===================
        st.subheader("Performance & Quality Metrics")
        if mode == "Single Model":
            metrics_table = st.session_state.metrics_df[selected_model_name]
        else:
            metrics_list = []
            for model_name in st.session_state.model_compare:
                if model_name in st.session_state.metrics_df:
                    metrics_list.append(st.session_state.metrics_df[model_name])
            if metrics_list:
                metrics_table = pd.concat(metrics_list)
            else:
                metrics_table = pd.DataFrame()
        st.dataframe(metrics_table.style.format({
            "Inference Time (ms)": "{:.4f}",
            "MSE": "{:.4f}",
            "PSNR": "{:.4f}",
            "SSIM": "{:.4f}"
        }))

        st.markdown(
        """
        <div style="
            text-align: justify;
            font-size: 0.95rem;
            color: gray;
            line-height: 1.5;
            max-width: 900px;
            margin: 0 auto;
        ">
        The table above summarizes the quantitative performance and quality metrics for the selected super-resolution model. 
        <b>Inference Time</b> indicates computational speed, while <b>MSE</b>, <b>PSNR</b>, and <b>SSIM</b> measure 
        reconstruction accuracy and perceptual quality relative to the High-Resolution ground truth. 
        Lower MSE and higher PSNR/SSIM values correspond to better image reconstruction.
        </div>
        """,
        unsafe_allow_html=True
    )

    


    