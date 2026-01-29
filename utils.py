import numpy as np
import pandas as pd
import torch
import base64
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from streamlit.components.v1 import html as components_html
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


def to_tensor(img):
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

def postprocess(tensor):
    tensor = tensor.squeeze(0).cpu().numpy()
    tensor = np.clip(np.transpose(tensor, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    return tensor


def im_mse(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "Images must be the same shape"
    return np.mean((im1 - im2) ** 2)

def im_psnr(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "Images must be the same shape"
    return peak_signal_noise_ratio(im1, im2)

def im_ssim(im1: np.ndarray, im2: np.ndarray) -> float:
    assert im1.shape == im2.shape, "Images must be the same shape"
    return structural_similarity(im1, im2, channel_axis=2)

def calculate_metrics(gt: np.ndarray, sr: np.ndarray, inference_time: float, selected_model_name: str) -> pd.DataFrame:
    mse = im_mse(gt, sr)
    psnr = im_psnr(gt, sr)
    ssim = im_ssim(gt, sr)

    data = {
        "Inference Time (ms)": [inference_time],
        "MSE": [mse],
        "PSNR": [psnr],
        "SSIM": [ssim]
    }
    return pd.DataFrame(data, index=[selected_model_name])


def compute_global_diff_maps(hr_full, pred_full_list):
    hr_gray = cv2.cvtColor(hr_full, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diffs = []
    for p in pred_full_list:
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(np.abs(hr_gray - g))
    global_max = max([d.max() for d in diffs]) if diffs else 1.0
    diff_color = []
    for d in diffs:
        dn = (d / global_max * 255).clip(0, 255).astype(np.uint8)
        cm = cv2.applyColorMap(dn, cv2.COLORMAP_JET)
        diff_color.append(cm)
    return diff_color


def plot_absolute_pixel_error_histogram(hr_img: np.ndarray, sr_img: np.ndarray, color: str = "#0c21c2") -> plt.Figure:
    
    if hr_img.dtype != np.uint8:
        hr_img = (hr_img * 255).astype(np.uint8)
    if sr_img.dtype != np.uint8:
        sr_img = (sr_img * 255).astype(np.uint8)

    hr_gray = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sr_gray = cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    pixel_diff = np.abs(hr_gray - sr_gray).ravel()
    stats = {
        "min": int(pixel_diff.min()),
        "max": int(pixel_diff.max()),
        "mean": float(pixel_diff.mean()),
        "median": float(np.median(pixel_diff)),
        "std": float(pixel_diff.std()),
        "p95": float(np.percentile(pixel_diff, 95)),
        "p99": float(np.percentile(pixel_diff, 99)),
    }

    max_diff = int(pixel_diff.max())
    bins = np.arange(0, max_diff + 2) # bins from 0 to max_diff inclusive
    
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.set_xlim(0, 25)
    ax.hist(pixel_diff, 
            bins=bins,
            color=color, 
            alpha=0.8, 
            edgecolor='white')
    ax.set_xlabel("Absolute Difference", fontsize=10)
    ax.set_ylabel("Number of Pixels", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()

    return fig, stats


def plot_signed_pixel_error_histogram(
    hr_img: np.ndarray,
    sr_img: np.ndarray,
    color: str = "#0c21c2",
    max_range: int = 15
) -> plt.Figure:

    if hr_img.dtype != np.uint8:
        hr_img = (hr_img * 255).astype(np.uint8)
    if sr_img.dtype != np.uint8:
        sr_img = (sr_img * 255).astype(np.uint8)

    hr_gray = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    sr_gray = cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY).astype(np.int16)

    pixel_diff = (hr_gray - sr_gray).ravel()

    bins = np.arange(-255, 256)

    fig, ax = plt.subplots(figsize=(6, 3.4))

    ax.hist(
        pixel_diff,
        bins=bins,
        color=color,
        alpha=0.8,
        edgecolor='white'
    )

    ax.set_xlim(-max_range, max_range)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel("Difference (HR − SR)", fontsize=10)
    ax.set_ylabel("Number of Pixels", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    return fig


def add_synced_magnifiers(gt_bytes, lr_bytes, sr_bytes, zoom=1, size=150, height=300):
    gt_b64 = base64.b64encode(gt_bytes).decode("ascii")
    lr_b64 = base64.b64encode(lr_bytes).decode("ascii")
    sr_b64 = base64.b64encode(sr_bytes).decode("ascii")

    html_code = f"""
    <style>
        .triple-container {{
            display: flex;
            gap: 20px;
            width: 100%;
            align-items: center;
            justify-content: center;
        }}
        .magnifier-wrapper {{
            position: relative;
            flex: 1;
            user-select: none;
        }}
        .mag-img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .magnifier {{
            position: absolute;
            width: {size}px;
            height: {size}px;
            border: 2px solid black;
            background-repeat: no-repeat;
            pointer-events: none;
            transform: translate(-50%, -50%);
            display: none;
            z-index: 999;
        }}
    </style>

    <div class="triple-container">
        <div class="magnifier-wrapper">
            <img id="img_gt" class="mag-img" src="data:image/png;base64,{gt_b64}">
            <div id="mag_gt" class="magnifier"></div>
        </div>
        <div class="magnifier-wrapper">
            <img id="img_lr" class="mag-img" src="data:image/png;base64,{lr_b64}">
            <div id="mag_lr" class="magnifier"></div>
        </div>
        <div class="magnifier-wrapper">
            <img id="img_sr" class="mag-img" src="data:image/png;base64,{sr_b64}">
            <div id="mag_sr" class="magnifier"></div>
        </div>
    </div>

    <script>
    const imgs = [
        document.getElementById("img_gt"),
        document.getElementById("img_lr"),
        document.getElementById("img_sr")
    ];
    const mags = [
        document.getElementById("mag_gt"),
        document.getElementById("mag_lr"),
        document.getElementById("mag_sr")
    ];

    const zoom = {zoom};
    const size = {size};

    function updateBG() {{
        for (let i = 0; i < imgs.length; i++) {{
            mags[i].style.backgroundSize =
                imgs[i].naturalWidth * zoom + "px " +
                imgs[i].naturalHeight * zoom + "px";
        }}
        autoResize();
    }}

    imgs.forEach(img => img.onload = updateBG);
    document.addEventListener("DOMContentLoaded", updateBG);

    function move(e, index) {{
        const src = imgs[index];
        const rect = src.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;

        if (x < 0 || y < 0 || x > rect.width || y > rect.height) {{
            mags.forEach(m => m.style.display = "none");
            return;
        }}

        mags.forEach(m => m.style.display = "block");

        const xNat = x * (src.naturalWidth / src.clientWidth);
        const yNat = y * (src.naturalHeight / src.clientHeight);

        for (let i = 0; i < imgs.length; i++) {{
            const img = imgs[i];
            const mag = mags[i];

            const xMapped = xNat * (img.clientWidth / src.naturalWidth);
            const yMapped = yNat * (img.clientHeight / src.naturalHeight);

            mag.style.left = xMapped + "px";
            mag.style.top = yMapped + "px";
            mag.style.backgroundImage = "url('" + img.src + "')";
            mag.style.backgroundPosition =
                "-" + (xNat * (img.naturalWidth / src.naturalWidth) * zoom - size/2) +
                "px -" + (yNat * (img.naturalHeight / src.naturalHeight) * zoom - size/2) + "px";
        }}
    }}

    imgs.forEach((img, idx) => {{
        img.addEventListener("mousemove", e => move(e, idx));
        img.addEventListener("mouseleave", () => {{
            mags.forEach(m => m.style.display = "none");
        }});
    }});
    </script>
    """

    components_html(html_code, height=height)



def add_multi_model_magnifiers(
    gt_bytes,
    lr_bytes,
    sr_bytes_list,
    model_names,
    zoom=1,
    size=150
):
    gt_b64 = base64.b64encode(gt_bytes).decode("ascii")
    lr_b64 = base64.b64encode(lr_bytes).decode("ascii")
    sr_b64_list = [base64.b64encode(sr_bytes).decode("ascii") for sr_bytes in sr_bytes_list]

    images = [("HR (Ground Truth)", gt_b64), ("LR Input", lr_b64)] + list(zip(model_names, sr_b64_list))

    top_row_count = 2
    remaining_images = images[top_row_count:]
    rows = [images[:top_row_count]]  # first row: HR + LR

    for i in range(0, len(remaining_images), 2):
        rows.append(remaining_images[i:i+2])

    rows_count = len(rows)
    component_height = max(400, rows_count * 400)  

    html_code = """
    <style>
        .row-container {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-bottom: 20px;
        }}
        .image-wrapper {{
            flex: 0 0 45%; 
            text-align: center;
            position: relative;
        }}
        .image-wrapper img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .magnifier {{
            position: absolute;
            width: {size}px;
            height: {size}px;
            border: 2px solid black;
            background-repeat: no-repeat;
            pointer-events: none;
            transform: translate(-50%, -50%);
            display: none;
            z-index: 999;
        }}
        .img-title {{
            margin-top: 5px;
            color: gray;
            font-weight: bold;
            font-size: 0.95rem;
            font-family: 'Inter', sans-serif;
        }}
    </style>
    """.format(size=size)

    html_code += "<div>"

    for r, row in enumerate(rows):
        html_code += '<div class="row-container">'
        for c, (title, b64) in enumerate(row):
            img_id = f"img_{r}_{c}"
            mag_id = f"mag_{r}_{c}"
            html_code += f"""
            <div class="image-wrapper">
                <img id="{img_id}" src="data:image/png;base64,{b64}">
                <div id="{mag_id}" class="magnifier"></div>
                <div class="img-title">{title}</div>
            </div>
            """
        html_code += "</div>" 

    html_code += "</div>"

    html_code += """
    <script>
    const zoom = {zoom};
    const size = {size};
    const imgs = [];
    const mags = [];

    // collect all images and magnifiers
    document.querySelectorAll('.image-wrapper').forEach(wrapper => {{
        const img = wrapper.querySelector('img');
        const mag = wrapper.querySelector('.magnifier');
        imgs.push(img);
        mags.push(mag);
    }});

    function updateBG() {{
        for (let i = 0; i < imgs.length; i++) {{
            mags[i].style.backgroundSize =
                imgs[i].naturalWidth * zoom + "px " +
                imgs[i].naturalHeight * zoom + "px";
        }}
    }}

    imgs.forEach(img => img.onload = updateBG);
    document.addEventListener("DOMContentLoaded", updateBG);

    function move(e, index) {{
        const src = imgs[index];
        const rect = src.getBoundingClientRect();
        let x = e.clientX - rect.left;
        let y = e.clientY - rect.top;

        if (x < 0 || y < 0 || x > rect.width || y > rect.height) {{
            mags.forEach(m => m.style.display = "none");
            return;
        }}

        mags.forEach(m => m.style.display = "block");

        const xNat = x * (src.naturalWidth / src.clientWidth);
        const yNat = y * (src.naturalHeight / src.clientHeight);

        for (let i = 0; i < imgs.length; i++) {{
            const img = imgs[i];
            const mag = mags[i];

            const xMapped = xNat * (img.clientWidth / src.naturalWidth);
            const yMapped = yNat * (img.clientHeight / src.naturalHeight);

            mag.style.left = xMapped + "px";
            mag.style.top = yMapped + "px";
            mag.style.backgroundImage = "url('" + img.src + "')";
            mag.style.backgroundPosition =
                "-" + (xNat * (img.naturalWidth / src.naturalWidth) * zoom - size/2) +
                "px -" + (yNat * (img.naturalHeight / src.naturalHeight) * zoom - size/2) + "px";
        }}
    }}

    imgs.forEach((img, idx) => {{
        img.addEventListener("mousemove", e => move(e, idx));
        img.addEventListener("mouseleave", () => {{
            mags.forEach(m => m.style.display = "none");
        }});
    }});
    </script>
    """.format(zoom=zoom, size=size)

    components_html(html_code, height=component_height)


    
    