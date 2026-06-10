import os
import cv2
import numpy as np
import pandas as pd
from itertools import combinations
from skimage.metrics import structural_similarity as ssim


def split_vertical_image(img, num_parts=8):
    h, w = img.shape
    part_h = h // num_parts
    return [
        img[i * part_h:(i + 1) * part_h if i < num_parts - 1 else h, :]
        for i in range(num_parts)
    ]


def preprocess(img, size=(224, 224)):
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_method_maps(method_dir, num_parts=8):
    imgs = [
        f for f in os.listdir(method_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if len(imgs) == 0:
        return None

    path = os.path.join(method_dir, imgs[0])
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    parts = split_vertical_image(img, num_parts=num_parts)
    parts = [preprocess(p) for p in parts]
    return parts


def compute_similarity(root_dir, output_csv="attention_similarity.csv", num_parts=8):
    methods = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    maps = {}
    for m in methods:
        method_dir = os.path.join(root_dir, m)
        parts = load_method_maps(method_dir, num_parts=num_parts)
        if parts is not None:
            maps[m] = parts

    results = []

    for m1, m2 in combinations(maps.keys(), 2):
        cos_list, ssim_list = [], []

        for i in range(num_parts):
            A = maps[m1][i]
            B = maps[m2][i]

            cos_val = cosine_similarity(A, B)
            ssim_val = ssim(A, B, data_range=1.0)

            cos_list.append(cos_val)
            ssim_list.append(ssim_val)

            results.append({
                "method_1": m1,
                "method_2": m2,
                "frame": i + 1,
                "cosine_similarity": cos_val,
                "ssim": ssim_val,
            })

        results.append({
            "method_1": m1,
            "method_2": m2,
            "frame": "mean",
            "cosine_similarity": np.mean(cos_list),
            "ssim": np.mean(ssim_list),
        })

        results.append({
            "method_1": m1,
            "method_2": m2,
            "frame": "std",
            "cosine_similarity": np.std(cos_list),
            "ssim": np.std(ssim_list),
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")
    return df


if __name__ == "__main__":
    root_dir = "/work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch/analysis/similarity"

    df = compute_similarity(
        root_dir=root_dir,
        output_csv="/work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch/analysis/attention_similarity.csv",
        num_parts=8
    )

    print(df)