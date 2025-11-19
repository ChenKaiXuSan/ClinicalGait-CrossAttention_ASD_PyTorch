# %%
# 一开始的save_CAM里面的jetcolormap的通道保存错了，导致之前训练的代码全是保存成了BGR格式的图片，现在把这些图片转换成RGB格式。
# 新的代码已经改正了这个问题。

# %%
from pathlib import Path

input_path = "/work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch/logs/train/3dcnn_attn_map_True_none_bgr_channel/2025-11-12/17-56-55/test_all_feature_maps"
bgr_path = Path("/work/SKIING/chenkaixu/code/ClinicalGait-CrossAttention_ASD_PyTorch/logs/train")



# %%
# find bgr paths 
bgr_folders = []

for p in bgr_path.rglob("*test_all_feature_maps*"):
    if p.name == "test_all_feature_maps" and p.is_dir():
        bgr_folders.append(p)
        print(f"Found BGR folder: {p}")


print(f"Total BGR folders found: {len(bgr_folders)}")

# %%
from tqdm import tqdm
from PIL import Image
import cv2 

for bgr_folder in tqdm(bgr_folders):
    output_path = bgr_folder.parent / "test_all_feature_maps_rgb"
    output_path.mkdir(parents=True, exist_ok=True)

    for one_img in bgr_folder.rglob("*.png"):
        if one_img.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
            continue

        # Load BGR image
        img_bgr = cv2.imread(str(one_img))

        # Convert BGR to RGB
        rgb_array = img_bgr

        # Save RGB image
        output_img_path = one_img.with_name(one_img.stem + "_rgb" + one_img.suffix)
        output_img_path = output_path / one_img.relative_to(bgr_folder)
        output_img_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_array).save(output_img_path)


