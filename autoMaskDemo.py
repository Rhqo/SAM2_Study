import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.img_utils import show_anns, show_bboxes
import os

checkpoint = "/workspace/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2 = build_sam2(model_cfg, checkpoint, apply_postprocessing=False)


mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    # points_per_side=64,
    # points_per_batch=128,
    # pred_iou_thresh=0.7,
    # stability_score_thresh=0.92,
    # stability_score_offset=0.7,
    # crop_n_layers=crop_n,
    # box_nms_thresh=0.7,
    # crop_n_points_downscale_factor=2,
    # min_mask_region_area=25.0,
    # use_m2m=False,
)

# image = Image.open('/workspace/segment-anything-2/notebooks/images/cars.jpg')
image = Image.open('/workspace/segment-anything-2/notebooks/videos/bedroom/00000.jpg')
image = np.array(image.convert("RGB"))
output_dir = "/workspace/segment-anything-2/output_img"

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    masks = mask_generator.generate(image)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)
    # show_bboxes(masks)
    plt.axis('off')
    # plt.show()
    plt.savefig(os.path.join(output_dir, f"output_bedroom.png"))

print(masks[0])