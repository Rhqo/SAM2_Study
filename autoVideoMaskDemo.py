import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils.video_utils import show_mask, create_video_from_images
from utils.img_utils import show_points

# Paths and model configurations
sam2_checkpoint = "/workspace/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
image_predictor = build_sam2(model_cfg, sam2_checkpoint, apply_postprocessing=False)

# Directories
video_dir = "/workspace/segment-anything-2/notebooks/videos/bedroom"
output_dir = "/workspace/segment-anything-2/notebooks/annotated_frames"
os.makedirs(output_dir, exist_ok=True)

# Mask generator setup
mask_generator = SAM2AutomaticMaskGenerator(model=image_predictor, pred_iou_thresh=0.97)

# Frame processing
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Initialize video prediction state
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = video_predictor.init_state(video_path=video_dir)

    # Process the first frame
    video_segments = {}
    obj_id_counter = 1  # Counter for unique object IDs

    first_frame_path = os.path.join(video_dir, frame_names[0])
    first_image = np.array(Image.open(first_frame_path).convert("RGB"))

    # Generate masks for the first frame
    masks = mask_generator.generate(first_image)
    print("total: ", len(masks))
    ### activate if you want specific point(s)
    # masks = [{'point_coords': [[200, 300]]}, {'point_coords': [[400, 150]]}]
    
    if masks:
        # Store masks and prepare points for the video predictor
        video_segments[0] = {}
        for mask_info in masks:
            # mask = mask_info['segmentation']  # Binary mask
            # bbox = mask_info['bbox']  # Bounding box
            point = mask_info['point_coords']

            # Convert mask to tensor
            points = np.array(point, dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # Label for positive point

            # Unique object ID for each mask
            obj_id = obj_id_counter
            obj_id_counter += 1

            # Add the generated mask as a prompt for the first frame
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels
            )
            
            # Store the mask for the first frame
            video_segments[0][obj_id] = (out_mask_logits[0] > 0.0).cpu().numpy()

    # Propagate masks through the remaining frames
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Render the segmentation results every few frames
    vis_frame_stride = 1
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        frame_str = f"{out_frame_idx:05d}"
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {frame_str}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.savefig(os.path.join(output_dir, f"{frame_str}.png"))

# Create video from annotated frames
output_video_path = "./children_tracking_demo_video.mp4"
create_video_from_images(output_dir, output_video_path)
