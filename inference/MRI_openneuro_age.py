import numpy as np
import os
import torch
import json
import argparse
import torch.backends.cudnn as cudnn
from functools import partial
from monai.transforms import (
    Resize,
    EnsureType,
    Compose,
    ScaleIntensityRangePercentiles,
    LoadImage,
    SpatialPad,
    CenterSpatialCrop,
    Orientation
)
import pandas as pd

from dinov2.eval.regression3d import LinearHead
from dinov2.eval.setup import setup_and_build_model_3d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
        default='/home/txu/flare2025/dinov2/configs/train/vit3d_lora.yaml'
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
        default='/project/amgrp/txu/flare2025/pretrain_outputs/eval/training_49999/teacher_checkpoint.pth'
    )
    parser.add_argument(
        "--output-dir",
        default="/project/amgrp/txu/flare2025/inference_out/openneuro_ds004856",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="path to cache directory for monai persistent dataset",
        default='.',
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        help="Path to dataset JSON file containing image and label paths",
        default='/project/amgrp/txu/flare2025/MRI_openneuro_age.json'
    )
    parser.add_argument(
        "--decoder-weights",
        type=str,
        help="Path to decoder weights for segmentation head",
        default='/project/amgrp/txu/flare2025/outputs/it50k_MRI_openneuro_age/best_model.pth'
    )
    return parser.parse_args()


image_size = 128
num_classes = 1
val_transforms = Compose([
    LoadImage(ensure_channel_first=True),
    EnsureType(),
    Orientation(axcodes="RAS"),
    ScaleIntensityRangePercentiles(lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True, channel_wise=True),
    Resize(spatial_size=(int(image_size * 1.25 * 0.625), int(image_size * 1.25), int(image_size * 1.25)), mode="trilinear"),
    SpatialPad(spatial_size=(image_size, image_size, image_size), value=-1),
    CenterSpatialCrop(roi_size=(image_size, image_size, image_size)),
])

args = get_args()
os.makedirs(args.output_dir, exist_ok=True)

with open(args.dataset_json, 'r') as f:
    dataset_json = json.load(f)
val_imgs = dataset_json['test']

feature_model, autocast_dtype = setup_and_build_model_3d(args)
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
cudnn.benchmark = True  # adjust for final inference
cls_model = LinearHead(feature_model, num_classes, autocast_ctx, n_last_layers=1, cat_tokens=False)
cls_model.train()
ps = cls_model.load_state_dict(torch.load(args.decoder_weights), strict=False)
print(ps)
cls_model.eval()
cls_model.cuda()

val_preds = []
val_labels = []
val_names = []
with torch.no_grad():
    for val_data in val_imgs:
        image = val_transforms(val_data['image'])

        val_names.append(val_data['image'].split('/')[-1])
        image = image.unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        logits = cls_model(image)

        val_labels.append(val_data['label'])
        val_preds.append(logits[0].item())

print("Validation completed")

train_mean, train_std = 60.170833333333334, 17.423164158830495
val_preds = np.array([pred * train_std + train_mean for pred in val_preds])
val_labels = np.array(val_labels)
val_mae = (np.abs(val_preds - val_labels)).mean()
print(f"Validation MAE: {val_mae}")

pd.DataFrame({
    'name': val_names,
    'prediction': val_preds,
}).to_csv(f"{args.output_dir}/predictions.csv", index=False)
