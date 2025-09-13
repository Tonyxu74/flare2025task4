import os
import torch
import json
import argparse
import torch.backends.cudnn as cudnn
from functools import partial
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from monai.transforms import (
    Spacing,
    EnsureType,
    Compose,
    ScaleIntensityRangePercentiles,
    LoadImage,
    CropForeground,
    SpatialPad,
    CenterSpatialCrop,
    Orientation
)
import pandas as pd

from dinov2.eval.classification3d import LinearHead
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
        default='/project/amgrp/txu/flare2025/pretrain_outputs/eval/training_124999/teacher_checkpoint.pth'
    )
    parser.add_argument(
        "--output-dir",
        default="/project/amgrp/txu/flare2025/inference_out/endo_classify",
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
        default='/project/amgrp/txu/flare2025/MRI_endo_classify.json'
    )
    parser.add_argument(
        "--decoder-weights",
        type=str,
        help="Path to decoder weights for segmentation head",
        default='/project/amgrp/txu/flare2025/outputs/it125k_MRI_endo_classify_lr1e-3/best_model.pth'
    )
    return parser.parse_args()

image_size = 128
num_classes = 2
val_transforms = Compose([
    LoadImage(ensure_channel_first=True),
    EnsureType(),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
    ScaleIntensityRangePercentiles(lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True, channel_wise=True),
    CropForeground(select_fn=lambda x: x > -1),
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
f0_weights = torch.load(args.decoder_weights, map_location='cpu')
f1_weights = torch.load('/project/amgrp/txu/flare2025/outputs/it125k_MRI_endo_classify_lr5e-5_f1/best_model.pth', map_location='cpu')
f2_weights = torch.load('/project/amgrp/txu/flare2025/outputs/it125k_MRI_endo_classify_lr5e-4_f2/best_model.pth', map_location='cpu')
cls_model.eval()
cls_model.cuda()

val_preds = []
val_probs = []
val_labels = []
val_names = []
with torch.no_grad():
    for val_data in val_imgs:
        image = val_transforms(val_data['image'])

        val_names.append(val_data['image'].split('/')[-1])
        image = image.unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        cls_model.train()
        cls_model.load_state_dict(f0_weights, strict=False)
        cls_model.eval()
        logits0 = cls_model(image)

        cls_model.train()
        cls_model.load_state_dict(f1_weights, strict=False)
        cls_model.eval()
        logits1 = cls_model(image)

        cls_model.train()
        cls_model.load_state_dict(f2_weights, strict=False)
        cls_model.eval()
        logits2 = cls_model(image)

        probs = (torch.softmax(logits0, dim=1) + torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1))/3
        preds = torch.argmax(probs, dim=1)

        val_labels.append(val_data['label'])
        val_probs.append(probs[0, 1].item())
        val_preds.append(preds[0].item())

print("Validation completed")

auc = roc_auc_score(val_labels, val_probs)
bal_acc = balanced_accuracy_score(val_labels, val_preds)
print(f"Validation AUC: {auc}")
print(f"Validation Balanced Accuracy: {bal_acc}")

pd.DataFrame({
    'name': val_names,
    'prediction': val_probs,
}).to_csv(f"{args.output_dir}/predictions.csv", index=False)
