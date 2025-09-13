import numpy as np
import os
import torch
import json
import argparse
import torch.backends.cudnn as cudnn
from functools import partial
from sklearn.metrics import average_precision_score
from monai.transforms import (
    Spacing,
    EnsureType,
    Compose,
    ScaleIntensityRangePercentiles,
    LoadImage,
    SpatialPad,
    CenterSpatialCrop,
    Orientation
)
import pandas as pd

from dinov2.eval.multilabel3d import LinearHead
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
        default="/project/amgrp/txu/flare2025/inference_out/abdomen_disease_classify",
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
        default='/project/amgrp/txu/flare2025/CT_abdomen_disease_classify.json'
    )
    parser.add_argument(
        "--decoder-weights",
        type=str,
        help="Path to decoder weights for segmentation head",
        default='/project/amgrp/txu/flare2025/outputs/it125k_CT_abdomen_disease_classify_lr5e-3_spacing/best_model.pth'
    )
    return parser.parse_args()


image_size = 144
num_classes = 24
val_transforms = Compose([
    LoadImage(ensure_channel_first=True),
    EnsureType(),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(2.0, 2.0, 2.0), mode='bilinear'),
    ScaleIntensityRangePercentiles(lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True, channel_wise=True),
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
cls_model = LinearHead(feature_model, num_classes, autocast_ctx, n_last_layers=1, cat_tokens=True)
cls_model.train()
ps = cls_model.load_state_dict(torch.load(args.decoder_weights), strict=False)
print(ps)
cls_model.eval()
cls_model.cuda()

val_probs = []
val_labels = []
val_names = []
with torch.no_grad():
    for val_data in val_imgs:
        image = val_transforms(val_data['image'])

        val_names.append(val_data['image'].split('/')[-1])
        image = image.unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        logits = cls_model(image)
        logits_flip1 = cls_model(torch.flip(image, dims=[-1]))
        logits_flip2 = cls_model(torch.flip(image, dims=[-2]))
        logits_flip3 = cls_model(torch.flip(image, dims=[-3]))
        probs = (torch.sigmoid(logits) + torch.sigmoid(logits_flip1) + torch.sigmoid(logits_flip2) +
                 torch.sigmoid(logits_flip3)) / 4
        val_labels.append(val_data['label'][0])
        val_probs.append(probs.cpu().numpy())

print("Validation completed")

val_labels = np.array(val_labels)
val_probs = np.concatenate(val_probs, axis=0)
val_map = average_precision_score(val_labels, val_probs)
print(f"Validation MAP: {val_map}")

column_order = ['bile_duct_stone', 'cholecystitis', 'splenomegaly', 'liver_cyst', 'renal_atrophy', 'kidney_stone',
                'fatty_liver', 'intestinal_obstruction', 'atherosclerosis', 'appendicolith',
                'intrahepatic_bile_duct_dilatation', 'adrenal_nodule', 'peritonitis', 'renal_cyst',
                'colorectal_cancer_(possible)', 'hydronephrosis', 'liver_lesion', 'liver_calcifications', 'ascites',
                'gallstone', 'appendicitis_(possible)', 'bile_duct_dilatation', 'adrenal_hyperplasia',
                'lymphadenopathy']
df_dict = {'name': val_names}
for i, column in enumerate(column_order):
    df_dict[column] = val_probs[:, i]

pd.DataFrame(df_dict).to_csv(f"{args.output_dir}/predictions.csv", index=False)
