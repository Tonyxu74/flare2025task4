import numpy as np
import os
import torch
import torch.nn.functional as F
import json
import argparse
import torch.backends.cudnn as cudnn
import nibabel as nib
from functools import partial
from monai.transforms import (
    Spacing,
    EnsureType,
    Compose,
    ScaleIntensityRangePercentiles,
    LoadImage,
    AsDiscrete,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead
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
        default="/project/amgrp/txu/flare2025/inference_out/abdomen_organ_seg",
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
        default='/project/amgrp/txu/flare2025/CT_abdomen_organ_seg.json'
    )
    parser.add_argument(
        "--decoder-weights",
        type=str,
        help="Path to decoder weights for segmentation head",
        default='/project/amgrp/txu/flare2025/outputs/it50k_CT_abdomen_organ_seg/best_model.pth'
    )
    return parser.parse_args()


load_image = LoadImage(ensure_channel_first=True)
val_transforms = Compose([
    EnsureType(),
    Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
    ScaleIntensityRangePercentiles(lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True, channel_wise=True),
])

args = get_args()
os.makedirs(args.output_dir, exist_ok=True)
image_size = 112
num_classes = 14
post_label = AsDiscrete(to_onehot=num_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

with open(args.dataset_json, 'r') as f:
    dataset_json = json.load(f)
val_imgs = dataset_json['test']

feature_model, autocast_dtype = setup_and_build_model_3d(args)
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
cudnn.benchmark = True  # adjust for final inference
seg_model = UNETRHead(feature_model, 1, image_size, num_classes, autocast_ctx)
seg_model.train()
ps = seg_model.load_state_dict(torch.load(args.decoder_weights), strict=False)
print(ps)
seg_model.eval()
seg_model.cuda()

total_val_dice = 0
val_steps = 0
with torch.no_grad():
    for val_data in val_imgs:
        image = load_image(val_data['image'])
        label = nib.load(val_data['label'])
        assert image.shape[1:] == label.shape, "Image and label shapes do not match"

        image = val_transforms(image)
        image = image.unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        logits = sliding_window_inference(
            image,
            (image_size, image_size, image_size),
            2,
            seg_model,
            overlap=0.75,
            # cval=-1.
        )
        logits = torch.softmax(logits, dim=1)
        logits = F.interpolate(logits, size=label.shape, mode='trilinear')
        logits = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        logits = nib.Nifti1Image(logits, label.affine)
        nib.save(logits, f"{args.output_dir}/{val_data['label'].split('/')[-1]}")

        # make sure logit saving worked
        label_tensor = torch.tensor(label.get_fdata(), dtype=torch.long).unsqueeze(0)
        pred = nib.load(f"{args.output_dir}/{val_data['label'].split('/')[-1]}")
        pred_tensor = torch.tensor(pred.get_fdata(), dtype=torch.long).unsqueeze(0)

        label_tensor = post_label(label_tensor).unsqueeze(0)
        pred_tensor = post_label(pred_tensor).unsqueeze(0)

        dice_metric(y_pred=pred_tensor, y=label_tensor)
        val_dice = dice_metric.aggregate().item()
        print(val_dice)
        dice_metric.reset()

        total_val_dice += val_dice
        val_steps += 1

print(f"Validation Dice Score: {total_val_dice / val_steps}")
print("Validation completed")
