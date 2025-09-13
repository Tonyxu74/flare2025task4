import numpy as np
import os
import torch
import torch.nn.functional as F
import json
import argparse
import torch.backends.cudnn as cudnn
import nibabel as nib
from functools import partial
from scipy.ndimage import center_of_mass
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
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
import pandas as pd

from dinov2.eval.segcls3d import UNETRClsHead
from dinov2.eval.setup import setup_and_build_model_3d


class UNETRInferenceHead(UNETRClsHead):
    def forward_seg(self, x_in):
        x2, x3, x4, x, cls_token = self.forward_features_multi(x_in)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(x2))
        enc3 = self.encoder3(self.proj_feat(x3))
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out_seg = self.decoder2(dec1, enc1)

        return self.out(out_seg)

    def forward_cls(self, x_in):
        x2, x3, x4, x, cls_token = self.forward_features_multi(x_in)
        out_cls = self.linear1(cls_token)
        out_cls = self.act_fn(out_cls)
        out_cls = self.bn1(out_cls)
        out_cls = self.linear2(out_cls)

        return out_cls


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
        default="/project/amgrp/txu/flare2025/inference_out/EMIDEC_heart_seg_and_class",
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
        default='/project/amgrp/txu/flare2025/MRI_EMIDEC.json'
    )
    parser.add_argument(
        "--decoder-weights",
        type=str,
        help="Path to decoder weights for segmentation head",
        default='/project/amgrp/txu/flare2025/outputs/test_segcls_maxpool_spacing/best_model.pth'
    )
    return parser.parse_args()


load_image = LoadImage(ensure_channel_first=True)
val_transforms = Compose([
    EnsureType(),
    Spacing(pixdim=(1.5, 1.5, 1.0), mode="bilinear"),
    ScaleIntensityRangePercentiles(lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True, channel_wise=True),
])

args = get_args()
os.makedirs(args.output_dir, exist_ok=True)
image_size = 112
num_classes = 5
post_label = AsDiscrete(to_onehot=num_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

with open(args.dataset_json, 'r') as f:
    dataset_json = json.load(f)
val_imgs = dataset_json['test']

feature_model, autocast_dtype = setup_and_build_model_3d(args)
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
cudnn.benchmark = True  # adjust for final inference
seg_model = UNETRInferenceHead(feature_model, 1, image_size, num_classes, autocast_ctx)
seg_model.train()
ps = seg_model.load_state_dict(torch.load(args.decoder_weights), strict=False)
print(ps)
seg_model.eval()
seg_model.cuda()

total_val_dice = 0
val_steps = 0
val_preds = []
val_probs = []
val_labels = []
val_names = []
with torch.no_grad():
    for val_data in val_imgs:
        image = load_image(val_data['image'])
        label = nib.load(val_data['label_seg'])
        assert image.shape[1:] == label.shape, "Image and label shapes do not match"

        val_names.append(val_data['image'].split('/')[-1])
        image = val_transforms(image)
        image = image.unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        logits = sliding_window_inference(
            image,
            (image_size, image_size, image_size),
            2,
            seg_model.forward_seg,
            overlap=0.75,
            # cval=-1.
        )
        seg_preds = torch.argmax(logits, dim=1)[0]
        center_x, center_y, center_z = center_of_mass((seg_preds > 0).cpu().numpy())

        x_start = int(max(0, center_x - image_size // 2))
        x_end = min(seg_preds.shape[0], x_start + image_size)
        y_start = int(max(0, center_y - image_size // 2))
        y_end = min(seg_preds.shape[1], y_start + image_size)
        z_start = int(max(0, center_z - image_size // 2))
        z_end = min(seg_preds.shape[2], z_start + image_size)
        x_crop = image[:, :, x_start:x_end, y_start:y_end, z_start:z_end]

        # pad symmetrically back to image_size^3
        pad_x = image_size - (x_end - x_start)
        pad_y = image_size - (y_end - y_start)
        pad_z = image_size - (z_end - z_start)

        # (left, right, top, bottom, front, back) for 3D padding
        padding = (
            pad_z // 2, pad_z - pad_z // 2,
            pad_y // 2, pad_y - pad_y // 2,
            pad_x // 2, pad_x - pad_x // 2,
        )
        x_crop = F.pad(x_crop, padding, mode='constant', value=-1)
        cls_logits = seg_model.forward_cls(x_crop)
        cls_logits = torch.softmax(cls_logits, dim=1)

        val_labels.append(val_data['label_cls'])
        val_probs.append(cls_logits[0, 1].item())
        val_preds.append(cls_logits[0, 1].item() > 0.5)

        logits = torch.softmax(logits, dim=1)
        logits = F.interpolate(logits, size=label.shape, mode='trilinear')
        logits = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        logits = nib.Nifti1Image(logits, label.affine)
        nib.save(logits, f"{args.output_dir}/{val_data['label_seg'].split('/')[-1]}")

        # make sure logit saving worked
        label_tensor = torch.tensor(label.get_fdata(), dtype=torch.long).unsqueeze(0)
        pred = nib.load(f"{args.output_dir}/{val_data['label_seg'].split('/')[-1]}")
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

val_labels = [int(x == 'P') for x in val_labels]
auc = roc_auc_score(val_labels, val_probs)
bal_acc = balanced_accuracy_score(val_labels, val_preds)
print(f"Validation AUC: {auc}")
print(f"Validation Balanced Accuracy: {bal_acc}")

pd.DataFrame({
    'name': val_names,
    'prediction': val_probs,
}).to_csv(f"{args.output_dir}/predictions.csv", index=False)
