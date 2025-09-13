import torch
import glob
import argparse
import torch.backends.cudnn as cudnn
from functools import partial
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
        default='./dinov2/configs/train/vit3d_lora.yaml'
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
        default='/models/teacher_checkpoint.pth'
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/outputs",
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
        "--input-dir",
        type=str,
        help="Path to dataset JSON file containing image and label paths",
        default='/workspace/inputs'
    )
    parser.add_argument(
        "--decoder-weights",
        type=str,
        help="Path to decoder weights for segmentation head",
        default='/models/MRI_endo_classify_best_model.pth'
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

val_imgs = glob.glob(args.input_dir + '/*.nii.gz')

feature_model, autocast_dtype = setup_and_build_model_3d(args)
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
cudnn.benchmark = False  # adjust for final inference
cls_model = LinearHead(feature_model, num_classes, autocast_ctx, n_last_layers=1, cat_tokens=False)
cls_model.train()
f0_weights = torch.load(args.decoder_weights, map_location='cpu')
f1_weights = torch.load('/models/MRI_endo_classify_best_model_f1.pth', map_location='cpu')
f2_weights = torch.load('/models/MRI_endo_classify_best_model_f2.pth', map_location='cpu')
cls_model.eval()
cls_model.cuda()

val_probs = []
val_names = []
with torch.no_grad():
    for val_img_path in val_imgs:
        image = val_transforms(val_img_path)

        val_names.append(val_img_path.split('/')[-1].replace('_0000.nii.gz', ''))
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

        val_probs.append(probs[0, 1].item())

print("Validation completed")

pd.DataFrame({
    'Name': val_names,
    'Probability': val_probs,
}).to_csv(f"{args.output_dir}/predictions.csv", index=False)
