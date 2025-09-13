import numpy as np
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
        default='/models/CT_abdomen_disease_classify_best_model.pth'
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
val_imgs = glob.glob(args.input_dir + '/*.nii.gz')
feature_model, autocast_dtype = setup_and_build_model_3d(args)
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
cudnn.benchmark = False  # adjust for final inference
cls_model = LinearHead(feature_model, num_classes, autocast_ctx, n_last_layers=1, cat_tokens=True)
cls_model.train()
ps = cls_model.load_state_dict(torch.load(args.decoder_weights), strict=False)
cls_model.eval()
cls_model.cuda()

val_probs = []
val_names = []
with torch.no_grad():
    for val_img_path in val_imgs:
        image = val_transforms(val_img_path)

        val_names.append(val_img_path.split('/')[-1].replace('.nii.gz', ''))
        image = image.unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        logits = cls_model(image)
        logits_flip1 = cls_model(torch.flip(image, dims=[-1]))
        logits_flip2 = cls_model(torch.flip(image, dims=[-2]))
        logits_flip3 = cls_model(torch.flip(image, dims=[-3]))
        probs = (torch.sigmoid(logits) + torch.sigmoid(logits_flip1) + torch.sigmoid(logits_flip2) +
                 torch.sigmoid(logits_flip3)) / 4
        val_probs.append(probs.cpu().numpy())

print("Validation completed")

val_probs = np.concatenate(val_probs, axis=0)

column_order = ['bile_duct_stone', 'cholecystitis', 'splenomegaly', 'liver_cyst', 'renal_atrophy', 'kidney_stone',
                'fatty_liver', 'intestinal_obstruction', 'atherosclerosis', 'appendicolith',
                'intrahepatic_bile_duct_dilatation', 'adrenal_nodule', 'peritonitis', 'renal_cyst',
                'colorectal_cancer_(possible)', 'hydronephrosis', 'liver_lesion', 'liver_calcifications', 'ascites',
                'gallstone', 'appendicitis_(possible)', 'bile_duct_dilatation', 'adrenal_hyperplasia',
                'lymphadenopathy']
df_dict = {'Name': val_names}
for i, column in enumerate(column_order):
    df_dict[column] = val_probs[:, i]

pd.DataFrame(df_dict).to_csv(f"{args.output_dir}/predictions.csv", index=False)
