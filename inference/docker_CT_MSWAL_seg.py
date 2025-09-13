import numpy as np
import torch
import torch.nn.functional as F
import glob
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
)
from monai.inferers import sliding_window_inference

from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead
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
        default='/models/CT_MSWAL_seg_best_model.pth'
    )
    return parser.parse_args()


load_image = LoadImage(ensure_channel_first=True)
val_transforms = Compose([
    EnsureType(),
    Spacing(pixdim=(1.0, 1.0, 1.25), mode="bilinear"),
    ScaleIntensityRangePercentiles(lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True, channel_wise=True),
])

args = get_args()
image_size = 112
num_classes = 8

val_imgs = glob.glob(args.input_dir + '/*.nii.gz')

feature_model, autocast_dtype = setup_and_build_model_3d(args)
autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
cudnn.benchmark = False  # adjust for final inference
seg_model = UNETRHead(feature_model, 1, image_size, num_classes, autocast_ctx)
seg_model.train()
ps = seg_model.load_state_dict(torch.load(args.decoder_weights), strict=False)
seg_model.eval()
seg_model.cuda()

with torch.no_grad():
    for val_img_path in val_imgs:
        image = load_image(val_img_path)
        orig_shape = image.shape[1:]
        orig_affine = image.peek_pending_affine()

        image = val_transforms(image)
        image = image.unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        logits = sliding_window_inference(
            image,
            (image_size, image_size, image_size),
            2,
            seg_model,
            overlap=0.5,
            device='cpu'
        )

        logits = torch.softmax(logits, dim=1)
        logits = F.interpolate(logits, size=orig_shape, mode='trilinear')
        logits = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        logits = nib.Nifti1Image(logits, orig_affine)
        nib.save(logits, f"{args.output_dir}/{val_img_path.split('/')[-1].replace('_0000.nii.gz', '.nii.gz')}")

print("Validation completed")
