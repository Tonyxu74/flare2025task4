# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    EnsureTyped,
    Resized,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandSpatialCropd,
    CenterSpatialCropd,
    Identityd,
    OneOf,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    Lambdad,
    SpatialPadd,
    RandRotate90d,
    Spacingd,
)
from torchio.transforms import RandomAffine
import torch


def make_classification_transform_3d(dataset_name: str, image_size: int, min_int: float):
    """
    Create a training and validation transform for 3D classification tasks.

    Args:
        dataset_name: Name of the classification dataset.
        image_size: Size of the image to be used for training.
        min_int: Minimum intensity value to map the image to.
    Returns:
        Training and validation transforms.
    """

    if image_size == 0:
        resize_transform = Identityd(keys=["image"])
    else:
        resize_transform = Resized(keys=["image"], spatial_size=(image_size, image_size, image_size), mode="trilinear")

    if dataset_name == 'ICBM':
        def label_map(x):
            if 20 <= x < 30:
                return 0
            elif 30 <= x < 40:
                return 1
            elif 40 <= x < 50:
                return 2
            elif 50 <= x <= 60:
                return 3

        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                resize_transform,
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                Lambdad(keys=["label"], func=label_map)
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                resize_transform,
                Lambdad(keys=["label"], func=label_map)
            ]
        )

    elif dataset_name == 'COVID-CT-MD':
        def label_map(x):
            if x == 'Normal':
                return 0
            elif x == 'COVID-19':
                return 1
            elif x == 'Cap':
                return 2
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                   keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                Resized(keys=["image"], spatial_size=(144, 144, 112), mode="trilinear"),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                Lambdad(keys=["label"], func=label_map)
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                   keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                Resized(keys=["image"], spatial_size=(144, 144, 112), mode="trilinear"),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
                Lambdad(keys=["label"], func=label_map)
            ]
        )
    elif dataset_name == 'MRI_ABIDEII':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.25, 1.25, 1.25), mode="bilinear"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.25, 1.25, 1.25), mode="bilinear"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )
    elif dataset_name == 'MRI_UPenn_GBM':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                Resized(keys=["image"], spatial_size=(int(image_size*1.25), int(image_size*1.25), int(image_size * 1.25* 0.64583333333)), mode="trilinear"),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                Resized(keys=["image"], spatial_size=(int(image_size*1.25), int(image_size*1.25), int(image_size * 1.25* 0.64583333333)), mode="trilinear"),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )

    elif dataset_name == 'CT_abdomen_disease_classify':

        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                Lambdad(keys=["label"], func=lambda x: torch.tensor(x[0]).float())
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
                Lambdad(keys=["label"], func=lambda x: torch.tensor(x[0]).float())
            ]
        )

    elif dataset_name == 'MRI_openneuro_phen':
        def label_map(x):
            if x == 'CONTROL':
                return 0
            elif x == 'SCHZ':
                return 1
            elif x == 'BIPOLAR':
                return 2
            elif x == 'ADHD':
                return 3
            else:
                raise ValueError(f"Unknown label: {x}")

        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                Resized(keys=["image"], spatial_size=(int(image_size * 1.25), int(image_size * 1.25), int(image_size * 1.25 * 0.6875)), mode="trilinear"),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                Lambdad(keys=["label"], func=label_map),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                Resized(keys=["image"], spatial_size=(int(image_size * 1.25), int(image_size * 1.25), int(image_size * 1.25 * 0.6875)), mode="trilinear"),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
                Lambdad(keys=["label"], func=label_map),
            ]
        )

    elif dataset_name == 'MRI_openneuro_age':

        train_mean, train_std = 60.170833333333334, 17.423164158830495
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                Resized(keys=["image"], spatial_size=(int(image_size * 1.25 * 0.625), int(image_size * 1.25), int(image_size * 1.25)), mode="trilinear"),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                Lambdad(keys=["label"], func=lambda x: (x - train_mean) / train_std),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                Resized(keys=["image"], spatial_size=(int(image_size * 1.25 * 0.625), int(image_size * 1.25), int(image_size * 1.25)), mode="trilinear"),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
                Lambdad(keys=["label"], func=lambda x: (x - train_mean) / train_std),
            ]
        )

    elif dataset_name == 'MRI_endo_classify':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                RandSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size), random_size=False),
                OneOf(transforms=[
                    RandomAffine(include=["image"], p=0.3, degrees=(30, 30, 30),
                                 scales=(0.8, 1.25), translation=(0.1, 0.1, 0.1),
                                 default_pad_value=min_int),
                    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.5, 2)),
                    RandGaussianSharpend(keys=["image"], prob=0.3),
                    RandGaussianSmoothd(keys=["image"], prob=0.3),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.002),
                ]),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 2)),
                RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                EnsureTyped(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0.05, upper=99.95, b_min=min_int, b_max=1, clip=True, channel_wise=True
                ),
                CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > min_int),
                SpatialPadd(keys=["image"], spatial_size=(image_size, image_size, image_size), value=min_int),
                CenterSpatialCropd(keys=["image"], roi_size=(image_size, image_size, image_size)),
            ]
        )

    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return train_transforms, val_transforms

