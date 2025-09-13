# Self-Distillation Representation Learning and Parameter Efficient Fine-Tuning for Pretrained Models in Multimodal 3D Medical Imaging

Submission to [FLARE 2025 Challenge Task 4: Foundation Models for 3D CT and MRI](https://www.codabench.org/competitions/7150/#/pages-tab). Project writeup [here](https://openreview.net/forum?id=5Zp3iXRBYc).

## Environments and Requirements

We use Python 3.9 for this project. To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

Datasets used in this project and instructions for access can be found in the [original challenge page](https://www.codabench.org/competitions/7150/#/pages-tab).

To set up datasets and JSONs for pretraining and downstream finetuning, refer to the [3DINO](https://github.com/AICONSlab/3DINO) codebase.

## Pretraining

To pretrain the ViT-Large model, run this command using 2 A100-80GB GPUs. Dataloader will save preprocessed data to `--cache-dir`, which could be a faster temporary storage system if training on a SLURM cluster.

```train
PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node 2 --master_port 29502 dinov2/train/train3d.py \
  --config-file 'dinov2/configs/ssl3d_default_config.yaml' \
  --output-dir 'path/to/output_dir' \
  --cache-dir 'path/to/cache_dir'
```

> ⚠ NOTE: we do not run high-resolution adaptation for this challenge due to time constraints.  

## Finetuning

TODO: Describe for each finetuning script (seg, cls, reg, multilabel, segcls, survival), note which datasets run which

Commands for finetuning on each downstream dataset can be found below:  

### Abdomen disease classification (CT, validation and test set)

```train
PYTHONPATH=. python dinov2/eval/multilabel3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'CT_abdomen_disease_classify' \
  --dataset-percent 100 \
  --n-last-layers 1 \
  --base-data-dir 'path/to/base/data_dir' \
  --epochs 300 \
  --epoch-length 30 \
  --eval-iters 30 \
  --warmup-iters 1800 \
  --image-size 144 \
  --batch-size 16 \
  --num-workers 6 \
  --learning-rate 1e-3 \
  --cache-dir 'path/to/cache_dir'
```

### Abdomen lesion segmentation (CT, validation set)

```train
PYTHONPATH=. python dinov2/eval/segmentation3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'CT_abdomen_lesion_seg' \
  --dataset-percent 100 \
  --base-data-dir 'path/to/base/data_dir' \
  --segmentation-head 'UNETR' \
  --epochs 100 \
  --epoch-length 300 \
  --eval-iters 600 \
  --warmup-iters 3000 \
  --image-size 112 \
  --batch-size 2 \
  --num-workers 20 \
  --learning-rate 1e-4 \
  --cache-dir 'path/to/cache_dir' \
  --resize-scale 1.0
```

### Abdomen organ segmentation (CT, validation set)

```train
PYTHONPATH=. python dinov2/eval/segmentation3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'CT_abdomen_organ_seg' \
  --dataset-percent 100 \
  --base-data-dir 'path/to/base/data_dir' \
  --segmentation-head 'UNETR' \
  --epochs 100 \
  --epoch-length 300 \
  --eval-iters 600 \
  --warmup-iters 3000 \
  --image-size 112 \
  --batch-size 2 \
  --num-workers 20 \
  --learning-rate 1e-4 \
  --cache-dir 'path/to/cache_dir' \
  --resize-scale 1.0
```

### Lung lesion segmentation (CT, validation set)

```train
PYTHONPATH=. python dinov2/eval/segmentation3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'CT_lung_lesion_seg' \
  --dataset-percent 100 \
  --base-data-dir 'path/to/base/data_dir' \
  --segmentation-head 'UNETR' \
  --epochs 100 \
  --epoch-length 300 \
  --eval-iters 600 \
  --warmup-iters 3000 \
  --image-size 112 \
  --batch-size 2 \
  --num-workers 20 \
  --learning-rate 1e-4 \
  --cache-dir 'path/to/cache_dir' \
  --resize-scale 1.0
```

### MSWAL (CT, test set)

```train
PYTHONPATH=. python dinov2/eval/segmentation3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'CT_MSWAL_seg' \
  --dataset-percent 100 \
  --base-data-dir 'path/to/base/data_dir' \
  --segmentation-head 'UNETR' \
  --epochs 100 \
  --epoch-length 300 \
  --eval-iters 600 \
  --warmup-iters 3000 \
  --image-size 112 \
  --batch-size 2 \
  --num-workers 20 \
  --learning-rate 1e-3 \
  --cache-dir 'path/to/cache_dir' \
  --resize-scale 1.0
```

### Abdomen organ segmentation (MRI, test set)

```train
PYTHONPATH=. python dinov2/eval/segmentation3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'MRI_abdomen_seg' \
  --dataset-percent 100 \
  --base-data-dir 'path/to/base/data_dir' \
  --segmentation-head 'UNETR' \
  --epochs 100 \
  --epoch-length 300 \
  --eval-iters 600 \
  --warmup-iters 3000 \
  --image-size 112 \
  --batch-size 2 \
  --num-workers 20 \
  --learning-rate 3e-4 \
  --cache-dir 'path/to/cache_dir' \
  --resize-scale 1.0
```

### ABIDEII classification (MRI, validation set)

```train
PYTHONPATH=. python dinov2/eval/classification3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'MRI_ABIDEII' \
  --dataset-percent 100 \
  --n-last-layers 1 \
  --base-data-dir 'path/to/base/data_dir' \
  --epochs 150 \
  --epoch-length 30 \
  --eval-iters 30 \
  --warmup-iters 600 \
  --image-size 128 \
  --batch-size 16 \
  --num-workers 6 \
  --learning-rate 5e-4 \
  --cache-dir 'path/to/cache_dir'
```

### ATLAS segmentation (MRI, validation set)

```train
PYTHONPATH=. python dinov2/eval/segmentation3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'MRI_ATLAS_seg' \
  --dataset-percent 100 \
  --base-data-dir 'path/to/base/data_dir' \
  --segmentation-head 'UNETR' \
  --epochs 100 \
  --epoch-length 300 \
  --eval-iters 600 \
  --warmup-iters 3000 \
  --image-size 112 \
  --batch-size 2 \
  --num-workers 20 \
  --learning-rate 3e-4 \
  --cache-dir 'path/to/cache_dir' \
  --resize-scale 1.0
```

### EMIDEC segmentation and classification (MRI, validation set)

```train
PYTHONPATH=. python dinov2/eval/segcls3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'MRI_EMIDEC' \
  --dataset-percent 100 \
  --base-data-dir 'path/to/base/data_dir' \
  --epochs 100 \
  --epoch-length 300 \
  --eval-iters 600 \
  --warmup-iters 3000 \
  --image-size 112 \
  --batch-size 2 \
  --num-workers 20 \
  --learning-rate 1e-4 \
  --cache-dir 'path/to/cache_dir' \
  --resize-scale 1.0
```

### Endometriosis classification (MRI, test set)

We averaged 3 folds for the final result, the following is for fold 0.

```train
PYTHONPATH=. python dinov2/eval/classification3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'MRI_endo_classify' \
  --dataset-percent 100 \
  --n-last-layers 1 \
  --base-data-dir 'path/to/base/data_dir' \
  --epochs 150 \
  --epoch-length 30 \
  --eval-iters 30 \
  --warmup-iters 600 \
  --image-size 128 \
  --batch-size 16 \
  --num-workers 6 \
  --learning-rate 1e-3 \
  --cache-dir 'path/to/cache_dir' \
  --fold 0
```

### Openneuro age classification (MRI, validation set)

```train
PYTHONPATH=. python dinov2/eval/regression3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'MRI_openneuro_age' \
  --dataset-percent 100 \
  --n-last-layers 1 \
  --base-data-dir 'path/to/base/data_dir' \
  --epochs 1000 \
  --epoch-length 30 \
  --eval-iters 30 \
  --warmup-iters 1800 \
  --image-size 128 \
  --batch-size 16 \
  --num-workers 6 \
  --learning-rate 5e-3 \
  --cache-dir 'path/to/cache_dir'
```

### Openneuro phenomics classification (MRI, validation set)

```train
PYTHONPATH=. python dinov2/eval/classification3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'MRI_openneuro_phen' \
  --dataset-percent 100 \
  --n-last-layers 1 \
  --base-data-dir 'path/to/base/data_dir' \
  --epochs 150 \
  --epoch-length 30 \
  --eval-iters 30 \
  --warmup-iters 600 \
  --image-size 128 \
  --batch-size 16 \
  --num-workers 6 \
  --learning-rate 5e-4 \
  --cache-dir 'path/to/cache_dir'
```

### UPenn-GBM survival prediction (MRI, validation set)

```train
PYTHONPATH=. python dinov2/eval/survival3d.py \
  --config-file 'dinov2/configs/train/vit3d_lora.yaml' \
  --output-dir 'path/to/output_dir' \
  --pretrained-weights 'path/to/output_dir/eval/training_124999/teacher_checkpoint.pth' \
  --dataset-name 'MRI_UPenn_GBM' \
  --dataset-percent 100 \
  --n-last-layers 1 \
  --n-bins 3 \
  --base-data-dir 'path/to/base/data_dir' \
  --epochs 150 \
  --epoch-length 30 \
  --eval-iters 30 \
  --warmup-iters 600 \
  --image-size 128 \
  --batch-size 16 \
  --num-workers 6 \
  --learning-rate 5e-3 \
  --cache-dir 'path/to/cache_dir'
```

## Inference

To run inference, adjust model, data, and output paths as needed and run the respective command for each downstream dataset: 

```train
PYTHONPATH=. python inference/<dataset_name>.py
```

## Results

We summarize validation results obtained using a ViT pretrained for 50k iterations (latest available checkpoint during validation submission). 

The following table summarizes the results for the segmentation tasks. Results are obtained from averaging the score on all non-background classes.

| Task               | CT Abdomen Lesion | CT Abdomen Organ | CT Lung Lesion | ATLAS23 | EMIDEC Segmentation |
|--------------------|-------------------|------------------|----------------|---------|---------------------|
| Metric             | DSC               | DSC              | DSC            | DSC     | DSC                 |
| Value              | 0.3254            | 0.7511           | 0.5798         | 0.5944  | 0.4969              |

The following table summarizes the results for the prediction tasks (classification, regression, and survival analysis).

| Task            | CT Abdomen Disease | MRI Age | MRI Phenomics |  ABIDEII  | UPenn-GBM | EMIDEC Classification |
|-----------------|--------------------|---------|---------------|-----------|-----------|-----------------------|
| Metric          | mAP                | MAE     | Bal. Acc.     | Bal. Acc. | C-Index   | Bal. Acc.             |
| Value           | 0.2469             | 3.262   | 0.4104        | 0.6439    | 0.5967    | 0.8439                |


## Acknowledgements

We thank the [challenge](https://www.codabench.org/competitions/7150/#/pages-tab) organizers, and the contributors of challenge datasets.

This code is adapted from our work on pretraining 3D ViT models for medical imaging: [3DINO](https://github.com/AICONSlab/3DINO).