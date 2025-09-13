# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from dinov2.data.loaders import make_segmentation_dataset_3d
from dinov2.data import SamplerType, make_data_loader
from dinov2.eval.segmentation_3d.segmentation_heads import UNETRHead
from dinov2.eval.setup import get_args_parser, setup_and_build_model_3d
from dinov2.eval.segmentation_3d.augmentations import make_transforms
from dinov2.eval.segmentation_3d.metrics import get_metric

import torch
import torch.nn.functional as F
import json
from functools import partial
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data.utils import list_data_collate
from monai.optimizers import WarmupCosineSchedule
from monai.utils.misc import set_determinism
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import loralib as lora
import numpy as np
from scipy.ndimage import center_of_mass

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
set_determinism(seed=0)


class UNETRClsHead(UNETRHead):
    def __init__(self, feature_model, input_channels, image_size, num_classes, autocast_ctx):
        super().__init__(feature_model, input_channels, image_size, num_classes, autocast_ctx)

        size = self.hidden_size
        self.linear1 = torch.nn.Linear(size, self.hidden_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size // 2)
        self.linear2 = torch.nn.Linear(self.hidden_size // 2, 2)

    def forward_features_multi(self, x_in):
        """Pass multi-channel input through feature model by reshaping batch and merging channels"""
        assert x_in.shape[1] == self.input_channels
        B = x_in.shape[0]

        # Change feature channel into individual batches B, C, H, W, D -> B*C, 1, H, W, D
        x_reshape = x_in.reshape(-1, 1, *x_in.shape[2:])
        with self.autocast_ctx():
            x2, x3, x4, x = self.feature_model.get_intermediate_layers(
                x_reshape,
                n=[5, 11, 17, 23],
                return_class_token=True
            )
        # get cls token and patch-level features
        cls_token = x[1]
        x2, x3, x4, x = x2[0], x3[0], x4[0], x[0]

        # reshape to merge channels B*C, N, F -> B, N, C*F
        x2 = x2.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)
        x3 = x3.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)
        x4 = x4.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)
        x = x.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)

        # Merge channels B, N, C*F -> B, N, F
        x2 = self.act_fn(self.channel_merge1(x2))
        x3 = self.act_fn(self.channel_merge2(x3))
        x4 = self.act_fn(self.channel_merge3(x4))
        x = self.act_fn(self.channel_merge4(x))

        return x2, x3, x4, x, cls_token

    def forward(self, x_in):
        """Forward function."""

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

        out_cls = self.linear1(cls_token)
        out_cls = self.act_fn(out_cls)
        out_cls = self.bn1(out_cls)
        out_cls = self.linear2(out_cls)

        return self.out(out_seg), out_cls.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def add_seg_args(parser):
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of finetuning dataset",
    )
    parser.add_argument(
        "--dataset-percent",
        type=int,
        help="Percent of finetuning dataset to use",
        default=100
    )
    parser.add_argument(
        "--base-data-dir",
        type=str,
        help="Base data directory for finetuning dataset",
    )
    parser.add_argument(
        "--train-feature-model",
        action="store_true",
        help="Freeze feature model or not",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total epochs",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Iterations to perform per epoch",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        help="Iterations to perform per evaluation",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Image side length",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        help="Scale factor for resizing images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="path to cache directory for monai persistent dataset"
    )

    return parser


def train_iter(model, batch, optimizer, scheduler, loss_seg, loss_cls, scaler):
    x, y_seg, y_cls = (batch["image"].cuda(), batch["label_seg"].cuda(), batch["label_cls"].cuda())

    fore_in_image = (y_seg > 0).float().mean(dim=(2, 3, 4))[:, 0] > 0.01
    logits_seg, logits_cls = model(x)
    l_seg = loss_seg(logits_seg, y_seg)
    l_cls = loss_cls(logits_cls.squeeze(-1).squeeze(-1).squeeze(-1)[fore_in_image], y_cls[fore_in_image])
    #l_cls = loss_cls(logits_cls.squeeze(-1).squeeze(-1).squeeze(-1), y_cls)
    loss = l_seg + l_cls
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item()


def val_iter(model, batch, metric, image_size, batch_size, overlap=0.5):
    x, y_seg, y_cls = (batch["image"].cuda(), batch["label_seg"].cuda(), batch["label_cls"])
    logits_seg, logits_cls = sliding_window_inference(x, image_size, batch_size, model, overlap=overlap)
    # logits_cls = logits_cls.mean(dim=(2, 3, 4))  # todo: also try max pooling
    # probs = torch.softmax(logits_cls, dim=1)
    # probs = torch.amax(probs, dim=(2,3,4), keepdim=False)
    # preds = torch.argmax(probs, dim=1)
    # preds = (probs[:, 1] > 0.5).long()
    seg_preds = torch.argmax(logits_seg, dim=1)[0]
    center_x, center_y, center_z = center_of_mass((seg_preds > 0).cpu().numpy())

    x_start = int(max(0, center_x - image_size[0] // 2))
    x_end = min(seg_preds.shape[0], x_start + image_size[0])
    y_start = int(max(0, center_y - image_size[0] // 2))
    y_end = min(seg_preds.shape[1], y_start + image_size[0])
    z_start = int(max(0, center_z - image_size[0] // 2))
    z_end = min(seg_preds.shape[2], z_start + image_size[0])
    x_crop = x[:, :, x_start:x_end, y_start:y_end, z_start:z_end]

    # pad symmetrically back to image_size^3
    pad_x = image_size[0] - (x_end - x_start)
    pad_y = image_size[0] - (y_end - y_start)
    pad_z = image_size[0] - (z_end - z_start)

    # (left, right, top, bottom, front, back) for 3D padding
    padding = (
        pad_z // 2, pad_z - pad_z // 2,
        pad_y // 2, pad_y - pad_y // 2,
        pad_x // 2, pad_x - pad_x // 2,
    )

    x_crop = F.pad(x_crop, padding, mode='constant', value=-1)
    logits_cls_crop = model(x_crop)[1].squeeze(-1).squeeze(-1).squeeze(-1)
    probs = torch.softmax(logits_cls_crop, dim=1)
    preds = torch.argmax(probs, dim=1)

    iter_metric = metric(logits_seg, y_seg)
    return iter_metric, probs.cpu().numpy(), preds.cpu().numpy(), y_cls.numpy()


def do_finetune(feature_model, autocast_dtype, args):

    # get transforms, dataset, dataloaders
    train_transforms, val_transforms = make_transforms(
        args.dataset_name,
        args.image_size,
        args.resize_scale,
        min_int=-1.0
    )
    train_ds, val_ds, test_ds, input_channels, num_classes = make_segmentation_dataset_3d(
        args.dataset_name,
        args.dataset_percent,
        args.base_data_dir,
        train_transforms,
        val_transforms,
        args.cache_dir,
        args.batch_size
    )
    train_loader = make_data_loader(
        dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=0,
        sampler_type=SamplerType.SHARDED_INFINITE,
        drop_last=False,
        persistent_workers=True,
        collate_fn=list_data_collate
    )
    val_loader = make_data_loader(
        dataset=val_ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=list_data_collate
    )
    test_loader = make_data_loader(
        dataset=test_ds,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=list_data_collate
    )

    # get model
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    scaler = torch.cuda.amp.GradScaler()
    seg_model = UNETRClsHead(feature_model, input_channels, args.image_size, num_classes, autocast_ctx)

    if args.train_feature_model:
        seg_model.feature_model.train()
    else:
        if 'lora' not in args.config_file:
            seg_model.feature_model.eval()
            for param in seg_model.feature_model.parameters():
                param.requires_grad = False
        else:
            seg_model.train()
            lora.mark_only_lora_as_trainable(seg_model.feature_model)

    trainable_params = [name for name, param in seg_model.named_parameters() if param.requires_grad]
    print(f"Trainable parameters: {trainable_params}")

    # get optimizer, scheduler, loss function, metric
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, seg_model.parameters()), lr=args.learning_rate)
    max_iter = args.epochs * args.epoch_length
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=args.warmup_iters,
        t_total=max_iter
    )
    loss_fn_seg = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_fn_cls = torch.nn.CrossEntropyLoss()

    dice_metric = get_metric(args.dataset_name)

    seg_model.cuda()
    loss_fn_seg.cuda()
    loss_fn_cls.cuda()

    best_val_metric = -1
    train_loss_sum = 0
    iters_list = []
    train_loss_list = []
    val_dice_list = []
    val_per_cls_dice_list = []
    val_bal_acc_list = []
    val_auc_list = []

    for it, train_data in enumerate(train_loader):

        # train for one iteration
        train_loss = train_iter(
            model=seg_model,
            batch=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_seg=loss_fn_seg,
            loss_cls=loss_fn_cls,
            scaler=scaler
        )
        train_loss_sum += train_loss

        if it % 100 == 0:
            print(f"[Iter {it}], Train loss: {train_loss}", flush=True)

        if it % args.eval_iters == 0:
            # valdation
            total_val_dice = 0
            total_per_cls_val_dice = [0 for _ in range(num_classes)]
            val_probs = []
            val_preds = []
            val_labels = []
            val_steps = 0
            seg_model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    (val_dice, val_per_cls_dice), probs, preds, labels = val_iter(
                        model=seg_model,
                        batch=val_data,
                        image_size=(args.image_size,) * 3,
                        batch_size=args.batch_size,
                        metric=dice_metric,
                        overlap=0.
                    )

                    total_val_dice += val_dice
                    for i in range(num_classes):
                        total_per_cls_val_dice[i] += val_per_cls_dice[i]
                    val_probs.append(probs)
                    val_preds.append(preds)
                    val_labels.append(labels)
                    val_steps += 1

            avg_val_dice = total_val_dice / val_steps
            avg_per_cls_val_dice = [total_per_cls_val_dice[i] / val_steps for i in range(num_classes)]
            avg_train_loss = train_loss_sum / args.eval_iters

            val_probs = np.concatenate(val_probs, axis=0)
            val_preds = np.concatenate(val_preds, axis=0)
            val_labels = np.concatenate(val_labels, axis=0)
            val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
            val_auc = roc_auc_score(val_labels, val_probs[:, 1])

            train_loss_list.append(avg_train_loss)
            val_dice_list.append(avg_val_dice)
            val_per_cls_dice_list.append(avg_per_cls_val_dice)
            val_bal_acc_list.append(val_bal_acc)
            val_auc_list.append(val_auc)
            iters_list.append(it)
            train_loss_sum = 0

            print(f"[Iter {it}], Train loss: {avg_train_loss}, Val dice: {avg_val_dice}, Val bal acc: {val_bal_acc}, Val AUC: {val_auc}")
            print(f"Val per class dice: {avg_per_cls_val_dice}")

            # save best model
            if (avg_val_dice + val_bal_acc + val_auc)/3 > best_val_metric:
                best_val_metric = (avg_val_dice + val_bal_acc + val_auc) / 3
                print(f"Saving best model with val metric: {best_val_metric} on iter: {it}")
                my_state_dict = seg_model.state_dict()
                to_save = {}
                for k, v in my_state_dict.items():
                    if 'lora_' in k:
                        to_save[k] = v
                    elif 'feature_model' in k:
                        if args.train_feature_model:
                            to_save[k] = v
                    else:
                        to_save[k] = v
                torch.save(to_save, args.output_dir + "/best_model.pth")

            # set back to train mode
            seg_model.train()
            if not args.train_feature_model and 'lora' not in args.config_file:
                seg_model.feature_model.eval()

        if it >= max_iter:
            break

    # test
    seg_model.load_state_dict(torch.load(args.output_dir + "/best_model.pth"), strict=False)
    seg_model.eval()

    total_test_dice = 0
    total_per_cls_test_dice = [0 for _ in range(num_classes)]
    test_probs = []
    test_preds = []
    test_labels = []
    test_steps = 0
    seg_model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            (test_dice, test_per_cls_dice), probs, preds, labels = val_iter(
                model=seg_model,
                batch=test_data,
                image_size=(args.image_size,) * 3,
                batch_size=args.batch_size,
                metric=dice_metric,
                overlap=0.75
            )

            total_test_dice += test_dice
            for i in range(num_classes):
                total_per_cls_test_dice[i] += test_per_cls_dice[i]
            test_probs.append(probs)
            test_preds.append(preds)
            test_labels.append(labels)
            test_steps += 1

    avg_test_dice = total_test_dice / test_steps
    avg_per_cls_test_dice = [total_per_cls_test_dice[i] / test_steps for i in range(num_classes)]

    test_probs = np.concatenate(test_probs, axis=0)
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_bal_acc = balanced_accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs[:, 1])

    print(f"Test dice: {avg_test_dice}, Test bal acc: {test_bal_acc}, Test AUC: {test_auc}")
    print(f"Test per class dice: {avg_per_cls_test_dice}")

    with open(f'{args.output_dir}/results.json', 'w') as fp:
        json.dump({
            'iters_list': iters_list,
            'train_loss_list': train_loss_list,
            'val_dice_list': val_dice_list,
            'val_per_cls_dice_list': val_per_cls_dice_list,
            'val_bal_acc_list': val_bal_acc_list,
            'val_auc_list': val_auc_list,
            'test_dice': avg_test_dice,
            'test_per_cls_dice': avg_per_cls_test_dice,
            'test_bal_acc': test_bal_acc,
            'test_auc': test_auc,
        }, fp)


def main(args):
    feature_model, autocast_dtype = setup_and_build_model_3d(args)
    do_finetune(feature_model, autocast_dtype, args)


if __name__ == "__main__":
    args = add_seg_args(get_args_parser(add_help=True)).parse_args()
    main(args)
