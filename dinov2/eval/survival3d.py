# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

from dinov2.data.loaders import make_classification_dataset_3d
from dinov2.data import SamplerType, make_data_loader
from dinov2.eval.setup import get_args_parser, setup_and_build_model_3d
from dinov2.data.transforms import make_classification_transform_3d

import torch
import json
import os
from functools import partial
from collections import Counter
from monai.data.utils import list_data_collate
from monai.optimizers import WarmupCosineSchedule
from monai.utils.misc import set_determinism
from sksurv.metrics import concordance_index_censored
import loralib as lora
import numpy as np

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
set_determinism(seed=0)


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return self.nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return self.nll_loss(hazards, S, Y, c, alpha=alpha)

    @staticmethod
    def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
        batch_size = len(Y)
        Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
        c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
        if S is None:
            S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
        # without padding, S(0) = S[0], h(0) = h[0]
        S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
        # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
        # h[y] = h(1)
        # S[1] = S(1)
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1-alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss


class LinearHead(torch.nn.Module):
    def __init__(self, feature_model, num_classes, autocast_ctx, n_last_layers=4, cat_tokens=False):
        super().__init__()

        self.autocast_ctx = autocast_ctx

        self.feature_model = feature_model
        self.n_last_layers = n_last_layers
        self.cat_tokens = cat_tokens

        self.hidden_size = self.feature_model.num_features

        self.act_fn = torch.nn.GELU()
        size = self.hidden_size * self.n_last_layers + int(self.cat_tokens) * self.hidden_size
        self.linear1 = torch.nn.Linear(size, self.hidden_size//2)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden_size//2)
        self.linear2 = torch.nn.Linear(self.hidden_size//2, num_classes)

    def forward(self, inputs):
        """Forward function."""
        with self.autocast_ctx():
            features = self.feature_model.get_intermediate_layers(
                inputs,
                n=self.n_last_layers,
                return_class_token=True,
                reshape=False
            )
        cls_tokens = [f[1] for f in features]
        output = torch.cat(cls_tokens, dim=1)
        if self.cat_tokens:
            patch_tokens = torch.mean(features[-1][0], dim=1)
            output = torch.cat([output, patch_tokens], dim=1)
        
        output = self.linear1(output)
        output = self.act_fn(output)
        output = self.bn1(output)
        output = self.linear2(output)

        Y_hat = torch.topk(output, 1, dim=1)[1]
        hazards = torch.sigmoid(output)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat


def add_cls_args(parser):
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
        "--n-last-layers",
        type=int,
        help="Number of last layers to use for classification head",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        help="Number of bins for survival analysis",
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


def train_iter(model, batch, censored, bin_edges, optimizer, scheduler, loss_function, scaler):
    x, y = (batch["image"].cuda(), batch["label"])
    y_binned = (torch.bucketize(y, bin_edges[:-1], right=True) - 1).cuda()  # Convert to bin indices
    c = censored.cuda()
    hazards, S, Y_hat = model(x)
    loss = loss_function(hazards=hazards, S=S, Y=y_binned, c=c)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return loss.item()


def do_finetune(feature_model, autocast_dtype, args):

    # get transforms, dataset, dataloaders
    train_transforms, val_transforms = make_classification_transform_3d(
        args.dataset_name,
        args.image_size,
        min_int=-1.0
    )
    train_ds, val_ds, test_ds, _ = make_classification_dataset_3d(
        args.dataset_name,
        args.dataset_percent,
        args.base_data_dir,
        train_transforms,
        val_transforms,
        args.cache_dir,
        dataset_seed=0
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
        batch_size=args.batch_size,
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        persistent_workers=False,
        collate_fn=list_data_collate
    )

    # use dataset to get bin edges to bin survival times, print label counts for train and val datasets
    with open(os.path.join(args.base_data_dir, 'MRI_UPenn_GBM.json'), 'r') as json_f:
        metadata = json.load(json_f)
    times = torch.tensor([metadata['training'][i]['label'] for i in range(len(metadata['training']))]).float()
    bin_edges = torch.quantile(times, torch.linspace(0, 1, args.n_bins + 1))
    bin_edges[0] = times.min() - 1e-6
    bin_edges[-1] = times.max() + 1e-6
    train_labs = (torch.bucketize(times, bin_edges[:-1], right=True) - 1)
    val_times = torch.tensor([metadata['validation'][i]['label'] for i in range(len(metadata['validation']))]).float()
    val_labs = (torch.bucketize(val_times, bin_edges[:-1], right=True) - 1)
    print(f"Train label counts: {Counter(train_labs.tolist())}")
    print(f"Validation label counts: {Counter(val_labs.tolist())}")

    # get model
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    scaler = torch.cuda.amp.GradScaler()
    # classification head
    cls_model = LinearHead(feature_model, args.n_bins, autocast_ctx, n_last_layers=args.n_last_layers, cat_tokens=False)

    if args.train_feature_model:
        cls_model.feature_model.train()
    else:
        if 'lora' not in args.config_file:
            cls_model.feature_model.eval()
            for param in cls_model.feature_model.parameters():
                param.requires_grad = False
        else:
            cls_model.train()
            lora.mark_only_lora_as_trainable(cls_model.feature_model)

    trainable_params = [name for name, param in cls_model.named_parameters() if param.requires_grad]
    print(f"Trainable parameters: {trainable_params}")

    # get optimizer, scheduler, loss function, metric
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, cls_model.parameters()), lr=args.learning_rate, weight_decay=0.01)
    max_iter = args.epochs * args.epoch_length
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=args.warmup_iters,
        t_total=max_iter
    )

    # loss_fn should just be NLLSurv loss for surv prediction
    loss_fn = NLLSurvLoss(alpha=0)
    cls_model.cuda()

    best_val_metric = -1
    train_loss_sum = 0
    iters_list = []
    train_loss_list = []
    val_c_index_list = []

    for it, train_data in enumerate(train_loader):

        # create censored (all zeros, as no censored cases)
        censored = torch.zeros(train_data["label"].shape)

        # train for one iteration
        train_loss = train_iter(
            model=cls_model,
            batch=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_fn,
            scaler=scaler,
            bin_edges=bin_edges,
            censored=censored,
        )
        train_loss_sum += train_loss

        if it % args.eval_iters == 0:
            # valdation
            val_risks = []
            val_times = []
            val_events = []
            val_loss = 0
            val_steps = 0
            cls_model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    x, y = (val_data["image"].cuda(), val_data["label"])
                    y_binned = (torch.bucketize(y, bin_edges[:-1], right=True) - 1).cuda()  # Convert to bin indices
                    censored = torch.zeros(val_data["label"].shape)
                    c = censored.cuda()
                    hazards, S, Y_hat = cls_model(x)
                    loss = loss_fn(hazards=hazards, S=S, Y=y_binned, c=c)
                    risk = -torch.sum(S, dim=1)

                    val_risks.append(risk.cpu().numpy())
                    val_times.append(y.cpu().numpy())
                    val_events.append(1 - censored.cpu().numpy())  # event is opposite of censored
                    val_loss += loss.item()
                    val_steps += 1

            # get metrics
            val_risks = np.concatenate(val_risks, axis=0)
            val_times = np.concatenate(val_times, axis=0)
            val_events = np.concatenate(val_events, axis=0)
            val_c_index = concordance_index_censored(val_events.astype(bool), val_times, val_risks)[0]
            avg_train_loss = train_loss_sum / args.eval_iters
            val_c_index_list.append(val_c_index)
            train_loss_list.append(avg_train_loss)

            iters_list.append(it)
            train_loss_sum = 0

            print(f"[Iter {it}], Train loss: {avg_train_loss}, Val C-Index: {val_c_index}, Val loss: {val_loss / val_steps}", flush=True)

            # save best model
            if val_c_index > best_val_metric:
                best_val_metric = val_c_index
                print(f"Saving best model with val avg metric: {best_val_metric} on iter: {it}")
                # only save trainable params (cls head or lora params)
                my_state_dict = cls_model.state_dict()
                to_save = {}
                for k, v in my_state_dict.items():
                    if 'lora_' in k:
                        to_save[k] = v
                    elif 'feature_model' in k:
                        if args.train_feature_model:
                            to_save[k] = v
                        else:
                            # only save feature model if training it
                            continue
                    else:
                        to_save[k] = v
                torch.save(to_save, args.output_dir + "/best_model.pth")

            # set back to train mode
            cls_model.train()
            if not args.train_feature_model and 'lora' not in args.config_file:
                cls_model.feature_model.eval()

        if it >= max_iter:
            break

    # test
    cls_model.load_state_dict(torch.load(args.output_dir + "/best_model.pth"), strict=False)
    cls_model.eval()
    test_risks = []
    test_times = []
    test_events = []
    test_loss = 0
    test_steps = 0
    with torch.no_grad():
        for test_data in test_loader:
            x, y = (test_data["image"].cuda(), test_data["label"])
            y_binned = (torch.bucketize(y, bin_edges[:-1], right=True) - 1).cuda()  # Convert to bin indices
            censored = torch.zeros(test_data["label"].shape)
            c = censored.cuda()
            hazards, S, Y_hat = cls_model(x)
            loss = loss_fn(hazards=hazards, S=S, Y=y_binned, c=c)
            risk = -torch.sum(S, dim=1)

            test_risks.append(risk.cpu().numpy())
            test_times.append(y.cpu().numpy())
            test_events.append(1 - censored.cpu().numpy())  # event is opposite of censored
            test_loss += loss.item()
            test_steps += 1

    test_risks = np.concatenate(test_risks, axis=0)
    test_times = np.concatenate(test_times, axis=0)
    test_events = np.concatenate(test_events, axis=0)

    test_c_index = concordance_index_censored(test_events.astype(bool), test_times, test_risks)[0]

    print(f"Test C-Index: {test_c_index}, Test loss: {test_loss / test_steps}")

    with open(f'{args.output_dir}/results.json', 'w') as fp:
        json.dump({
            'iters_list': iters_list,
            'train_loss_list': train_loss_list,
            'val_c_index_list': val_c_index_list,
            'best_val_metric': best_val_metric,
            'test_c_index': test_c_index,
        }, fp)


def main(args):
    feature_model, autocast_dtype = setup_and_build_model_3d(args)
    do_finetune(feature_model, autocast_dtype, args)


if __name__ == "__main__":
    args = add_cls_args(get_args_parser(add_help=True)).parse_args()
    main(args)
