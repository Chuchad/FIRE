import torch
import os
import torch.optim
from tap import Tap
from torch.utils.data import DataLoader

from utils.network_utils import get_resnet_model, FIRE_model
from dataset import InversionDataset

import torch.nn.functional as F

from tqdm import tqdm
from torchmetrics.classification.accuracy import BinaryAccuracy
from torchmetrics.aggregation import MeanMetric
import wandb

from typing import Literal
class TrainArgs(Tap):

    real_train_dir: str
    fake_train_dir: str

    real_val_dir: str = None
    fake_val_dir: str = None

    batch_size: int = 24

    total_epochs: int = 100
    earlystop_epoch: int = 10

    beta1: float = 0.9
    lr: float = 0.0001
    min_lr: float = 1e-6

    loss_freq: int = 400

    loader_workers: int = 16

    topk: int = 3

    save_dir: str = None

    mode: Literal["ours", "rgb", "frq", "fire"] = "ours"

    norm_layer: Literal["batch", "instance"] = "instance"

    pretrained: bool = True

    resize: bool = False

    resume: str = None
 
    weak_augment: bool = False

    strong_augment: bool = False

    job_type: str = None


# keep top k best ckpt
def sort_top_k(top_k_list, candidate, k=3): 
    if len(top_k_list) == 0:
        top_k_list.append(candidate)
        return top_k_list, []
    else:
        if candidate[0] >= top_k_list[-1][0] and len(top_k_list) == k:
            return top_k_list, [candidate]
        to_be_del = []
        while len(top_k_list) > 0 and top_k_list[-1][0] > candidate[0]:
            to_be_del.append(top_k_list.pop())
        top_k_list.append(candidate)
        while len(to_be_del) > 0 and len(top_k_list) < k:
            top_k_list.append(to_be_del.pop())
        return top_k_list, to_be_del


def train_model(args: TrainArgs):
    os.makedirs(args.save_dir, exist_ok=True)

    print("Creating Datasets...")
    train_ds = InversionDataset(real_dir=args.real_train_dir, fake_dir=args.fake_train_dir,\
                                mode=args.mode, resize=args.resize, \
                                do_weak_aug=args.weak_augment, do_strong_aug=args.strong_augment)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_workers)

    print("Creating Model...")
    if args.mode == "fire":
        model = FIRE_model()
    else:
        model = get_resnet_model(mode=args.mode, norm_layer=args.norm_layer, pretrained=args.pretrained)
    if args.resume is not None:
        try:
            model.load_state_dict(torch.load(args.resume))
            model = model.cuda()
        except:
            state_dict = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(state_dict)
            model.cuda()

    model.cuda()
    opt = torch.optim.Adam(model.parameters(),lr=args.lr, betas=(args.beta1, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, threshold=0.002)


    print("Begin Training")
    # best_val_loss = float("inf")
    last_loss = float("inf")
    step = 0
    patience = 0
    top_k = []
    for e in tqdm(range(args.total_epochs), desc="Training Epochs..."):
        # model.train()

        train_acc = BinaryAccuracy().cuda()
        train_loss = MeanMetric().cuda()
        train_loss_b = MeanMetric().cuda()
        train_loss_mse_rec = MeanMetric().cuda()
        train_loss_mse_mask = MeanMetric().cuda()
        train_loss_mse_mask_mid_frq = MeanMetric().cuda()
        train_loss_mse_mask_mid_filtered = MeanMetric().cuda()
        train_loss_mse_mask_norm = MeanMetric().cuda()

        # train loop
        pbar = tqdm(train_loader, desc="train batches...", leave=False)
        for x, y in pbar:
            # pbar.set_description()
            x = x.cuda()
            y = y.cuda()

            out, middle_freq_image, raw_reconstructions_delta, mask_mid_frq, mask_mid_filterd = model(x)

            loss_mse_rec = F.mse_loss(middle_freq_image, raw_reconstructions_delta)

            i_mask = model.fft_filter_module.i_mask.unsqueeze(0).repeat(x.shape[0], 1, 1, 1).detach()

            r_i_mask = model.fft_filter_module.r_i_mask.unsqueeze(0).repeat(x.shape[0], 1, 1, 1).detach()

            all_mask = torch.ones_like(model.fft_filter_module.r_i_mask.unsqueeze(0)).repeat(x.shape[0], 1, 1, 1).detach()

            loss_mse_mask_mid_frq = F.mse_loss(mask_mid_frq, i_mask)

            loss_mse_mask_mid_filterd = F.mse_loss(mask_mid_filterd, r_i_mask)

            loss_mse_mask_norm = F.mse_loss(mask_mid_frq+mask_mid_filterd, all_mask)

            loss_mse_mask = loss_mse_mask_mid_frq + loss_mse_mask_mid_filterd + loss_mse_mask_norm

            loss_b = F.binary_cross_entropy_with_logits(out[:, 0], y.float())
            
            loss = 0.6 * loss_b + 0.2 * loss_mse_rec + 0.2 * loss_mse_mask
            opt.zero_grad()
            loss.backward()
            opt.step()
            print("loss: ", loss.item())
            train_acc.update(F.sigmoid(out)[:, 0], y)
            train_loss.update(loss)
            train_loss_b.update(loss_b)
            train_loss_mse_rec.update(loss_mse_rec)
            train_loss_mse_mask.update(loss_mse_mask)
            train_loss_mse_mask_mid_frq.update(loss_mse_mask_mid_frq)
            train_loss_mse_mask_mid_filtered.update(loss_mse_mask_mid_filterd)
            train_loss_mse_mask_norm.update(loss_mse_mask_norm)

            # log loss
            if step % args.loss_freq == 0:
                wandb.log({"loss/batch": loss.item()}, step=step)
                wandb.log({"loss_mse_rec/batch": loss_mse_rec.item()}, step=step)
                wandb.log({"loss_mse_mask/batch": loss_mse_mask.item()}, step=step)
                wandb.log({"loss_b/batch": loss_b.item()}, step=step)
                wandb.log({"loss_mse_mask_mid_frq/batch": loss_mse_mask_mid_frq.item()}, step=step)
                wandb.log({"loss_mse_mask_mid_filtered/batch": loss_mse_mask_mid_filterd.item()}, step=step)
                wandb.log({"loss_mse_mask_norm/batch": loss_mse_mask_norm.item()}, step=step)

            step += 1
        # log epoch loss
        wandb.log({
            "loss/train": train_loss.compute().item(),
            "acc/train": train_acc.compute().item(),
            "loss_b/train": train_loss_b.compute().item(),
            "loss_mse_rec/train": train_loss_mse_rec.compute().item(),
            "loss_mse_mask/train": train_loss_mse_mask.compute().item(),
            "loss_mse_mask_mid_frq/train": train_loss_mse_mask_mid_frq.compute().item(),
            "loss_mse_mask_mid_filtered/train": train_loss_mse_mask_mid_filtered.compute().item(),
            "loss_mse_mask_norm/train": train_loss_mse_mask_norm.compute().item(),
        }, step=step)

        sched.step(train_loss.compute().item())

        if e % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_{:04d}.pt".format(e)))

        cur_loss = train_loss.compute().item()
        # save top k best ckpt
        _, del_list = sort_top_k(top_k, (cur_loss, f"top_e{e:03d}_{cur_loss:.3f}.pt"), k=args.topk)
        if del_list.__len__() == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"top_e{e:03d}_{cur_loss:.3f}.pt"))
        else:
            if del_list[0][1] == f"top_{e:03d}_{cur_loss:.3f}.pt":
                continue
            else:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"top_e{e:03d}_{cur_loss:.3f}.pt"))
                for item in del_list:
                    os.remove(os.path.join(args.save_dir, item[1]))
        
        if last_loss < train_loss.compute().item():
            patience += 1
            if patience >= args.earlystop_epoch:
                print("Early Stopping...")
                break
        else:
            patience = 0
        
        last_loss = cur_loss
        
        if opt.param_groups[0]["lr"] < args.min_lr:
            print("Minimum LR Reached...")
            break


if __name__ == "__main__":
    args = TrainArgs(explicit_bool=True).parse_args()
    wandb.init(
        project="FIRE",
        config=args.as_dict(),
        job_type=args.job_type,
        entity="",
    )
    if args.save_dir is None:
        args.save_dir = os.path.join("models", wandb.run.name)
    train_model(args)