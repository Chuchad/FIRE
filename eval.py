import os, torch, json
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, det_curve, precision_recall_curve, roc_curve, accuracy_score
from dataset import InversionDataset
from utils.network_utils import get_resnet_model, get_frq_resnet_model, FIRE_model
from tap import Tap
from typing import Literal


class EvalArgs(Tap):

    real_test_dir: str
    fake_test_dir: str

    ckpt: str

    batch_size: int = 12

    loader_workers: int = 16

    mode: Literal["rgb", "ours", "frq", "fire"] = "ours"

    norm_layer: Literal["batch", "instance"] = "batch"

    resize: bool = False


def eval_model(args: EvalArgs):
    if args.mode == "frq":
        model = get_frq_resnet_model(mode=args.mode, norm_layer=args.norm_layer, pretrained=False)
    elif args.mode == "fire":
        model = FIRE_model()
    else:
        model = get_resnet_model(mode=args.mode, norm_layer=args.norm_layer, pretrained=False)

    try:
        model.load_state_dict(torch.load(args.ckpt))
        model = model.cuda()
    except:
        state_dict = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state_dict)
        model.cuda()


    model.eval()

    all_stats = {}

    # testing on corresponding subsets
    if args.real_test_dir == "test/imagenet/":
        EVAL_SETS = ["adm", "sdv1"]
    elif args.real_test_dir == "test/lsun_bedroom/":
        EVAL_SETS = ["adm", "dalle2", "ddpm", "diff-projectedgan", "diff-stylegan", "iddpm", "if", "ldm", "midjourney", "pndm", "projectedgan", "sdv2", "stylegan_official", "vqdiffusion"]
    else:
        EVAL_SETS = ["dalle3", "kandinsky3", "midjourney", "sdxl", "vega"]
    for eval_set in EVAL_SETS:
        if "0_real" in args.real_test_dir:
            real_path = os.path.join("test", eval_set, args.real_test_dir)
            fake_path = os.path.join("test", eval_set, args.fake_test_dir)
        else:
            real_path = os.path.join(args.real_test_dir, "real")
            fake_path = os.path.join(args.fake_test_dir, eval_set)
        

        ds = InversionDataset(real_path, fake_path, mode=args.mode, resize=args.resize, do_strong_aug=False, do_weak_aug=False)

        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=16)

        preds = []
        truth = []

        for im, lbl in tqdm(loader):
            # out, middle_freq_image, raw_reconstructions_delta, mask_mid_frq, mask_mid_filterd = model(im.cuda())
            out = model(im.cuda())[0]
            out = out.sigmoid()
            preds += [o.item() for o in out]
            truth += [l.item() for l in lbl]

        stats = {}

        # log all the metrics
        stats["preds"] = preds
        stats["truth"] = truth

        stats["AUROC"] = roc_auc_score(truth, preds)
        stats["AP"] = average_precision_score(truth, preds)

        fpr, fnr, thresholds = det_curve(truth, preds)
        stats["DET"] = {}
        stats["DET"]["fpr"] = fpr.tolist()
        stats["DET"]["fnr"] = fnr.tolist()
        stats["DET"]["thresholds"] = thresholds.tolist()

        p, r, t = precision_recall_curve(truth, preds)
        stats["PRC"] = {}
        stats["PRC"]["precision"] = p.tolist()
        stats["PRC"]["recall"] = r.tolist()
        stats["PRC"]["thresholds"] = t.tolist()

        fpr, tpr, thresholds = roc_curve(truth, preds)
        stats["ROC"] = {}
        stats["ROC"]["fpr"] = fpr.tolist()
        stats["ROC"]["tpr"] = tpr.tolist()
        stats["ROC"]["thresholds"] = thresholds.tolist()

        all_stats[eval_set] = stats
        acc = accuracy_score(truth, [1 if p > 0.5 else 0 for p in preds])
        print(f"acc is {acc}")


    with open(os.path.join(os.path.dirname(args.ckpt), "stats.json".format()), "w") as f:
        json.dump(all_stats, f)

    for k in all_stats.keys():
        print(all_stats[k]["AUROC"])



if __name__ == "__main__":
    args = EvalArgs(explicit_bool=True).parse_args()
    eval_model(args)