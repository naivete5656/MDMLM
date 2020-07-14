from pathlib import Path
import torch
import h5py
from utils import VisShow
import numpy as np
from networks import VNet
from for_evaluation import (
    candidate_to_whole,
    save,
    visualize,
    evaluation,
    ablation,
    eval_ab,
)
import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="data path")
    parser.add_argument(
        "-r",
        "--root_path",
        dest="root_path",
        help="root path",
        default="./images/train",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weght_path",
        dest="save_path",
        help="save path",
        default="./weights",
        type=str,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        dest="gpu",
        help="whether use CUDA",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--vis",
        dest="vis",
        help="whether use visdom",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    return args


def create_vis_show(self):
    return self.vis.images(torch.ones((self.batch_size, 1, 256, 256)), self.batch_size)


def update_vis_show(self, images, window1):
    self.vis.images(images, self.batch_size, win=window1)


class Predict(VisShow):
    def __init__(self, **kwargs):
        self.net = kwargs["net"]
        self.ori_paths = self.gather_path(kwargs["data_paths"])

        self.net = kwargs["net"]
        self.gpu = kwargs["gpu"]
        self.save_path_root = kwargs["save_path"]
        self.need_vis = kwargs["vis"]
        self.batch_size = 16

    def gather_path(self, train_path):
        ori_paths = []
        for train_dir in train_path:
            ori_paths.extend(sorted(train_dir.glob("*.h5py")))
        return ori_paths

    def __call__(self, *args, **kwargs):
        self.net.eval()
        for iteration, ori_path in enumerate(self.ori_paths):
            img_name = ori_path
            with h5py.File(str(img_name), "r") as f:
                img = f["img"].value
                # gt = f["target"].value

            img = torch.from_numpy(img.astype(np.float32))

            if self.gpu:
                img = img.cuda()

            pred_img = net(img.unsqueeze(0).unsqueeze(0))
            pred_img = pred_img - pred_img.min()
            if (pred_img > 1).sum():
                pred_img = pred_img / pred_img.max()

            pred_img = pred_img.detach().cpu().numpy()

            save_path = self.save_path_root.joinpath(ori_path.name)

            outfh = h5py.File(str(save_path), "w")
            outfh.create_dataset("img", data=img.cpu().numpy())
            # outfh.create_dataset("gt", data=gt)
            outfh.create_dataset("pred", data=pred_img)
            outfh.flush()
            outfh.close()


if __name__ == "__main__":
    Groupes = {
        "Control": [1, 2, 3],
        "FGF2": [5, 6, 7],
        "BMP2": [9, 10, 11],
        "FGF2+BMP2": [13, 14, 15],
    }

    args = parse_args()
    root_path = Path(args.root_path)
    for condition in ["Control", "FGF2", "BMP2", "FGF2+BMP2"]:
        data_paths = [root_path.joinpath(f"F{Groupes[condition][2]:04d}/test")]
        weight_path = Path(args.weight_path).joinpath(f"Group_{condition}/best.pth")

        save_path = Path(args.save_path).joinpath(f"Group_{condition}")
        save_path.mkdir(parents=True, exist_ok=True)

        net = VNet(elu=False, nll=False, sig=False)
        net.cuda()
        net.load_state_dict(torch.load(weight_path, map_location="cuda:1"))

        args = {
            "net": net,
            "gpu": args.gpu,
            "data_paths": data_paths,
            "save_path": save_path,
            "vis": args.vis,
        }

        pre = Predict(**args)
        pre()
