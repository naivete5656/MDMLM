from networks import VNet
from utils import *
from pathlib import Path
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from eval import eval_net
import copy
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
        "-s",
        "--save_weght_path",
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
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", help="batch_size", default=8, type=int
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", help="epochs", default=100, type=int
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        help="learning late",
        default=1e-3,
        type=float,
    )

    args = parser.parse_args()
    return args


class _TrainBase(VisShow):
    def __init__(self, **kwargs):
        self.save_weight_path = kwargs["save_weight_path"]
        self.epochs = kwargs["epochs"]
        self.net = kwargs["net"]
        self.gpu = kwargs["gpu"]
        self.need_vis = kwargs["vis"]
        self.batch_size = kwargs["batch_size"]
        ori_paths = self.gather_path(kwargs["train_paths"])
        data_loader = CellImageLoad3d(ori_paths, "train")
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=kwargs["batch_size"], shuffle=True, num_workers=4
        )
        self.number_of_traindata = data_loader.__len__()
        if kwargs["val_paths"] is not None:
            ori_paths = self.gather_path(kwargs["val_paths"])
            data_loader = CellImageLoad3d(ori_paths, "val")
            self.val_loader = torch.utils.data.DataLoader(
                data_loader,
                batch_size=kwargs["batch_size"],
                shuffle=False,
                num_workers=4,
            )
        else:
            self.val_loader = None

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters(), lr=kwargs["lr"])
        self.need_vis = kwargs["vis"]

        self.iteration = 1
        self.decay = 0.1

        # loss counters
        self.loc_loss = 0
        self.conf_loss = 0
        self.epoch_loss = 0
        self.bad = 0
        self.losses = []
        self.evals = []
        self.val_losses = []

    def gather_path(self, train_paths):
        ori_paths = []
        for train_path in train_paths:
            ori_paths.extend(train_path.glob("*.h5py"))
        return ori_paths

    def validation(self, number_of_train_data, epoch):
        loss = self.epoch_loss / (number_of_train_data + 1)
        print("Epoch finished ! Loss: {}".format(loss))
        if epoch % 10 == 0:
            torch.save(
                self.net.state_dict(),
                str(
                    self.save_weight_path.parent.joinpath(
                        "epoch_weight/{:05d}.pth".format(epoch)
                    )
                ),
            )
        val_loss = eval_net(
            self.net,
            self.val_loader,
            self.vis,
            self.img_view_val,
            gpu=self.gpu,
            only_loss=True,
        )

        print("val_loss: {}".format(val_loss))
        try:
            if min(self.val_losses) > val_loss:
                torch.save(self.net.state_dict(), str(self.save_weight_path))
                self.bad = 0
                print("update bad")
                with self.save_weight_path.parent.joinpath("best.txt").open("w") as f:
                    f.write(f"{epoch}")
                pass
            else:
                self.bad += 1
                print("bad ++")
        except ValueError:
            torch.save(self.net.state_dict(), str(self.save_weight_path))
        self.val_losses.append(val_loss)
        print("bad = {}".format(self.bad))
        self.epoch_loss = 0


class TrainNet(_TrainBase):
    def main(self):
        self.vis_init("test")
        for epoch in range(self.epochs):
            self.net.train()
            pbar = tqdm(total=self.number_of_traindata)
            for iteration, data in enumerate(self.train_dataset_loader):
                img = data["image"]
                target = data["gt"]
                if self.gpu:
                    img = img.cuda()
                    target = target.cuda()

                pred_img = self.net(img)

                loss = self.criterion(pred_img, target)

                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.iteration % 10000 == 0:
                    torch.save(
                        self.net.state_dict(),
                        str(
                            self.save_weight_path.parent.joinpath(
                                "epoch_weight/{:05d}.pth".format(epoch)
                            )
                        ),
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.decay * param_group["lr"]
                if self.need_vis:
                    self.update_vis_plot(
                        iteration, [loss.item()], self.iter_plot, None, "append"
                    )
                    self.update_vis_show(img[:, :, 8, :, :].cpu(), self.ori_view)
                    pred_img = pred_img - pred_img.min()
                    if (pred_img > 1).any():
                        pred_img = pred_img / pred_img.max()
                    self.update_vis_show(pred_img[:, :, 8, :, :], self.img_view)
                    self.update_vis_show(target[:, :, 8, :, :].cpu(), self.gt_view)
                if self.iteration >= 10000:
                    torch.save(
                        self.net.state_dict(),
                        str(self.save_weight_path.parent.joinpath("final.pth")),
                    )
                    print("stop running")
                    break
                pbar.update(self.batch_size)
            pbar.close()
            if self.iteration >= 10000:
                print("stop running")
                break
            if self.val_loader is not None:
                self.validation(iteration, epoch)
            else:
                torch.save(
                    self.net.state_dict(),
                    str(
                        self.save_weight_path.parent.joinpath("temp.pth".format(epoch))
                    ),
                )
                if epoch % 10 == 0:
                    torch.save(
                        self.net.state_dict(),
                        str(
                            self.save_weight_path.parent.joinpath(
                                "epoch_weight/{:05d}.pth".format(epoch)
                            )
                        ),
                    )

            if self.bad >= 50:
                print("stop running")
                break


class TrainExtractNet(TrainNet):
    def gather_path(self, train_paths):
        paths = []
        neg_paths = []
        for train_path in train_paths:
            temp_paths = []
            temp_neg_paths = []

            temp_paths.extend(
                sorted(train_path.joinpath("positive_sequence").glob("*.h5py"))
            )
            temp_neg_paths.extend(
                sorted(train_path.joinpath("negative_sequence").glob("*.h5py"))
            )
            sample_dif = len(temp_paths) - len(temp_neg_paths)

            if np.sign(sample_dif) == 1:
                temp_neg_paths.extend(random.sample(temp_neg_paths, abs(sample_dif)))
            else:
                if abs(sample_dif) > len(temp_paths):
                    temp_paths2 = copy.copy(temp_paths)
                    repeat_num = int(abs(sample_dif) / len(temp_paths))
                    for i in range(repeat_num):
                        temp_paths.extend(temp_paths2)
                    Remaining_num = abs(sample_dif) - len(temp_paths2) * repeat_num
                    temp_paths.extend(random.sample(temp_paths2, Remaining_num))
                else:
                    temp_paths.extend(random.sample(temp_paths, abs(sample_dif)))
            paths.extend(temp_paths)
            neg_paths.extend(temp_neg_paths)
        paths.extend(neg_paths)
        return paths


if __name__ == "__main__":
    args = parse_args()
    for condition in ["Control", "FGF2", "BMP2", "FGF2+BMP2"]:
        Groupes = {
            "Control": [1, 2, 3],
            "FGF2": [5, 6, 7],
            "BMP2": [9, 10, 11],
            "FGF2+BMP2": [13, 14, 15],
        }
        root_path = Path(args.root_path)

        seqs = Groupes[condition]

        train_paths = [
            root_path.joinpath(f"F{seqs[0]:04d}"),
            root_path.joinpath(f"F{seqs[1]:04d}"),
        ]

        val_paths = [
            root_path.parent.joinpath(f"val/F{seqs[0]:04d}"),
            root_path.parent.joinpath(f"val/F{seqs[1]:04d}"),
        ]

        save_weights_path = Path(f"./weights").joinpath(f"Group_{condition}/best.pth")
        save_weights_path.parent.joinpath("epoch_weight").mkdir(
            parents=True, exist_ok=True
        )
        save_weights_path.parent.mkdir(parents=True, exist_ok=True)

        net = VNet(elu=False, nll=False, sig=False)
        if args.gpu:
            net.cuda()

        args = {
            "gpu": args.gpu,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.learning_rate,
            "train_paths": train_paths,
            "val_paths": val_paths,
            "save_weight_path": save_weights_path,
            "net": net,
            "vis": args.vis,
        }
        train = TrainExtractNet(**args)
        train.main()
