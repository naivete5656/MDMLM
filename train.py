from networks import VNet
from utils import *
from pathlib import Path
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from eval import eval_net
import copy


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

        self.criterion = kwargs["criterion"]
        if "criterion2" in kwargs.keys():
            self.criterion2 = kwargs["criterion2"]
        else:
            self.criterion2 = None
        if "criterion3" in kwargs.keys():
            self.criterion3 = kwargs["criterion3"]
        else:
            self.criterion3 = None

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

                loss1 = self.criterion(pred_img, target)
                loss = loss1
                if self.criterion2 is not None:
                    loss2 = self.criterion2(target - pred_img)
                    loss = loss + loss2

                if self.criterion3 is not None:
                    loss3 = self.criterion3(pred_img, target)
                    loss = loss + loss3

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
                        iteration, [loss1.item()], self.iter_plot, None, "append"
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
    torch.cuda.set_device(1)
    plot_size = 9
    mode = "train"

    for num in [1, 2, 3, 4]:
        Groupes = {1: [1, 2, 3], 2: [5, 6, 7], 3: [9, 10, 11], 4: [13, 14, 15]}
        root_path = Path(f"./images/{mode}")

        seqs = Groupes[num]

        train_paths = [
            root_path.joinpath(f"F{seqs[0]:04d}"),
            root_path.joinpath(f"F{seqs[1]:04d}"),
        ]

        val_paths = [
            root_path.parent.joinpath(f"{mode}_val/F{seqs[0]:04d}"),
            root_path.parent.joinpath(f"{mode}_val/F{seqs[1]:04d}"),
        ]

        save_weights_path = Path(f"./weights/{mode}/Group_{num}/best.pth")
        save_weights_path.parent.joinpath("epoch_weight").mkdir(
            parents=True, exist_ok=True
        )
        save_weights_path.parent.mkdir(parents=True, exist_ok=True)

        net = VNet(elu=False, nll=False, sig=False)
        net.cuda()

        args = {
            "gpu": True,
            "batch_size": 8,
            "epochs": 100,
            "lr": 1e-3,
            "train_paths": train_paths,
            "val_paths": val_paths,
            "save_weight_path": save_weights_path,
            "net": net,
            "vis": False,
            "plot_size": plot_size,
            "criterion": nn.MSELoss(),
        }
        train = TrainExtractNet(**args)
        train.main()
