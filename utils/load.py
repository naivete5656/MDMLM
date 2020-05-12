from PIL import Image
import matplotlib.pyplot as plt
from .custom_transform import *
from torch.utils.data import Dataset
from pathlib import Path

from scipy.ndimage.interpolation import rotate
import h5py


def rand_flipud(img, gt):
    rand_value = np.random.randint(0, 4)
    if rand_value == 1:
        img = np.flipud(img)
        gt = np.flipud(gt)
    elif rand_value == 2:
        img = np.fliplr(img)
        gt = np.fliplr(gt)
    elif rand_value == 3:
        img = np.flipud(img)
        gt = np.flipud(gt)
        img = np.fliplr(img)
        gt = np.fliplr(gt)
    return img, gt


class OriImageLoad(Dataset):
    def __init__(self, ori_path):
        self.ori_paths = ori_path
        self.trans = None

    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        img = Image.open(img_name)
        img = self.trans(img) / 4095
        img2_name = img_name.parent.joinpath(
            f"{img_name.name[:-9]}{int(img_name.name[-9:-4]) + 1:05d}.tif"
        )
        img2 = Image.open(img2_name)
        img2 = self.trans(img2) / 4095

        img = torch.cat([img, img2], dim=0)
        return img


class OriImageLoad3d:
    def __init__(self, ori_path, transform):
        self.ori_paths = ori_path
        self.trans = None

    def __len__(self):
        return len(self.ori_paths)

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        with h5py.File(str(img_name), "r") as f:
            img = f["img"].value
        img = torch.from_numpy(img.astype(np.float32))

        datas = {"image": img.unsqueeze(0)}

        return datas


class CellImageLoad3d(OriImageLoad):
    def __init__(self, ori_path, transform):
        self.ori_paths = ori_path
        self.trans = None

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        with h5py.File(str(img_name), "r") as f:
            img = f["img"].value
            gt = f["target"].value

        # data augumentation
        rand_value = np.random.randint(0, 4)
        img = rotate(img, 90 * rand_value, axes=(2, 1))
        gt = rotate(gt, 90 * rand_value, axes=(2, 1))
        img, gt = rand_flipud(img, gt)

        img = torch.from_numpy(img.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0)}

        return datas


class CellImageLoad3dTest(OriImageLoad):
    def __init__(self, ori_path, transform):
        self.ori_paths = ori_path
        if transform == "train":
            self.trans = tra_trans()
        elif transform == "val":
            self.trans = val_trans()
        self.trans = None

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        with h5py.File(str(img_name), "r") as f:
            img = f["img"].value
            tra = f["tra"].value
            tra = tra / tra.max()
            gt = f["target"].value
        # if self.trans is not None:
        # img, gt = self.trans([img, gt])
        img = torch.from_numpy(img.astype(np.float32))
        tra = torch.from_numpy(tra.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))

        # datas = {"image": img.unsqueeze(0)}
        # datas = {"image": img.unsqueeze(0), "gt": gt.unsqueeze(0)}
        datas = {"image": img.unsqueeze(0), "tra": tra, "gt": gt.unsqueeze(0)}

        return datas


class CNN3dLoader(CellImageLoad3d):
    def __init__(self, ori_path, transform):
        self.ori_paths = ori_path
        self.trans = None

    def __getitem__(self, data_id):
        img_name = self.ori_paths[data_id]
        with h5py.File(str(img_name), "r") as f:
            img = f["img"].value
            gt = f["target"].value
        img = img[:, 32:96, 32:96]
        # img = img[:, 40:88, 40:88]
        # gt = gt[:, 40:88, 40:88]
        gt = gt[:, 32:96, 32:96]

        # data augumentation
        rand_value = np.random.randint(0, 4)
        img = rotate(img, 90 * rand_value, axes=(2, 1))
        img, gt = rand_flipud(img, gt)
        # frame, _, length, _, _ = img_name.stem.split('-')
        # frame = int(frame)
        # length = int(length)
        # first_frame = max(round(length / 2 - 2), 0)

        # if gt[first_frame:first_frame + 4, :, :].max() > 0.99:
        #     gt = torch.tensor([1], dtype=torch.float32)
        # else:
        #     gt = torch.tensor([0], dtype=torch.float32)
        gt = gt.max((1, 2))
        gt[gt < 0.9] = 0
        gt[gt > 0.9] = 1

        img = torch.from_numpy(img.astype(np.float32))

        datas = {"image": img.unsqueeze(0), "gt": gt}

        return datas


if __name__ == "__main__":
    transform = Compose(
        [
            RandomCrop(270),
            RandomRotation(5),
            CenterCrop(256),
            # Normalize(125, 40),
            ToTensor(),
        ]
    )
    ori_paths = sorted(
        Path(
            "/home/kazuya/ssd/CVPR_workshop/images/circle_t1_radis8/F0001/positive_sequence"
        ).glob("*.h5py")
    )
    data_loader = CellImageLoad3d(ori_paths, "train")
    length = data_loader.__len__()
    for data in data_loader:
        img = data["image"]
        gt = data["gt"]
        pos_seqs = []
        for i, g in enumerate(gt[0]):
            if g.max().numpy() > 0.5:
                pos_seqs.append(i)
        for i in pos_seqs:
            plt.imshow(img[0, i]), plt.show()
            plt.imshow(gt[0, i]), plt.show()
    # print(data_loader[0])
    dataset_loader = torch.utils.data.DataLoader(
        data_loader, batch_size=1, shuffle=False, num_workers=4
    )
    for data in dataset_loader:
        print(data)
