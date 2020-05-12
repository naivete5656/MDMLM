from pathlib import Path
from scipy import io
import numpy as np
import cv2
import h5py


def gaussian(T, Y, X, t, y, x, sigma, sigma_t=1):
    """
    the code is generate gaussian distribution.
    T, Y, X is mesh grid array.
    where t, y, x is center plot of gaussian.
    :param meshs:
    :param coords:
    :param sigma:
    :param sigma_t:
    :return:
    """
    const_value = np.sqrt(2 * np.pi * sigma) ** 3
    norm = np.exp(
        -(
                ((X - x) ** 2) / (2 * sigma ** 2)
                + ((Y - y) ** 2) / (2 * sigma ** 2)
                + ((T - t) ** 2) / (2 * sigma_t ** 2)
        )
    )
    return norm / const_value


class ImageTargetGenerator(object):
    def __init__(self, sequence=1, depth=16, plot_size=9, sigma_t=1, pad=True, pd_mode="constant",
                 mode="cand_likelihood", ):
        super().__init__()
        self.sequence = sequence
        self.depth = depth
        self.plot_size = plot_size
        self.sigma_t = sigma_t
        self.sigma_t = 1
        self.pad = pad
        self.mode = mode
        if self.pad:
            self.cand_mode = "new_pad_mitosis"
        else:
            self.cand_mode = "output"
        self.cand_path = Path(f"/home/kazuya/{self.cand_mode}/F{self.sequence:04d}/new_candidate2.mat")
        self.img_path_root = Path(f"../CVPR_workshop/F{self.sequence:04d}")
        self.save_path_root = Path(
            f"/home/kazuya/hdd/CVPR_workshop/images/{self.mode}/F{self.sequence:04d}"
        )
        self.pd_mode = pd_mode
        self.mode = mode
        self.cands = io.loadmat(str(self.cand_path))["candi"]
        try:
            self.annotations = Path(
                f"../MitosisAnnotations/090303_F{sequence:04d}.mat"
            )
            self.annotations = io.loadmat(str(self.annotations))["result"]
            if pad:
                self.annotations[:, 1] = self.annotations[:, 1] + 30
                self.annotations[:, 2] = self.annotations[:, 2] + 30
        except FileNotFoundError:
            self.annotations = None
        self.lim_size = np.array([1040, 1392])
        if self.pad:
            self.lim_size = self.lim_size + 60

    def generate_cand(self):
        for cand_id, cand in enumerate(self.cands):
            length = cand[0]
            frame = cand[6]
            center_min = (cand[1:3] + cand[3:5]) / 2 - 64
            center_min = center_min[::-1]
            center_min[0] = center_min[0].clip(0, self.lim_size[0] - 128)
            center_min[1] = center_min[1].clip(0, self.lim_size[1] - 128)
            center_min = center_min.astype(np.int)
            center_max = center_min + 128

            self.call_data_generater(length, frame, center_min, center_max, cand_id)

    def call_data_generater(self, length, frame, center_min, center_max, cand_id):
        if frame != 1:
            if frame != 2:
                frame = frame - 2
                length = length + 4
            else:
                frame = frame - 1
                length = length + 3
        else:
            length = length + 2

        for_calcurate_max_frame = length / 16 + 1
        if (frame + for_calcurate_max_frame * 16) > 1013:
            frame = 1013 - for_calcurate_max_frame * 16

        if length < self.depth:
            self.train_data(
                length,
                frame,
                center_min,
                center_max,
                cand_id,
            )
        else:
            if length < 256:
                for add_frame in range(0, int(length), int(self.depth / 2)):
                    if (length - add_frame) > self.depth:
                        len = self.depth
                    else:
                        len = length - add_frame
                    self.train_data(
                        len,
                        frame + add_frame,
                        center_min,
                        center_max,
                        cand_id,
                    )

    def train_data(self, length, frame, center_min, center_max, cand_id):
        imgs = self.img_gen(length, frame, center_min, center_max)

        if self.annotations is not None:
            targets = self.target_gen(frame, length, center_max, center_min)

            if targets.max() == 0:
                save_path = self.save_path_root.joinpath(
                    f"negative_sequence/{int(frame):05d}-{int(cand_id):03d}-{int(length):03d}-{int(center_min[0]):04d}-{int(center_min[1]):04d}.h5py"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                save_path = self.save_path_root.joinpath(
                    f"positive_sequence/{int(frame):05d}-{int(cand_id):03d}-{int(length):03d}-{int(center_min[0]):04d}-{int(center_min[1]):04d}.h5py"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_path = self.save_path_root.joinpath(
                f"{int(frame):05d}-{int(cand_id):03d}-{int(length):03d}-{int(center_min[0]):04d}-{int(center_min[1]):04d}.h5py"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

        outfh = h5py.File(str(save_path), "w")
        outfh.create_dataset("img", data=imgs)
        if self.annotations is not None:
            outfh.create_dataset("target", data=targets)
        outfh.flush()
        outfh.close()

    def img_gen(self, length, frame, center_min, center_max):
        imgs = []
        for i in range(self.depth):
            if i < length:
                img_path = self.img_path_root.joinpath(
                    f"exp1_F{self.sequence:04d}-{int(frame + i):05d}.tif"
                )
                img = cv2.imread(str(img_path), -1)
                if self.pad:
                    img = np.pad(img, [30, 30], mode=self.pd_mode)
                try:
                    img = img[center_min[0]: center_max[0], center_min[1]: center_max[1]]
                    img = (img / 4096).astype(np.float32)
                except ValueError:
                    img = np.zeros((128, 128))
            else:
                img = np.zeros((128, 128))
            imgs.append(img)
        return np.array(imgs)

    def target_gen(self, frame, length, center_max, center_min):
        annotations_in_frame = self.annotations[
            (self.annotations[:, 0] >= frame)
            & (self.annotations[:, 0] < frame + length)
            & (self.annotations[:, 1] > center_min[1])
            & (self.annotations[:, 1] < center_max[1])
            & (self.annotations[:, 2] > center_min[0])
            & (self.annotations[:, 2] < center_max[0])
            ]
        if annotations_in_frame.shape[0] != 0:
            targets = np.zeros((128, 128, self.depth), dtype=np.float32)
            for annotation in annotations_in_frame:
                t, y, x = annotation.astype(np.int)
                Y, X, T = np.meshgrid(
                    list(range(128)), list(range(128)), list(range(self.depth))
                )
                target = gaussian(
                    T,
                    Y,
                    X,
                    t - frame,
                    y - center_min[1],
                    x - center_min[0],
                    self.plot_size,
                    self.sigma_t,
                )
                targets = np.maximum(targets, target)
            targets = targets / targets.max() - 0.000_001
            targets = targets.astype(np.float32).transpose(2, 0, 1)
            if np.isnan(targets.max()):
                pass
        else:
            targets = np.zeros((self.depth, 128, 128), dtype=np.float32)
        return targets


class ImageTargetGeneratorValidation(ImageTargetGenerator):
    def __init__(self, sequence=1, sigma_t=2, other=False):
        super().__init__(sequence=sequence, sigma_t=sigma_t)
        self.other = other
        if other:
            self.tra_path_root = Path(
                f"/home/kazuya/file_server2/cvpr_workshop/output/F{self.sequence:02d}_track_result"
            )

    def call_data_generater(self, length, frame, center_min, center_max, cand_id):

        if frame == 1:
            length += 2
        elif frame == 2:
            length += 3
            frame -= 1
        else:
            length += 4
            frame -= 2

        for_calculate_max_frame = length / 16 + 1
        if (frame + for_calculate_max_frame * 16) > 1013:
            if (frame + length) > 1013:
                length = length - ((frame + length) - 1013)

        depth = int(length / 16) + 1

        self.depth = depth * 16

        if length < 256:
            self.val_data(
                length,
                frame,
                center_min,
                center_max,
                cand_id,
            )

    def val_data(self, length, frame, center_min, center_max, cand_id):
        imgs = self.img_gen(length, frame, center_min, center_max)

        if self.other:
            tras = []
            for i in range(self.depth):
                if i < length:
                    img_path = self.tra_path_root.joinpath(f"{int(frame + i):04d}.png")
                    img = cv2.imread(str(img_path))

                    if self.pad:
                        img = np.pad(img, ((30, 30), (30, 30), (0, 0)), mode="constant")

                    img = img[
                          center_min[0]: center_max[0], center_min[1]: center_max[1]
                          ]
                    img = (img / 4096).astype(np.float32)
                else:
                    img = np.zeros((128, 128, 3))
                tras.append(img)
            tras = np.array(tras)

            tras = tras.transpose(3, 0, 1, 2)

        targets = self.target_gen(frame, length, center_max, center_min)

        center = ((center_max + center_min) / 2).astype(np.int)

        save_path = self.save_path_root.joinpath(
            f"{int(frame):05d}-{int(cand_id):03d}-{int(length):03d}-{int(center[0]):04d}-{int(center[1]):04d}.h5py"
        )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        outfh = h5py.File(str(save_path), "w")
        outfh.create_dataset("img", data=imgs)

        if self.other:
            outfh.create_dataset("tra", data=tras)
        if self.annotations is not None:
            outfh.create_dataset("target", data=targets)
        outfh.flush()
        outfh.close()


if __name__ == "__main__":
    # for seq in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
    #     cand_generater = ImageTargetGenerator(sequence=seq, sigma_t=2)
    #     cand_generater.generate_cand()

    for seq in range(3, 17, 4):
        cand_generater = ImageTargetGeneratorValidation(sequence=seq, sigma_t=2, other=False)
        cand_generater.generate_cand()
