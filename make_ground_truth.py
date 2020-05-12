from pathlib import Path
import scipy.io as io
import numpy as np
import h5py
import cv2
from utils import sophicicated_method


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


def gen_imgs_targets(depth, length, img_path_root, sequence, frame, center_min, center_max, annotations, plot_size,
                     sigma_t):
    imgs = []
    for i in range(depth):
        if i < length:
            img_path = img_path_root.joinpath(
                f"exp1_F{sequence:04d}-{int(frame + i):05d}.tif"
            )
            img = cv2.imread(str(img_path), -1)
            img = np.pad(img, [30, 30], mode="constant")
            try:
                img = img[center_min[0]: center_max[0], center_min[1]: center_max[1]]
                img = (img / 4096).astype(np.float32)
            except ValueError:
                img = np.zeros((128, 128))
        else:
            img = np.zeros((128, 128))
        imgs.append(img)
    imgs = np.array(imgs)

    if annotations is not None:
        annotations_in_frame = annotations[
            (annotations[:, 0] >= frame)
            & (annotations[:, 0] < frame + length)
            & (annotations[:, 1] > center_min[1])
            & (annotations[:, 1] < center_max[1])
            & (annotations[:, 2] > center_min[0])
            & (annotations[:, 2] < center_max[0])
            ]

        if annotations_in_frame.shape[0] != 0:
            targets = np.zeros((128, 128, depth), dtype=np.float32)
            for annotation in annotations_in_frame:
                t, y, x = annotation.astype(np.int)
                Y, X, T = np.meshgrid(
                    list(range(128)), list(range(128)), list(range(depth))
                )
                target = gaussian(
                    T,
                    Y,
                    X,
                    t - frame,
                    y - center_min[1],
                    x - center_min[0],
                    plot_size,
                    sigma_t,
                )
                targets = np.maximum(targets, target)
            targets = targets / targets.max() - 0.000_001
            targets = targets.astype(np.float32).transpose(2, 0, 1)
            if np.isnan(targets.max()):
                pass
        else:
            targets = np.zeros((depth, 128, 128), dtype=np.float32)

        return imgs, targets
    return imgs, None


def imgs_targets(
        length,
        img_path_root,
        frame,
        sequence,
        center_min,
        center_max,
        annotations,
        depth,
        plot_size,
        sigma_t,
        save_path_root,
        cand_id,
):
    imgs, targets = gen_imgs_targets(depth, length, img_path_root, sequence, frame, center_min, center_max, annotations,
                                     plot_size, sigma_t)

    if annotations is not None:
        if targets.max() == 0:
            save_path = save_path_root.joinpath(
                f"negative_sequence/{int(frame):05d}-{int(cand_id):03d}-{int(length):03d}-{int(center_min[0]):04d}-{int(center_min[1]):04d}.h5py"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_path = save_path_root.joinpath(
                f"positive_sequence/{int(frame):05d}-{int(cand_id):03d}-{int(length):03d}-{int(center_min[0]):04d}-{int(center_min[1]):04d}.h5py"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_path = save_path_root.joinpath(
            f"{int(frame):05d}-{int(cand_id):03d}-{int(length):03d}-{int(center_min[0]):04d}-{int(center_min[1]):04d}.h5py"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

    outfh = h5py.File(str(save_path), "w")
    outfh.create_dataset("img", data=imgs)
    if annotations is not None:
        outfh.create_dataset("target", data=targets)
    outfh.flush()
    outfh.close()


def generate_cand(
        cand_path, img_path_root, save_path_root, annotation_path, sequence=1, depth=16, plot_size=9, sigma_t=1,
):
    cands = io.loadmat(cand_path)["candi"]
    try:

        annotations = io.loadmat(str(annotation_path))["result"]

        annotations[:, 1] = annotations[:, 1] + 30
        annotations[:, 2] = annotations[:, 2] + 30
    except:
        annotations = None

    lim_size = np.array([1040, 1392])

    lim_size = lim_size + 60

    for cand_id, cand in enumerate(cands):
        length = cand[0]
        frame = cand[6]
        center_min = (cand[1:3] + cand[3:5]) / 2 - 64
        center_min = center_min[::-1]
        center_min[0] = center_min[0].clip(0, lim_size[0] - 128)
        center_min[1] = center_min[1].clip(0, lim_size[1] - 128)
        center_min = center_min.astype(np.int)
        center_max = center_min + 128

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
        if (frame + for_calcurate_max_frame * 16) > 1014:
            frame = 1014 - for_calcurate_max_frame * 16

        if length < depth:
            imgs_targets(
                length,
                img_path_root,
                frame,
                sequence,
                center_min,
                center_max,
                annotations,
                depth,
                plot_size,
                sigma_t,
                save_path_root,
                cand_id,
            )
        else:
            if length < 256:
                for add_frame in range(0, int(length), int(depth / 2)):
                    if (length - add_frame) > depth:
                        len = depth
                    else:
                        len = length - add_frame
                    imgs_targets(
                        len,
                        img_path_root,
                        frame + add_frame,
                        sequence,
                        center_min,
                        center_max,
                        annotations,
                        depth,
                        plot_size,
                        sigma_t,
                        save_path_root,
                        cand_id,
                    )


def imgs_targets_tra(
        length,
        img_path_root,
        frame,
        sequence,
        center_min,
        center_max,
        annotations,
        depth,
        plot_size,
        sigma_t,
        save_path_root,
        cand_id,
):
    imgs, targets = gen_imgs_targets(depth, length, img_path_root, sequence, frame, center_min, center_max, annotations,
                                     plot_size, sigma_t)

    center = ((center_max + center_min) / 2).astype(np.int)

    save_path = save_path_root.joinpath(
        f"{int(frame):05d}-{int(cand_id):03d}-{int(length):03d}-{int(center[0]):04d}-{int(center[1]):04d}.h5py"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    outfh = h5py.File(str(save_path), "w")
    outfh.create_dataset("img", data=imgs)

    if annotations is not None:
        outfh.create_dataset("target", data=targets)
    outfh.flush()
    outfh.close()


def generate_cand_val(
        cand_path, img_path_root, save_path_root, annotation_path, sequence=1, plot_size=9, sigma_t=1):
    """
    meke cand image & tracking res & try annotation iamge
    the code use only test sequence
    the length of cand img is same to default candidate length
    :param sequence: sequence index
    :param plot_size: the param decide img direction gaussian spred
    :param sigma_t: the param decide time direction gaussian spread
    :return: save img h5py
    """

    cands = io.loadmat(cand_path)["candi"]
    try:
        annotations = io.loadmat(str(annotation_path))["result"]
        annotations[:, 1] = annotations[:, 1] + 30
        annotations[:, 2] = annotations[:, 2] + 30
    except:
        annotations = None

    lim_size = np.array([1040, 1392])

    lim_size = lim_size + 60

    for cand_id, cand in enumerate(cands):
        length = int(cand[0])
        frame = cand[6] + 1
        center_min = (cand[1:3] + cand[3:5]) / 2 - 64
        center_min = center_min[::-1]
        center_min[0] = center_min[0].clip(0, lim_size[0] - 128)
        center_min[1] = center_min[1].clip(0, lim_size[1] - 128)
        center_min = center_min.astype(np.int)
        center_max = center_min + 128

        if frame == 1:
            length += 2
        elif frame == 2:
            length += 3
            frame -= 1
        else:
            length += 4
            frame -= 2

        for_calcurate_max_frame = length / 16 + 1
        if (frame + for_calcurate_max_frame * 16) > 1013:
            if (frame + length) > 1013:
                print(length)
                length = length - ((frame + length) - 1013)

        depth = int(length / 16) + 1

        depth = depth * 16

        if length < 256:
            imgs_targets_tra(
                length,
                img_path_root,
                frame,
                sequence,
                center_min,
                center_max,
                annotations,
                depth,
                plot_size,
                sigma_t,
                save_path_root,
                cand_id,
            )


MODES = ["train"]
if __name__ == "__main__":
    for i in range(1, 17):
        print(f"sequence:{i}")
        cand_path = Path(f"/home/kazuya/hdd/for_upload_code/output/candidate/F{i:04d}/candidate.mat")
        annotation_path = Path(f"/Annotation_path/090303_F{i:04d}.mat")
        z(cand_path, annotation_path, i)

    for i in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
        cand_path = Path(f"./output/candidate/F{i:04d}/prepro_candidate.mat")
        img_path_root = Path(f"/input_file_path/F{i:04d}")


        annotation_path = Path(
            f"/Annotation_path/090303_F{i:04d}.mat"
        )

        if i % 4 != 0:
            save_path_root = Path(
                f"./images/train/F{i:04d}"
            )
            generate_cand(cand_path, img_path_root, save_path_root, annotation_path, sequence=i, sigma_t=2)
        else:
            save_path_root = Path(
                f"./images/val/F{i:04d}"
            )
            generate_cand_val(cand_path, img_path_root, save_path_root, annotation_path, sequence=i)
