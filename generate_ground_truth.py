from pathlib import Path
import scipy.io as io
import numpy as np
import h5py
import cv2
from utils import associate_candidate, separate_validation
import argparse


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="data path")
    parser.add_argument(
        "-c",
        "--cand_path",
        dest="cand_path",
        help="candidate path that is generate by candidate_extractor.m",
        default="./output/candidate",
        type=str,
    )
    parser.add_argument(
        "-a",
        "--annotation_path",
        dest="annotation_path",
        help="annotation path that includes in dataset",
        # default="./MitosisAnnotations",
        default="/home/kazuya/ssd/cvpr_workshop/MitosisAnnotations",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--img_path",
        dest="img_path",
        help="image path",
        # default="./MicroscopyImages",
        default="/home/kazuya/ssd/cvpr_workshop/CVPR_workshop",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--save_path",
        dest="save_path",
        help="save path",
        default="./images",
        type=str,
    )

    args = parser.parse_args()
    return args


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


def gen_imgs_targets(
    depth,
    length,
    img_path_root,
    sequence,
    frame,
    center_min,
    center_max,
    annotations,
    plot_size,
    sigma_t,
):
    imgs = []
    for i in range(depth):
        if i < length:
            img_path = img_path_root.joinpath(
                f"exp1_F{sequence:04d}-{int(frame + i):05d}.tif"
            )
            img = cv2.imread(str(img_path), -1)
            img = np.pad(img, [30, 30], mode="constant")
            try:
                img = img[center_min[0] : center_max[0], center_min[1] : center_max[1]]
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
    imgs, targets = gen_imgs_targets(
        depth,
        length,
        img_path_root,
        sequence,
        frame,
        center_min,
        center_max,
        annotations,
        plot_size,
        sigma_t,
    )

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
    cand_path,
    img_path_root,
    save_path_root,
    annotation_path,
    sequence=1,
    depth=16,
    plot_size=9,
    sigma_t=1,
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
    imgs, targets = gen_imgs_targets(
        depth,
        length,
        img_path_root,
        sequence,
        frame,
        center_min,
        center_max,
        annotations,
        plot_size,
        sigma_t,
    )

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
    cand_path,
    img_path_root,
    save_path_root,
    annotation_path,
    sequence=1,
    plot_size=9,
    sigma_t=1,
):
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


if __name__ == "__main__":
    args = parse_args()
    for seq in [1, 2, 5, 6, 9, 10, 13, 14]:
        print(f"sequence:{seq}")
        cand_path = Path(args.cand_path).joinpath(f"F{seq:04d}/candidate.mat")
        annotation_path = Path(args.annotation_path).joinpath(f"090303_F{seq:04d}.mat")
        associate_candidate(cand_path, seq, annotation_path)

    for seq in [1, 2, 5, 6, 9, 10, 13, 14]:
        cand_path = Path(args.cand_path).joinpath(f"F{seq:04d}/prepro_candidate.mat")
        img_path_root = Path(args.img_path).joinpath(f"F{seq:04d}")
        annotation_path = Path(args.annotation_path).joinpath(f"090303_F{seq:04d}.mat")
        save_path_root = Path(args.save_path).joinpath(f"train/F{seq:04d}")
        generate_cand(
            cand_path, img_path_root, save_path_root, annotation_path, sequence=seq
        )

    for i in [3, 7, 11, 15]:
        cand_path = Path(args.cand_path).joinpath(f"F{seq:04d}/prepro_candidate.mat")
        img_path_root = Path(args.img_path).joinpath(f"F{seq:04d}")
        annotation_path = Path(args.annotation_path).joinpath(f"090303_F{seq:04d}.mat")
        save_path_root = Path(args.save_path).joinpath(f"test/F{seq:04d}")
        generate_cand_val(cand_path, img_path_root, save_path_root, annotation_path)

    separate_validation(Path(args.save_path))
