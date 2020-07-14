from pathlib import Path
import numpy as np
import scipy.io as io
from utils.matching import optimum
import scipy.io as io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


data_type = {"train": [1, 2, 5, 6, 9, 10, 13, 14], "test": [3, 7, 11, 15]}
Group = {
    "Control": list(range(1, 5)),
    "FGF2": list(range(5, 9)),
    "BMP2": list(range(9, 13)),
    "BMP2+FGF2": list(range(13, 17)),
}


def check_candidate(cand, annotation_path, sequence):
    """
    :param cand: extracted candidate (t, x, y)
    :param annotation_path: ground_truth path (t, x, y)
    :param save_path:
    :return: check candidate accuracy
    """
    annotations = io.loadmat(str(annotation_path))["result"]

    annotations[:, 1] = annotations[:, 1] + 30
    annotations[:, 2] = annotations[:, 2] + 30

    include_rate = []
    for annotation in annotations:
        t, x, y = annotation
        cand_in_frame = cand[cand[:, 0] == t]

        flag = np.any(
            (cand_in_frame[:, 1] < x)
            & (cand_in_frame[:, 3] > x)
            & (cand_in_frame[:, 2] < y)
            & (cand_in_frame[:, 4] > y)
        )
        include_rate.append(flag)
    include_rate = sum(include_rate)

    print(f"sequence{sequence}=")
    print(include_rate / annotations.shape[0])

    return include_rate / annotations.shape[0]


def cand_get_centor(cand_pos):
    """
    patch position to image position
    :param cand_pos: candidate position
    :return: center coodinates of image
    """
    cand_in_frame_center = [
        ((cand_pos[:, 3] + cand_pos[:, 1]) / 2),
        ((cand_pos[:, 4] + cand_pos[:, 2]) / 2),
    ]
    return np.array(cand_in_frame_center).transpose(1, 0)


def associate_candidate(cand_path, sequence, annotation_path=None):
    """
    Associate candidate patch to time direction.
    associate_cand = [length, x_min, y_min, x_max, y_max, frame]

    :param cand_path: file path for extracted candidate   
    :param sequence: chandidate sequence
    :param annotation_path: To confirm candidate extraction accuracy
    :return associate_candidate
    """

    # load candidate patch
    cands = io.loadmat(str(cand_path))["candi"][0]
    associate_cand = []
    for cand in cands:
        associate_cand.append(cand[0])
    cand = np.array(associate_cand)
    # cand = [frame, x_min, y_min, x_max, y_max]

    # add_id
    id_box = []
    ids = np.arange(cand.shape[0]).reshape(cand.shape[0], 1)
    cand = np.append(cand, ids.reshape((ids.shape[0], 1)), axis=1)

    associate_cand = cand[cand[:, 0] == 1]
    associate_cand = np.append(
        associate_cand, np.ones((associate_cand.shape[0], 1)), axis=1
    )

    for id in associate_cand[:, 5]:
        id_box.append([int(id)])

    # candidate 2 cand sequence
    for frame in range(1, 1014):
        cand_in_frame = cand[cand[:, 0] == frame]
        in_farme_cand_center = cand_get_centor(cand_in_frame)

        cand_next_frame = cand[cand[:, 0] == frame + 1]
        next_farme_cand_center = cand_get_centor(cand_next_frame)

        # associate candidate based on distance
        associate_ids = optimum(in_farme_cand_center, next_farme_cand_center, 20)
        cand_index = set(np.arange(cand_next_frame.shape[0])) - set(associate_ids[:, 1])
        cand_next_frame[:, 0] = 1

        # add new candidate
        if cand_next_frame[list(cand_index)].shape[0] != 0:
            for temp in cand_next_frame[list(cand_index)]:
                temp = np.array([np.append(temp, frame + 1)])
                associate_cand = np.append(associate_cand, temp, axis=0)

        for temp in cand_next_frame[list(cand_index)][:, 5]:
            id_box.append([int(temp)])

        for associate_id in associate_ids.astype(np.int):
            temp = associate_cand[
                associate_cand[:, 5] == cand_in_frame[associate_id[0]][5]
            ]

            change_mat_index = int(
                np.where(associate_cand[:, 5] == cand_in_frame[associate_id[0]][5])[0]
            )
            id_box[change_mat_index].append(int(cand_next_frame[associate_id[1]][5]))
            temp[0, 0] += 1
            temp[0, 5] = cand_next_frame[associate_id[1]][5]
            associate_cand[
                associate_cand[:, 5] == cand_in_frame[associate_id[0]][5]
            ] = temp

    if annotation_path is not None:
        cut_index = np.where((associate_cand[:, 0] > 1))[0]
        new_id_box = []
        for index in cut_index:
            new_id_box.extend(id_box[index])
        cand_for_conf = []
        for index in new_id_box:
            cand_for_conf.append(cand[cand[:, 5] == index][0])
        result = check_candidate(np.array(cand_for_conf), annotation_path, sequence)

    cand_save_path = cand_path.parent.joinpath("prepro_candidate.mat")
    io.savemat(cand_save_path, {"candi": associate_cand})
