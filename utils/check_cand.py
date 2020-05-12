import scipy.io as io
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


data_type = {
    "train": [1, 2, 5, 6, 9, 10, 13, 14],
    "val": [3, 7, 11, 15],
    "test": [4, 8, 12, 16],
}
Group = {
    1: list(range(1, 5)),
    2: list(range(5, 9)),
    3: list(range(9, 13)),
    4: list(range(13, 17)),
}


def check_candidate(cand, annotation_path, sequence):
    """
    :param cand: extracted candidate (t, x, y)
    :param annotation_path: ground_truth path (t, x, y)
    :param save_path:
    :return:
    """
    for key in data_type.keys():
        if sequence in data_type[key]:
            mode = key
            break

    if mode != "test":
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
