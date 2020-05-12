from pathlib import Path
import numpy as np
import scipy.io as io
from utils.matching import optimum
from utils.check_cand import check_candidate


def cand_get_centor(cand_in_frame):

    cand_in_frame_center = [
        ((cand_in_frame[:, 3] + cand_in_frame[:, 1]) / 2),
        ((cand_in_frame[:, 4] + cand_in_frame[:, 2]) / 2),
    ]
    return np.array(cand_in_frame_center).transpose(1, 0)


def sophicicated_method(cand_path, annotation_path, sequence):
    cands = io.loadmat(str(cand_path))["candi"][0]

    new_cand = []
    for cand in cands:
        new_cand.append(cand[0])
    cand = np.array(new_cand)

    # add_id
    id_box = []
    ids = np.arange(cand.shape[0]).reshape(cand.shape[0], 1)
    cand = np.append(cand, ids.reshape((ids.shape[0], 1)), axis=1)

    new_cand = cand[cand[:, 0] == 1]
    new_cand = np.append(new_cand, np.ones((new_cand.shape[0], 1)), axis=1)

    for id in new_cand[:, 5]:
        id_box.append([int(id)])

    # candidate 2 cand sequence
    for frame in range(1, 1014):
        cand_in_frame = cand[cand[:, 0] == frame]
        in_farme_cand_center = cand_get_centor(cand_in_frame)

        cand_next_frame = cand[cand[:, 0] == frame + 1]
        next_farme_cand_center = cand_get_centor(cand_next_frame)

        associate_ids = optimum(in_farme_cand_center, next_farme_cand_center, 20)
        cand_index = set(np.arange(cand_next_frame.shape[0])) - set(associate_ids[:, 1])
        cand_next_frame[:, 0] = 1

        if cand_next_frame[list(cand_index)].shape[0] != 0:
            for temp in cand_next_frame[list(cand_index)]:
                temp = np.array([np.append(temp, frame + 1)])
                new_cand = np.append(new_cand, temp, axis=0)

        for temp in cand_next_frame[list(cand_index)][:, 5]:
            id_box.append([int(temp)])

        for associate_id in associate_ids.astype(np.int):
            temp = new_cand[new_cand[:, 5] == cand_in_frame[associate_id[0]][5]]

            change_mat_index = int(
                np.where(new_cand[:, 5] == cand_in_frame[associate_id[0]][5])[0]
            )
            id_box[change_mat_index].append(int(cand_next_frame[associate_id[1]][5]))
            temp[0, 0] += 1
            temp[0, 5] = cand_next_frame[associate_id[1]][5]
            new_cand[new_cand[:, 5] == cand_in_frame[associate_id[0]][5]] = temp

    cut_index = np.where((new_cand[:, 0] > 0))[0]
    new_id_box = []
    for index in cut_index:
        new_id_box.extend(id_box[index])

    cand_for_show = []
    for index in new_id_box:
        cand_for_show.append(cand[cand[:, 5] == index][0])

    result = check_candidate(np.array(cand_for_show), annotation_path, sequence)

    cand_save_path = cand_path.parent.joinpath("prepro_candidate.mat")
    io.savemat(cand_save_path, {"candi": result})




