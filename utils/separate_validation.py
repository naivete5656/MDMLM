from pathlib import Path
import numpy as np
import shutil


def separate_validation(root_path):
    """
    Separate validation data from train sequence
    :param root_path: maked ground_truth
    """
    for seq_type in ["positive_sequence", "negative_sequence"]:
        for seq in [1, 2, 5, 6, 9, 10, 13, 14]:
            data_path = root_path.joinpath(f"F{seq:04d}/{seq_type}")
            paths = list(data_path.glob("*.h5py"))

            save_path = data_path.parent.parent.parent.joinpath(
                f"val/F{seq:04d}/{seq_type}"
            )
            save_path.mkdir(parents=True, exist_ok=True)

            for path in paths:
                frame, cand_id, length, x, y = path.stem.split("-")

                x = int(x)
                y = int(y)

                if (x < 512) & (y < 512):
                    shutil.move(str(path), save_path.joinpath(path.name))
