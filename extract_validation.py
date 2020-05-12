from pathlib import Path
import numpy as np
import shutil

modes = [
    "train",
]
seq_type = ["positive_sequence", "negative_sequence"]

for mode in modes:
    for s_type in [0, 1]:
        for seq in [1, 2, 5, 6, 9, 10, 13, 14]:
            root_path = Path(
                f"/home/kazuya/hdd/CVPR_workshop/images/{mode}/F{seq:04d}/{seq_type[s_type]}"
            )
            paths = list(root_path.glob("*.h5py"))

            save_path = root_path.parent.parent.parent.joinpath(
                f"{mode}_val/F{seq:04d}/{seq_type[s_type]}"
            )
            save_path.mkdir(parents=True, exist_ok=True)

            for path in paths:
                frame, cand_id, length, x, y = path.stem.split("-")

                x = int(x)
                y = int(y)

                if (x < 512) & (y < 512):
                    shutil.move(str(path), save_path.joinpath(path.name))
                    # print(path)
                    # print(save_path.joinpath(path.name))

    # new_path = shutil.move("temp/dir1/file.txt", "temp/dir2/")
