from utils import Config

from sklearn.model_selection import train_test_split

import pandas as pd
import os

from tqdm import tqdm


class NumberOfFileNotSame(Exception):
    def __init__(self):
        super().__init__("The number of image files and label files doesn't same.")


def main():
    cfg = Config()
    total_img_file_paths = list()
    total_labels = list()
    total_lndmrks = list()

    for i, dir_name in enumerate(os.listdir(cfg.dataset_root_path)):
        dir_path = os.path.join(cfg.dataset_root_path, dir_name)
        img_dir_path = os.path.join(dir_path, "lit")
        anno_dir_path = os.path.join(dir_path, "annotation")

        read_data_tqdm = tqdm(
            iterable=zip(os.listdir(os.path.join(dir_path, "lit")), os.listdir(os.path.join(dir_path, "annotation"))),
            desc=f"Read data from {dir_name}...(label_name: {i})",
            total=len(os.listdir(os.path.join(dir_path, "lit")))
        )
        for img_name, anno_name in read_data_tqdm:
            # Append image file path
            img_file_path = os.path.join(img_dir_path, img_name)
            total_img_file_paths.append(img_file_path)

            # Append label and landmark value
            label_file_path = os.path.join(anno_dir_path, anno_name)
            total_labels.append(i)
            train_df = pd.read_json(label_file_path)
            lndmrks = train_df["Landmarks"].to_numpy()
            total_lndmrks.append(lndmrks)

    if len(total_img_file_paths) != len(total_labels):
        raise NumberOfFileNotSame()

    train_all_img_file_paths, test_img_file_paths, train_all_labels, test_labels, train_all_lndmrks, test_lndmrks = train_test_split(
        total_img_file_paths,
        total_labels,
        total_lndmrks,
        test_size=0.4,
        stratify=total_labels
    )
    train_img_file_paths, valid_img_file_paths, train_labels, valid_labels, train_lndmrks, valid_lndmrks = train_test_split(
        train_all_img_file_paths,
        train_all_labels,
        train_all_lndmrks,
        test_size=0.2,
        stratify=train_all_labels
    )
    print(len(train_img_file_paths), len(train_labels), len(train_lndmrks))
    print(len(valid_img_file_paths), len(valid_labels), len(valid_lndmrks))
    print(len(test_img_file_paths), len(test_labels), len(test_lndmrks))

    train_df = pd.DataFrame({
        "img_file_paths": train_img_file_paths,
        "labels": train_labels,
        "lndmrks": train_lndmrks
    })
    train_df.to_json(
        path_or_buf=f"{os.path.dirname(cfg.dataset_root_path)}/train_dataset.json",
        orient="records",
        index=False,
        indent=4
    )
    valid_df = pd.DataFrame({
        "img_file_paths": valid_img_file_paths,
        "labels": valid_labels,
        "lndmrks": valid_lndmrks,
    })
    valid_df.to_json(
        path_or_buf=f"{os.path.dirname(cfg.dataset_root_path)}/valid_dataset.json",
        orient="records",
        index=False,
        indent=4
    )
    test_df = pd.DataFrame({
        "img_file_paths": test_img_file_paths,
        "labels": test_labels,
        "lndmrks": test_lndmrks
    })
    test_df.to_json(
        path_or_buf=f"{os.path.dirname(cfg.dataset_root_path)}/test_dataset.json",
        orient="records",
        index=False,
        indent=4
    )


if __name__ == '__main__':
    main()
