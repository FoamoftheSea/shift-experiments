import json
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

from shift_lab.semantic_segmentation.labels import id2label


def get_frame_counts(mask_path) -> dict:
    mask = np.array(Image.open(mask_path))
    class_ids, counts = np.unique(mask, return_counts=True)
    frame_counts = {id2label[cid].name: count for cid, count in zip(class_ids, counts)}
    return frame_counts


def normalize_dict(d):
    return {k: v / sum(d.values()) for k, v in d.items()}


def main(args):
    semseg_path = Path(args.dataset_root) / "discrete" / "images" / "train" / "front" / "semseg"
    class_counts = Counter()

    with ThreadPoolExecutor() as pool:
        frame_futures = set()
        for scene_folder in semseg_path.iterdir():
            for mask_path in scene_folder.glob("*.png"):
                frame_futures.add(
                    pool.submit(
                        get_frame_counts,
                        mask_path
                    )
                )

        for res in as_completed(frame_futures):
            class_counts.update(res.result())

    with open(args.output, "w") as f:
        json.dump(normalize_dict(dict(class_counts)), f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset-root", default="E:/shift")
    parser.add_argument("-o", "--output", default="./shift_semseg_stats.json")

    args = parser.parse_args()
    main(args)