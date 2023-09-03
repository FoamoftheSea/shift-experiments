from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from PIL import Image

from shift_lab.semantic_segmentation.labels import id2label


def get_frame_counts(mask_path):
    mask = Image.open(mask_path)


def main():
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



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset-root", default="D:/shift_small")

    args = parser.parse_args()
    main(args)