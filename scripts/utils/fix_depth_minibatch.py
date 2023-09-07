from argparse import ArgumentParser
from pathlib import Path
from shutil import copy


def main(args):
    data_root = Path(args.data_root)
    data_path = data_root / args.shift_type / args.frame_rate

    minival_path = data_path / "minival"
    for img_path in minival_path.glob("*/img/*/*.jpg"):
        depth_minival_path = str(img_path).replace("img", "depth").replace(".jpg", ".png")
        depth_val_path = depth_minival_path.replace("minival", "val")
        if not Path(depth_minival_path).exists():
            copy(depth_val_path, depth_minival_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data-root", type=str, default="E:/shift/", help="Path to SHIFT dataset.")
    parser.add_argument("-t", "--shift-type", type=str, default="discrete", help="Domain shift type.")
    parser.add_argument("-fr", "--frame-rate", type=str, default="images", help="Select 'images' or 'videos'.")
    args = parser.parse_args()
    main(args)
