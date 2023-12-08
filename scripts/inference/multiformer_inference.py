import json
import os
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Mapping

import cv2

import numpy as np
import torch
from shift_dev import SHIFTDataset
from shift_dev.dataloader.image_processors import MultitaskImageProcessor
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend
from shift_dev.utils.load import im_decode
from transformers import Multiformer
from transformers.models.multiformer.image_processing_multiformer import post_process_object_detection


def save_json(d, fp):
    with open(fp, "w") as f:
        json.dump(nested_numpify(d), f)


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, torch.Tensor) and len(tensors.shape) == 0:
        return tensors.item()
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    elif isinstance(tensors, (np.ndarray, torch.Tensor)):
        return [nested_numpify(t) for t in tensors]
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})
    elif isinstance(tensors, (int, float)) or tensors is None:
        return tensors
    elif isinstance(tensors, (str, Path)):
        return str(tensors)
    else:
        raise ValueError("Unexpected type:", type(tensors))



def shift_multiformer_collator(features):

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k == "labels" or k == "labels_3d":
            batch[k] = [f[k] for f in features]
        elif k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


class SHIFTInferenceCameraFrame:

    def __init__(self, frame, camera_name):
        self.frame = frame
        self.camera_name = camera_name
        self._image = None
        self._semseg = None
        self._depth = None
        self.data_root = self.frame.scene.scene_path / self.camera_name

    @property
    def image(self):
        if self._image is None:
            fp = (
                self.data_root
                / "img"
                / f"{self.frame.frame_id}_img_{self.camera_name}.png"
            )
            self._image = self._load_image(filepath=str(fp))

        return self._image

    @property
    def semantic_mask(self):
        if self._semseg is None:
            fp = (
                self.data_root
                / "semseg"
                / f"{self.frame.frame_id}_semseg_{self.camera_name}.png"
            )
            self._semseg = self._load_semseg(filepath=str(fp))

        return self._semseg

    @property
    def depth(self):
        if self._depth is None:
            fp = (
                self.data_root
                / "depth"
                / f"{self.frame.frame_id}_depth_{self.camera_name}.npy"
            )
            self._depth = self._load_depth(filepath=str(fp))

        return self._depth

    def _load_image(self, filepath: str) -> np.ndarray:
        im_bytes = self.frame.scene.dataset.backend.get(filepath)
        image = im_decode(im_bytes, mode="RGB")
        return image  # torch.as_tensor(image, dtype=torch.int64)

    def _load_semseg(self, filepath: str) -> np.ndarray:
        im_bytes = self.frame.scene.dataset.backend.get(filepath)
        semseg = im_decode(im_bytes)[..., 0]
        return semseg

    def _load_depth(self, filepath: str) -> np.ndarray:
        depth = np.load(filepath)
        return depth


class SHIFTInferenceFrame:

    def __init__(self, scene, frame_id):
        self.scene = scene
        self.frame_id = frame_id

    def get_camera_frame(self, camera_name):
        return SHIFTInferenceCameraFrame(self, camera_name)


class SHIFTInferenceScene:

    def __init__(self, dataset, scene_name):
        self.dataset = dataset
        self.name = scene_name
        self.scene_path = self.dataset.dataset_path / self.name
        self.camera_names = [path.stem for path in self.scene_path.iterdir()]
        self.frame_ids = [str(path.stem).split("_")[0] for path in (self.scene_path / self.camera_names[0] / "img").glob("*.png")]

    def get_frame(self, frame_id):
        return SHIFTInferenceFrame(self, frame_id)


def prepare_inputs(input, device):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    elif isinstance(input, np.ndarray):
        return torch.tensor(input).to(device)
    elif isinstance(input, Mapping):
        return {k: prepare_inputs(v, device) for k, v in input.items()}
    else:
        raise ValueError("input must be tensor, array or mapping. Got {}".format(type(input)))


class SHIFTInferenceDataset:

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.scene_names = [path.stem for path in self.dataset_path.iterdir()]
        self.backend = FileBackend()

    def get_scene(self, scene_name: str):
        return SHIFTInferenceScene(self, scene_name)


def main(args):
    keys_to_load = [Keys.images, Keys.boxes2d]
    if args.load_gt:
        if "semseg" in args.tasks:
            keys_to_load.append(Keys.segmentation_masks)
        if "depth" in args.tasks:
            keys_to_load.append(Keys.depth_maps)

    views_to_load = args.views if not isinstance(args.views, str) else (args.views, )

    dataset = SHIFTDataset(
        data_root=args.data_root,
        split=args.split,
        keys_to_load=keys_to_load,
        views_to_load=views_to_load,
        framerate=args.framerate,
        shift_type=args.shift_type,
        backend=FileBackend(),
        num_workers=1,
        verbose=False,
        image_transforms=None,
        frame_transforms=None,
        image_processor=None,
        load_full_res=True,
        depth_mask_semantic_ids=None,
        depth_mask_value=0.0,
    )

    model = Multiformer.from_pretrained(
        args.checkpoint,
        ignore_mismatched_sizes=False,
        # device_map=args.device,
    ).to(args.device)
    model.eval()

    for view in views_to_load:
        sd = dataset.scalabel_datasets[f"{view}/det_2d"]
        id2label_boxes2d = {v: k for k, v in sd.cats_name2id["boxes2d"].items()}
        frame_ids = dataset.video_to_indices[args.scene_name]
        view_path = Path(args.output) / args.shift_type / args.framerate / args.split / view
        det2d_path = view_path / "det_2d.json"
        det2d_pred_frames = []

        if not os.path.exists(det2d_path):
            write_json = {
                "config": {
                    "imageSize": dict(sd.cfg.imageSize),
                    "categories": [{"name": category.name} for category in sd.cfg.categories]
                },
                "frames": [],
            }
        else:
            with open(det2d_path, "r") as f:
                write_json = json.load(f)

        for frame_id in frame_ids:
            rgb_frame = dataset._load_image(sd.frames[frame_id].url)
            processed_frame = shift_multiformer_collator([prepare_inputs(dataset[frame_id], args.device)])

            outputs = model(**processed_frame)
            rgb_path = view_path / "img" / args.scene_name

            out_path = rgb_path / sd.frames[frame_id].name
            out_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))

            if hasattr(outputs, "logits_semantic"):
                semseg_path = view_path / "semseg" / args.scene_name
                semseg_path.mkdir(exist_ok=True, parents=True)
                logits_tensor = torch.nn.functional.interpolate(
                    outputs.logits_semantic,
                    size=processed_frame["pixel_values"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).argmax(dim=1)

                pred_labels = logits_tensor.detach().cpu().numpy()

                if model.config.backbone_config.do_reduce_labels:
                    pred_labels += 1
                    pred_labels[pred_labels == 256] = 0

                for i, pred_label in enumerate(pred_labels):
                    out_path = semseg_path / f"{sd.frames[frame_id].frameIndex:08d}_semseg_{view}.png"
                    out_path.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(out_path), pred_label)

            if hasattr(outputs, "pred_depth"):
                depth_path = view_path / "depth" / args.scene_name
                depth_path.mkdir(exist_ok=True, parents=True)
                depth = np.exp(outputs.pred_depth.detach().cpu().numpy()) / 1000
                depth *= 16777216.0
                depth = depth.transpose(1, 2, 0).round().astype(np.int32)
                depth_8bit = np.concatenate([depth & 0xFF, depth >> 8 & 0xFF, depth >> 16 & 0xFF], axis=-1).astype(np.uint8)
                cv2.imwrite(str(depth_path / f"{sd.frames[frame_id].frameIndex:08d}_depth_{view}.png"), cv2.cvtColor(depth_8bit, cv2.COLOR_BGR2RGB))

            if hasattr(outputs, "pred_boxes"):

                preds = post_process_object_detection(
                    outputs=outputs, threshold=0.30, target_sizes=[rgb_frame.shape[:2]]
                )
                pred_frame = {
                    "name": sd.frames[frame_id].name,
                    "videoName": sd.frames[frame_id].videoName,
                    "intrinsics": dict(sd.frames[frame_id].intrinsics),
                    "extrinsics": dict(sd.frames[frame_id].extrinsics),
                    "attributes": dict(sd.frames[frame_id].attributes),
                    "frameIndex": sd.frames[frame_id].frameIndex,
                    "labels": [
                        {
                            "id": i + 1,
                            "category": id2label_boxes2d[label.item()],
                            "box2d": {"x1": box2d[0], "y1": box2d[1], "x2": box2d[2], "y2": box2d[3]},
                        } for i, (box2d, label) in enumerate(zip(preds[0]["boxes"], preds[0]["labels"]))
                    ],
                }
                # pred_frame["url"] = out_path
                det2d_pred_frames.append(pred_frame)

        if len(det2d_pred_frames) > 0:
            write_json["frames"].extend(det2d_pred_frames)
            save_json(write_json, det2d_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Use model checkpoint to generate inference")
    parser.add_argument("-d", "--data-root", type=str, default="C:/Users/indez/shift_small", help="Path to SHIFT dataset.")
    parser.add_argument("-s", "--split", type=str, default="val", help="Split to load for inference.")
    parser.add_argument("-v", "--views", nargs="*", type=str, default="front", help="Views to load for inference.")
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to the model checkpoint to load.")
    parser.add_argument("-t", "--tasks", nargs="*", type=str, help="Tasks to run inference on.")
    parser.add_argument("-o", "--output", type=str, default="./inference_out", help="Path for output.")
    parser.add_argument("-fr", "--framerate", type=str, default="images", help="Framerate to load.")
    parser.add_argument("-sn", "--scene-name", type=str, default="4bed-91bf", help="Name of scene to load.")
    parser.add_argument("-bs", "--batch-size", type=int, default=4, help="Number of frames to run at once.")
    parser.add_argument("--device", type=str, default="cpu", help="Pytorch device for computation.")
    parser.add_argument("-gt", "--load-gt", action="store_true", help="Flag to load ground truth for loss calculation.")
    parser.add_argument("-shift", "--shift_type", type=str, default="discrete", help="Domain shift type (continuous/discrete).")

    with torch.no_grad():
        main(parser.parse_args())
