from argparse import ArgumentParser
from pathlib import Path

import cv2

import numpy as np
import torch
from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend
from shift_dev.utils.load import im_decode

from shift_lab.semantic_segmentation.segformer.constants import SegformerTask
from shift_lab.semantic_segmentation.segformer.trainer import MultitaskSegformer


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


class SHIFTInferenceDataset:

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.scene_names = [path.stem for path in self.dataset_path.iterdir()]
        self.backend = FileBackend()

    def get_scene(self, scene_name: str):
        return SHIFTInferenceScene(self, scene_name)


def main(args):
    keys_to_load = [Keys.images]
    if args.load_gt:
        if SegformerTask.SEMSEG in args.tasks:
            keys_to_load.append(Keys.segmentation_masks)
        if SegformerTask.DEPTH in args.tasks:
            keys_to_load.append(Keys.depth_maps)

    views_to_load = args.views if not isinstance(args.views, str) else (args.views, )

    dataset = SHIFTDataset(
        data_root=args.data_root,
        split=args.split,
        keys_to_load=keys_to_load,
        views_to_load=views_to_load,
        framerate=args.framerate,
        shift_type="discrete",
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

    model = MultitaskSegformer.from_pretrained(
        args.checkpoint,
        ignore_mismatched_sizes=False,
        tasks=args.tasks,
        device_map=args.device,
    )

    scene = dataset.get_scene(args.scene_name)
    for idx in range(0, len(dataset.frame_ids), args.batch_size):
        frame_ids = dataset.frame_ids[idx:idx+args.batch_size]
        prepared_frames = None
        rgb_frames = []
        for view in views_to_load:
            for frame_id in frame_ids:
                frame = scene.get_frame(frame_id)
                camera_frame = frame.get_camera_frame(view)
                rgb_frames.append(camera_frame.image)
                processed_frame = {k: np.array(v) for k, v in camera_frame.processed_frame.items()}
                if prepared_frames is None:
                    prepared_frames = processed_frame
                else:
                    prepared_frames = {k: np.concatenate((v, processed_frame[k]), axis=0) for k, v in prepared_frames.items()}
            prepared_frames = {k: torch.tensor(v).to(args.device) for k, v in prepared_frames.items()}
            outputs = model(**prepared_frames)

            view_path = Path(args.output) / args.scene_name / view

            rgb_path = view_path / "img"
            for i, rgb_frame in enumerate(rgb_frames):
                out_path = rgb_path / f"{frame_ids[i]}_img_{view}.png"
                out_path.parent.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(out_path), cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))

            if hasattr(outputs, "logits"):
                semseg_path = view_path / "semseg"
                semseg_path.mkdir(exist_ok=True, parents=True)
                logits_tensor = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=prepared_frames["pixel_values"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).argmax(dim=1)

                pred_labels = logits_tensor.detach().cpu().numpy()

                if model.config.do_reduce_labels:
                    pred_labels += 1
                    pred_labels[pred_labels == 256] = 0

                for i, pred_label in enumerate(pred_labels):
                    out_path = semseg_path / f"{frame_ids[i]}_semseg_{view}.png"
                    out_path.parent.mkdir(exist_ok=True, parents=True)
                    cv2.imwrite(str(out_path), pred_label)

            if hasattr(outputs, "depth_pred"):
                depth_path = view_path / "depth"
                depth_path.mkdir(exist_ok=True, parents=True)
                depth = np.exp(outputs.depth_pred.detach().cpu().numpy())
                for i, depth_map in enumerate(depth):
                    np.save(file=depth_path / f"{frame_ids[i]}_depth_{view}", arr=depth_map)



if __name__ == "__main__":
    parser = ArgumentParser(description="Use model checkpoint to generate inference")
    parser.add_argument("-d", "--data-root", type=str, default="E:/shift", help="Path to SHIFT dataset.")
    parser.add_argument("-s", "--split", type=str, default="val", help="Split to load for inference.")
    parser.add_argument("-v", "--views", nargs="*", type=str, default="front", help="Views to load for inference.")
    parser.add_argument("-c", "--checkpoint", type=str, help="Path to the model checkpoint to load.")
    parser.add_argument("-t", "--tasks", nargs="*", type=str, help="Tasks to run inference on.")
    parser.add_argument("-o", "--output", type=str, default="./inference_out", help="Path for output.")
    parser.add_argument("-fr", "--framerate", type=str, default="images", help="Framerate to load.")
    parser.add_argument("-sn", "--scene-name", type=str, default="0aee-69fd", help="Name of scene to load.")
    parser.add_argument("-bs", "--batch-size", type=int, default=4, help="Number of frames to run at once.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Pytorch device for computation.")
    parser.add_argument("-gt", "--load-gt", action="store_true", help="Flag to load ground truth for loss calculation.")

    with torch.no_grad():
        main(parser.parse_args())
