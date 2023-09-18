from argparse import ArgumentParser
from pathlib import Path

import cv2

import numpy as np
import torch
from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend

from shift_lab.semantic_segmentation.segformer.constants import SegformerTask
from shift_lab.semantic_segmentation.segformer.trainer import MultitaskSegformer


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
        for view in views_to_load:
            for frame_id in frame_ids:
                frame = scene.get_frame(frame_id)
                camera_frame = frame.get_camera_frame(view)
                processed_frame = {k: np.array(v) for k, v in camera_frame.processed_frame.items()}
                if prepared_frames is None:
                    prepared_frames = processed_frame
                else:
                    prepared_frames = {k: np.concatenate((v, processed_frame[k]), axis=0) for k, v in prepared_frames.items()}
            prepared_frames = {k: torch.tensor(v).to(args.device) for k, v in prepared_frames.items()}
            outputs = model(**prepared_frames)

            view_path = Path(args.output) / args.scene_name / view
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
                    np.save(file=depth_path / f"{frame_ids[i]}_depth_{view}", arr=depth)



if __name__ == "__main__":
    parser = ArgumentParser(description="Use model checkpoint to generate inference")
    parser.add_argument("-d", "--data-root", type=str, default="D:/shift_small", help="Path to SHIFT dataset.")
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
