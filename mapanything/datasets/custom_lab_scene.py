from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from mapanything.datasets.base.base_dataset import BaseDataset


class LabSceneDataset(BaseDataset):
    """Minimal multi-view dataset that serves scenes exported by init_geo_original.py."""

    def __init__(
        self,
        root: str,
        scene: str = "sparse_12/0",
        image_dir: str = "imgs_12",
        mask_dir: str = "overlapping_masks_12",
        points_file: str = "points3D_all.npy",
        confidence_file: str = "confidence.npy",
        use_masks: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.root = Path(root)
        self.scene = scene
        self.scene_path = self.root / scene
        self.image_dir = self.scene_path / image_dir
        self.mask_dir = self.scene_path / mask_dir
        self.points_path = self.scene_path / points_file
        self.conf_path = self.scene_path / confidence_file
        self.use_masks = use_masks

        if not self.points_path.exists():
            raise FileNotFoundError(f"Could not find dense point map at {self.points_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Could not find images at {self.image_dir}")
        if self.use_masks and not self.mask_dir.exists():
            raise FileNotFoundError(f"Could not find overlapping masks at {self.mask_dir}")

        self._load_data()

    # ------------------------------------------------------------------
    # Scene parsing helpers
    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        # Load dense pointmaps & confidences saved by init_geo_original.py
        self.pointmaps = np.load(self.points_path)  # (N, H, W, 3)
        self.confidence = None
        if self.conf_path.exists():
            try:
                conf = np.load(self.conf_path)
                if conf.ndim == 2:
                    conf = conf.reshape(self.pointmaps.shape[0], -1)
                    conf = conf.reshape(self.pointmaps.shape[0], *self.pointmaps.shape[1:3])
                self.confidence = conf.astype(np.float32)
            except Exception:
                self.confidence = None

        self.image_names = sorted([p.name for p in self.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        if len(self.image_names) != self.pointmaps.shape[0]:
            raise RuntimeError(
                f"Mismatch between point maps ({self.pointmaps.shape[0]}) and images ({len(self.image_names)})"
            )

        # Precompute camera intrinsics and extrinsics from COLMAP sparse model
        cameras = self._parse_cameras(self.scene_path / "cameras.txt")
        self.poses, self.intrinsics = self._parse_images(
            self.scene_path / "images.txt", cameras
        )
        if len(self.poses) != len(self.image_names):
            raise RuntimeError(
                "Number of camera poses does not match number of images."
            )

        # Preload masks if requested
        self.masks: Dict[str, np.ndarray] = {}
        if self.use_masks:
            for name in self.image_names:
                mask_path = self.mask_dir / name
                if mask_path.exists():
                    mask = np.array(Image.open(mask_path).convert("L")) > 0
                else:
                    mask = np.ones(self.pointmaps.shape[1:3], dtype=bool)
                self.masks[name] = mask

        self.num_of_scenes = 1
        self.scenes = [self.scene]

    @staticmethod
    def _parse_cameras(camera_file: Path) -> Dict[int, Dict[str, float]]:
        cameras: Dict[int, Dict[str, float]] = {}
        with open(camera_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                elems = line.split()
                cam_id = int(elems[0])
                model = elems[1]
                width = float(elems[2])
                height = float(elems[3])
                params = list(map(float, elems[4:]))
                if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
                    fx = fy = params[0]
                    cx = params[1]
                    cy = params[2]
                elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE"}:
                    fx, fy, cx, cy = params[:4]
                else:
                    raise NotImplementedError(f"Unsupported COLMAP camera model: {model}")
                cameras[cam_id] = {
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height,
                }
        return cameras

    @staticmethod
    def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
        q1, q2, q3, q4 = qvec
        return np.array(
            [
                [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q4 * q1), 2 * (q2 * q4 + q3 * q1)],
                [2 * (q2 * q3 + q4 * q1), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q2 * q1)],
                [2 * (q2 * q4 - q3 * q1), 2 * (q3 * q4 + q2 * q1), 1 - 2 * (q2 * q2 + q3 * q3)],
            ],
            dtype=np.float32,
        )

    def _parse_images(
        self, images_file: Path, cameras: Dict[int, Dict[str, float]]
    ) -> (List[np.ndarray], List[np.ndarray]):
        poses = []
        intrinsics = []
        names = []
        with open(images_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        idx = 0
        while idx < len(lines):
            pose_line = lines[idx]
            elems = pose_line.split()
            qvec = np.array(list(map(float, elems[1:5])), dtype=np.float32)
            tvec = np.array(list(map(float, elems[5:8])), dtype=np.float32)
            cam_id = int(elems[8])
            name = Path(elems[9]).name
            names.append(name)
            R_wc = self._qvec2rotmat(qvec)  # world to camera
            t_wc = tvec.reshape(3, 1)
            R_cw = R_wc.T
            C = -R_cw @ t_wc
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R_cw
            pose[:3, 3:4] = C
            poses.append(pose)
            cam_params = cameras[cam_id]
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = cam_params["fx"]
            K[1, 1] = cam_params["fy"]
            K[0, 2] = cam_params["cx"]
            K[1, 2] = cam_params["cy"]
            intrinsics.append(K)
            idx += 1
            if idx < len(lines):
                next_tokens = lines[idx].split()
                first_token = next_tokens[0]
                if any(ch in first_token for ch in ".eE-") and not first_token.isdigit():
                    idx += 1
        name_to_idx = {name: i for i, name in enumerate(names)}
        reordered_poses = [poses[name_to_idx[name]] for name in self.image_names]
        reordered_intrinsics = [intrinsics[name_to_idx[name]] for name in self.image_names]
        return reordered_poses, reordered_intrinsics

    def _sample_view_indices(self, num_views_to_sample, num_views_in_scene, *_):
        rng = np.random.default_rng()
        if num_views_to_sample >= num_views_in_scene:
            indices = np.arange(num_views_in_scene)
            rng.shuffle(indices)
            return indices[:num_views_to_sample]
        return rng.choice(num_views_in_scene, size=num_views_to_sample, replace=False)

    def _get_views(self, idx, num_views_to_sample, resolution):
        selected = self._sample_view_indices(num_views_to_sample, len(self.image_names))
        views = []
        for vid in selected:
            name = self.image_names[vid]
            with Image.open(self.image_dir / name) as pil_img:
                img = pil_img.convert("RGB")
            pts = self.pointmaps[vid].astype(np.float32)
            depth = np.linalg.norm(pts, axis=-1).astype(np.float32)
            mask = None
            if self.use_masks:
                mask = self.masks.get(name)
            view = dict(
                img=img,
                depthmap=depth,
                camera_pose=self.poses[vid],
                camera_intrinsics=self.intrinsics[vid],
                dataset="LabScene",
                label=self.scene,
                instance=str(name),
            )
            if mask is not None:
                view["non_ambiguous_mask"] = mask.astype(np.uint8)
            if self.confidence is not None:
                view["confidence"] = self.confidence[vid]
            views.append(view)
        return views


def build_lab_scene_dataset(**kwargs) -> LabSceneDataset:
    return LabSceneDataset(**kwargs)
