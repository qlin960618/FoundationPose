#!/usr/bin/env python3
import argparse
import importlib
import os
import sys

import cv2
import numpy as np
import torch
import trimesh

from estimater import *


DA2_MODEL_CONFIGS = {
  "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
  "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
  "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
  "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

DEFAULT_DA2_REPO = "/mnt/e/Github/FoundationPose/DepthAnything"
# DEFAULT_DA2_CHECKPOINT = "/mnt/e/Github/FoundationPose/weights/DepthAnything/depth_anything_v2_vitb.pth"  # relative-depth checkpoint
DEFAULT_DA2_CHECKPOINT = "/mnt/e/Github/FoundationPose/weights/DepthAnything/depth_anything_v2_metric_hypersim_vitb.pth" # metric-depth checkpoint
DEFAULT_DA2_ENCODER = "vitb"
DEFAULT_REALSENSE_PRESET = "/mnt/e/Github/FoundationPose/preset/real_sense.json"
DEPTH_DISPLAY_RANGE_M = (0.02, 1.00)
# DEFAULT_MESH_FILE = "/demo_data/pipette_300ul/mesh/sartoriusPicus2_m.obj"
# DEFAULT_MESH_FILE = "/demo_data/bottle/mesh/square_bottle_and_cap_flatten_m.obj"
DEFAULT_MESH_FILE = "/demo_data/bottle/mesh/suquarebottlecapwithjig_reoriented.obj"


class OpenCVCapture:
  def __init__(self, args):
    self.cap = cv2.VideoCapture(args.cam_index)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not self.cap.isOpened():
      raise SystemExit(f"Failed to open webcam at index {args.cam_index}")
    self.args = args
    self.K = None

  def read(self):
    ok, frame_bgr = self.cap.read()
    if not ok:
      return False, None, None
    if self.K is None:
      H, W = frame_bgr.shape[:2]
      self.K = build_camera_matrix(W, H, fx=self.args.fx, fy=self.args.fy, cx=self.args.cx, cy=self.args.cy)
    return True, frame_bgr, None

  def release(self):
    self.cap.release()


class RealSenseCapture:
  def __init__(self, args):
    try:
      import pyrealsense2 as rs
    except ModuleNotFoundError as e:
      raise SystemExit(
        "RealSense SDK path requested, but pyrealsense2 is not installed.\n"
        "Install it in this environment, then rerun.\n"
        "Example: python -m pip install pyrealsense2"
      ) from e

    self.rs = rs
    self.pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, 30)
    profile = self.pipeline.start(config)
    self._load_preset(profile.get_device(), args.realsense_preset)
    self.align = rs.align(rs.stream.color)
    self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    self.K = np.array([
      [intr.fx, 0.0, intr.ppx],
      [0.0, intr.fy, intr.ppy],
      [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    for _ in range(10):
      self.pipeline.wait_for_frames()

  def _load_preset(self, device, preset_path):
    if not preset_path:
      return

    preset_path = os.path.abspath(preset_path)
    if not os.path.exists(preset_path):
      raise SystemExit(f"RealSense preset file not found: {preset_path}")

    with open(preset_path, "r", encoding="utf-8") as f:
      preset_json = f.read()

    try:
      advanced_mode = self.rs.rs400_advanced_mode(device)
      advanced_mode.load_json(preset_json)
      print(f"Loaded RealSense preset: {preset_path}")
    except Exception as e:
      print(
        f"Warning: failed to load RealSense preset {preset_path}. "
        f"Continuing with current device settings. {type(e).__name__}: {e}"
      )

  def read(self):
    frames = self.pipeline.wait_for_frames()
    frames = self.align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
      return False, None, None

    frame_bgr = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
    depth[depth < 0.001] = 0
    return True, frame_bgr, depth

  def release(self):
    self.pipeline.stop()


def parse_args():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser(
    description="Minimal webcam entry point for tracking a custom object mesh with Depth Anything V2."
  )
  parser.add_argument(
    "--mesh_file",
    type=str,
    default=f"{code_dir}{DEFAULT_MESH_FILE}",
  )
  parser.add_argument("--camera_backend", type=str, default="realsense", choices=["realsense", "opencv"])
  parser.add_argument("--cam_index", type=int, default=4)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--realsense_preset", type=str, default=DEFAULT_REALSENSE_PRESET)
  parser.add_argument("--fx", type=float, default=None)
  parser.add_argument("--fy", type=float, default=None)
  parser.add_argument("--cx", type=float, default=None)
  parser.add_argument("--cy", type=float, default=None)
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument("--track_refine_iter", type=int, default=2)
  parser.add_argument("--debug", type=int, default=1)
  parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug_custom")
  parser.add_argument("--depth_backend", type=str, default="realsense", choices=["realsense", "da2_relative", "da2_metric"])
  parser.add_argument(
    "--ignore_realsense_depth",
    action="store_true",
    default=False,
    help="Use RealSense only for RGB capture and ignore native depth. If depth backend is realsense, it will switch to da2_metric.",
  )
  parser.add_argument("--da2_input_size", type=int, default=518)
  parser.add_argument("--da2_repo", type=str, default=DEFAULT_DA2_REPO, help="Path to the local Depth-Anything-V2 checkout.")
  parser.add_argument("--da2_checkpoint", type=str, default=DEFAULT_DA2_CHECKPOINT, help="Path to the Depth Anything V2 checkpoint.")
  parser.add_argument("--min_depth_m", type=float, default=0.02, help="Near depth in meters for clipping or relative-depth remapping.")
  parser.add_argument("--max_depth_m", type=float, default=2.00, help="Far depth in meters for clipping or relative-depth remapping.")
  parser.add_argument("--da2_relative_invert", action="store_true", help="Invert relative-depth output before range remapping.")
  parser.add_argument("--da2_percentile_low", type=float, default=2.0, help="Low percentile for robust relative-depth normalization.")
  parser.add_argument("--da2_percentile_high", type=float, default=98.0, help="High percentile for robust relative-depth normalization.")
  parser.add_argument("--da2_scale", type=float, default=5.0, help="Global multiplier applied to Depth Anything depth after inference/remapping.")
  parser.add_argument("--depth_ema", type=float, default=0.0, help="Temporal smoothing factor in [0,1).")
  return parser.parse_args()


def build_camera_matrix(width, height, fx=None, fy=None, cx=None, cy=None):
  if fx is None:
    fx = 0.9 * width
  if fy is None:
    fy = 0.9 * width
  if cx is None:
    cx = width / 2.0
  if cy is None:
    cy = height / 2.0

  K = np.eye(3, dtype=np.float32)
  K[0, 0] = fx
  K[1, 1] = fy
  K[0, 2] = cx
  K[1, 2] = cy
  return K


def import_depth_anything_v2(repo_dir, use_metric_depth):
  if repo_dir is None:
    raise SystemExit("Please provide --da2_repo pointing to a local Depth-Anything-V2 checkout.")

  repo_dir = os.path.abspath(repo_dir)
  module_root = os.path.join(repo_dir, "metric_depth") if use_metric_depth else repo_dir
  if not os.path.isdir(module_root):
    raise SystemExit(f"Depth Anything module path not found: {module_root}")

  if module_root not in sys.path:
    sys.path.insert(0, module_root)

  dpt_module = importlib.import_module("depth_anything_v2.dpt")
  return dpt_module.DepthAnythingV2


def load_depth_model(args):
  if args.depth_backend == "realsense":
    return None

  use_metric_depth = args.depth_backend == "da2_metric"
  DepthAnythingV2 = import_depth_anything_v2(args.da2_repo, use_metric_depth=use_metric_depth)

  da2_encoder = DEFAULT_DA2_ENCODER
  if use_metric_depth and da2_encoder == "vitg":
    raise SystemExit("Depth Anything V2 metric checkpoints do not currently support --da2_encoder vitg.")

  model_kwargs = dict(DA2_MODEL_CONFIGS[da2_encoder])
  if use_metric_depth:
    model_kwargs["max_depth"] = args.max_depth_m

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = DepthAnythingV2(**model_kwargs)
  state_dict = torch.load(args.da2_checkpoint, map_location="cpu")
  model.load_state_dict(state_dict)
  model = model.to(device).eval()
  return model


def remap_relative_depth_to_meters(depth_raw, min_depth_m, max_depth_m, percentile_low, percentile_high, invert=False):
  depth = np.asarray(depth_raw, dtype=np.float32)
  if invert:
    depth = depth.max() - depth

  lo = np.percentile(depth, percentile_low)
  hi = np.percentile(depth, percentile_high)
  if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
    return np.full(depth.shape, (min_depth_m + max_depth_m) * 0.5, dtype=np.float32)

  depth = np.clip(depth, lo, hi)
  depth = (depth - lo) / (hi - lo + 1e-6)
  depth = min_depth_m + depth * (max_depth_m - min_depth_m)
  return depth.astype(np.float32)


def depth_to_display(depth_m, min_depth_m, max_depth_m):
  min_depth_m, max_depth_m = DEPTH_DISPLAY_RANGE_M
  depth_vis = np.asarray(depth_m, dtype=np.float32)
  depth_vis = np.clip(depth_vis, 0 if min_depth_m is None else min_depth_m, max_depth_m)
  depth_vis = (depth_vis - min_depth_m) / max(max_depth_m - min_depth_m, 1e-6)
  depth_vis = (depth_vis * 255.0).astype(np.uint8)
  depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
  bar_w = 60
  bar_h = depth_vis.shape[0]
  gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
  color_bar = cv2.applyColorMap(np.repeat(gradient, bar_w, axis=1), cv2.COLORMAP_INFERNO)
  cv2.putText(color_bar, f"{max_depth_m:.2f}m", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
  cv2.putText(color_bar, f"{min_depth_m:.2f}m", (4, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
  mid_depth = 0.5 * (min_depth_m + max_depth_m)
  cv2.putText(color_bar, f"{mid_depth:.2f}m", (4, bar_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
  return np.concatenate([depth_vis, color_bar], axis=1)


def predict_depth(frame_bgr, native_depth, args, depth_model, prev_depth=None):
  if args.depth_backend == "realsense":
    if native_depth is None:
      raise SystemExit("Depth backend is realsense, but no native depth frame was available.")
    depth = np.asarray(native_depth, dtype=np.float32)
    depth[depth > args.max_depth_m] = 0
  else:
    depth_raw = depth_model.infer_image(frame_bgr, args.da2_input_size)
    depth_raw = np.asarray(depth_raw, dtype=np.float32)
    if args.depth_backend == "da2_metric":
      depth = (depth_raw * args.da2_scale).astype(np.float32)
      depth = np.clip(depth, args.min_depth_m, args.max_depth_m).astype(np.float32)
    else:
      depth = remap_relative_depth_to_meters(
        depth_raw,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        percentile_low=args.da2_percentile_low,
        percentile_high=args.da2_percentile_high,
        invert=args.da2_relative_invert,
      )
      depth = (depth * args.da2_scale).astype(np.float32)
      depth = np.clip(depth, args.min_depth_m, args.max_depth_m).astype(np.float32)

  if prev_depth is not None and 0.0 < args.depth_ema < 1.0:
    depth = (args.depth_ema * prev_depth + (1.0 - args.depth_ema) * depth).astype(np.float32)

  return depth


def select_init_mask(frame_bgr):
  roi = cv2.selectROI("Select object ROI", frame_bgr, showCrosshair=True, fromCenter=False)
  cv2.destroyWindow("Select object ROI")
  x, y, w, h = [int(v) for v in roi]
  if w <= 0 or h <= 0:
    raise SystemExit("No ROI selected. Please rerun and draw a box around the object.")

  mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
  mask[y:y + h, x:x + w] = 1
  return mask


def main():
  args = parse_args()

  if not os.path.exists(args.mesh_file):
    raise SystemExit(f"Mesh file not found: {args.mesh_file}")
  if args.max_depth_m <= args.min_depth_m:
    raise SystemExit("--max_depth_m must be greater than --min_depth_m")
  if args.ignore_realsense_depth and args.camera_backend != "realsense":
    raise SystemExit("--ignore_realsense_depth requires --camera_backend realsense")
  if args.ignore_realsense_depth and args.depth_backend == "realsense":
    args.depth_backend = "da2_metric"
    print("Ignoring native RealSense depth and using Depth Anything metric depth on RealSense RGB.")

  set_logging_format()
  set_seed(0)
  depth_model = load_depth_model(args)
  if args.camera_backend == "realsense":
    capture = RealSenseCapture(args)
  else:
    capture = OpenCVCapture(args)

  mesh = trimesh.load(args.mesh_file)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=args.debug_dir,
    debug=args.debug,
    glctx=glctx,
  )
  logging.info("estimator initialization done")

  ok, frame_bgr, native_depth = capture.read()
  if not ok:
    capture.release()
    raise SystemExit("Failed to read the first frame from camera")

  K = capture.K
  mask = select_init_mask(frame_bgr)

  frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
  depth = predict_depth(frame_bgr, native_depth, args, depth_model)
  pose = est.register(
    K=K,
    rgb=frame_rgb,
    depth=depth,
    ob_mask=mask,
    iteration=args.est_refine_iter,
  )

  print("Tracking started. Press 'q' to exit.")
  depth_prev = depth

  try:
    while True:
      ok, frame_bgr, native_depth = capture.read()
      if not ok:
        raise SystemExit("Failed to read frame from camera")

      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      depth = predict_depth(frame_bgr, native_depth, args, depth_model, prev_depth=depth_prev)
      depth_prev = depth
      pose = est.track_one(
        rgb=frame_rgb,
        depth=depth,
        K=K,
        iteration=args.track_refine_iter,
      )

      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(K, img=frame_rgb, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=max(mesh.extents.max() * 0.5, 0.03),
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
      )
      vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
      depth_vis = depth_to_display(depth, args.min_depth_m, args.max_depth_m)
      cv2.putText(
        vis_bgr,
        f"depth backend: {args.depth_backend}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
      )
      cv2.imshow("track_custom", vis_bgr)
      cv2.imshow("track_custom_depth", depth_vis)
      if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break
  except KeyboardInterrupt:
    pass
  finally:
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
