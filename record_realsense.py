#!/usr/bin/env python3
import argparse
import os
import time

import cv2
import numpy as np


DEFAULT_OUT_DIR = "/mnt/e/Github/FoundationPose/demo_data/realsense_capture"
DEFAULT_REALSENSE_PRESET = "/mnt/e/Github/FoundationPose/preset/real_sense.json"


class RealSenseRecorder:
  def __init__(self, width, height, fps, preset_path=None):
    try:
      import pyrealsense2 as rs
    except ModuleNotFoundError as e:
      raise SystemExit(
        "pyrealsense2 is required for record_realsense.py. "
        "Please run this script in the environment where RealSense SDK is installed."
      ) from e

    self.rs = rs
    self.pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = self.pipeline.start(config)
    self._load_preset(profile.get_device(), preset_path)
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

    color_bgr = np.asanyarray(color_frame.get_data())
    depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
    depth_m[depth_m < 0.001] = 0
    return True, color_bgr, depth_m

  def release(self):
    self.pipeline.stop()


def parse_args():
  parser = argparse.ArgumentParser(
    description="Record an aligned RGB-D sequence from RealSense in run_demo.py format."
  )
  parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--fps", type=int, default=30)
  parser.add_argument("--max_frames", type=int, default=0, help="0 means unlimited until you press q.")
  parser.add_argument("--realsense_preset", type=str, default=DEFAULT_REALSENSE_PRESET)
  return parser.parse_args()


def ensure_output_dirs(out_dir):
  os.makedirs(out_dir, exist_ok=True)
  for subdir in ("rgb", "depth", "masks"):
    os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)


def save_frame(out_dir, frame_idx, color_bgr, depth_m):
  stem = f"{frame_idx:06d}"
  rgb_path = os.path.join(out_dir, "rgb", f"{stem}.png")
  depth_path = os.path.join(out_dir, "depth", f"{stem}.png")
  cv2.imwrite(rgb_path, color_bgr)
  depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
  cv2.imwrite(depth_path, depth_mm)


def select_init_mask(frame_bgr):
  roi = cv2.selectROI("Select object ROI", frame_bgr, showCrosshair=True, fromCenter=False)
  cv2.destroyWindow("Select object ROI")
  x, y, w, h = [int(v) for v in roi]
  if w <= 0 or h <= 0:
    raise SystemExit("No ROI selected. Please rerun and draw a box around the object.")

  mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
  mask[y:y + h, x:x + w] = 255
  return mask


def depth_to_display(depth_m, max_depth_m=2.0):
  depth_vis = np.clip(depth_m, 0.0, max_depth_m)
  depth_vis = depth_vis / max(max_depth_m, 1e-6)
  depth_vis = (depth_vis * 255.0).astype(np.uint8)
  depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
  return depth_vis


def main():
  args = parse_args()
  ensure_output_dirs(args.out_dir)

  recorder = RealSenseRecorder(
    width=args.width,
    height=args.height,
    fps=args.fps,
    preset_path=args.realsense_preset,
  )

  np.savetxt(os.path.join(args.out_dir, "cam_K.txt"), recorder.K.reshape(3, 3))
  print(f"Saving sequence to: {args.out_dir}")
  print("Controls: draw ROI on the first frame, then press ENTER or SPACE to confirm; q = stop recording")

  ok, color_bgr, depth_m = recorder.read()
  if not ok:
    recorder.release()
    raise SystemExit("Failed to read the first frame from RealSense")

  mask = select_init_mask(color_bgr)
  cv2.imwrite(os.path.join(args.out_dir, "masks", "000000.png"), mask)
  save_frame(args.out_dir, 0, color_bgr, depth_m)

  frame_idx = 1
  started = time.time()

  try:
    while True:
      ok, color_bgr, depth_m = recorder.read()
      if not ok:
        raise SystemExit("Failed to read frame from RealSense")

      save_frame(args.out_dir, frame_idx, color_bgr, depth_m)

      preview = color_bgr.copy()
      cv2.putText(
        preview,
        f"recording frame {frame_idx:06d}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
      )
      cv2.imshow("record_realsense_rgb", preview)
      cv2.imshow("record_realsense_depth", depth_to_display(depth_m))

      frame_idx += 1
      if args.max_frames > 0 and frame_idx >= args.max_frames:
        break
      if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break
  except KeyboardInterrupt:
    pass
  finally:
    recorder.release()
    cv2.destroyAllWindows()

  elapsed = max(time.time() - started, 1e-6)
  print(f"Recorded {frame_idx} frames in {elapsed:.2f}s")
  print("Run demo with:")
  print(f"python run_demo.py --test_scene_dir {args.out_dir}")


if __name__ == "__main__":
  main()
