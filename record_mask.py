#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np


DEFAULT_SCENE_DIR = "/mnt/e/Github/FoundationPose/demo_data/realsense_capture"


class MaskEditor:
  def __init__(self, image_bgr, init_mask=None, brush_radius=8, display_scale=1.0):
    self.image_bgr = image_bgr
    self.mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8) if init_mask is None else init_mask.copy()
    self.brush_radius = max(1, int(brush_radius))
    self.display_scale = max(1.0, float(display_scale))
    self.mode = "fg"
    self.drawing = False
    self.window = "record_mask"

  def on_mouse(self, event, x, y, flags, param):
    x = int(round(x / self.display_scale))
    y = int(round(y / self.display_scale))
    x = int(np.clip(x, 0, self.image_bgr.shape[1] - 1))
    y = int(np.clip(y, 0, self.image_bgr.shape[0] - 1))
    if event == cv2.EVENT_LBUTTONDOWN:
      self.drawing = True
      self._paint(x, y)
    elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
      self._paint(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
      self.drawing = False
      self._paint(x, y)

  def _paint(self, x, y):
    value = 255 if self.mode == "fg" else 0
    cv2.circle(self.mask, (x, y), self.brush_radius, value, thickness=-1, lineType=cv2.LINE_AA)

  def make_overlay(self):
    overlay = self.image_bgr.copy()
    fg = self.mask > 0
    overlay[fg] = (0.35 * overlay[fg] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    label = f"mode:{self.mode} brush:{self.brush_radius}px  keys: f=fg e=erase [ ]=brush c=clear s=save q=quit"
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if self.display_scale > 1.0:
      overlay = cv2.resize(
        overlay,
        dsize=None,
        fx=self.display_scale,
        fy=self.display_scale,
        interpolation=cv2.INTER_LINEAR,
      )
    return overlay


def parse_args():
  parser = argparse.ArgumentParser(description="Paint a pixel mask for the first frame of a recorded scene.")
  parser.add_argument("--scene_dir", type=str, default=DEFAULT_SCENE_DIR)
  parser.add_argument("--frame", type=str, default="000000")
  parser.add_argument("--brush", type=int, default=8)
  parser.add_argument("--display_scale", type=float, default=1.5)
  return parser.parse_args()


def main():
  args = parse_args()
  frame_name = args.frame if args.frame.endswith(".png") else f"{args.frame}.png"
  rgb_path = os.path.join(args.scene_dir, "rgb", frame_name)
  mask_path = os.path.join(args.scene_dir, "masks", frame_name)

  if not os.path.exists(rgb_path):
    raise SystemExit(f"RGB frame not found: {rgb_path}")

  image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
  if image_bgr is None:
    raise SystemExit(f"Failed to read image: {rgb_path}")

  init_mask = None
  if os.path.exists(mask_path):
    init_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if init_mask is None:
      raise SystemExit(f"Failed to read existing mask: {mask_path}")
    init_mask = (init_mask > 0).astype(np.uint8) * 255

  editor = MaskEditor(
    image_bgr=image_bgr,
    init_mask=init_mask,
    brush_radius=args.brush,
    display_scale=args.display_scale,
  )
  cv2.namedWindow(editor.window, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(
    editor.window,
    int(image_bgr.shape[1] * editor.display_scale),
    int(image_bgr.shape[0] * editor.display_scale),
  )
  cv2.setMouseCallback(editor.window, editor.on_mouse)

  print(f"Editing mask for: {rgb_path}")
  print("Controls:")
  print("  Left mouse: paint")
  print("  f: foreground mode")
  print("  e: erase mode")
  print("  [: smaller brush")
  print("  ]: larger brush")
  print("  c: clear mask")
  print("  s: save")
  print("  q or ESC: quit")

  while True:
    cv2.imshow(editor.window, editor.make_overlay())
    key = cv2.waitKey(20) & 0xFF
    if key == ord("f"):
      editor.mode = "fg"
    elif key == ord("e"):
      editor.mode = "erase"
    elif key == ord("["):
      editor.brush_radius = max(1, editor.brush_radius - 1)
    elif key == ord("]"):
      editor.brush_radius += 1
    elif key == ord("c"):
      editor.mask[:] = 0
    elif key == ord("s"):
      os.makedirs(os.path.dirname(mask_path), exist_ok=True)
      cv2.imwrite(mask_path, editor.mask)
      print(f"Saved mask to: {mask_path}")
    elif key in (ord("q"), 27):
      break

  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
