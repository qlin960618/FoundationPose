#!/usr/bin/env python3
import argparse
import os

import numpy as np
import trimesh


def parse_args():
  parser = argparse.ArgumentParser(description="Display or render an OBJ mesh with trimesh.")
  parser.add_argument("input_obj", type=str, help="Path to the OBJ file.")
  parser.add_argument("--scale", type=float, default=1.0, help="Optional scale factor to apply before viewing.")
  parser.add_argument("--save", type=str, default=None, help="Optional output image path for an offscreen render.")
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--show_ruler", action="store_true", help="Add a simple 3D scale bar next to the mesh.")
  parser.add_argument("--ruler_length", type=float, default=0.1, help="Ruler length in mesh units, typically meters.")
  return parser.parse_args()


def add_ruler(scene, bounds, length):
  min_corner, max_corner = bounds
  size = max(max_corner - min_corner)
  radius = max(size * 0.003, length * 0.02, 1e-4)
  tick_height = max(size * 0.02, length * 0.1, radius * 4)

  start = np.array([
    min_corner[0],
    min_corner[1] - tick_height * 2.0,
    min_corner[2],
  ], dtype=float)
  end = start + np.array([length, 0.0, 0.0], dtype=float)

  bar = trimesh.creation.cylinder(radius=radius, segment=np.stack([start, end], axis=0))
  bar.visual.vertex_colors = np.tile(np.array([[255, 0, 0, 255]], dtype=np.uint8), (len(bar.vertices), 1))
  scene.add_geometry(bar)

  for point in (start, end):
    tick_end = point + np.array([0.0, tick_height, 0.0], dtype=float)
    tick = trimesh.creation.cylinder(radius=radius, segment=np.stack([point, tick_end], axis=0))
    tick.visual.vertex_colors = np.tile(np.array([[0, 255, 0, 255]], dtype=np.uint8), (len(tick.vertices), 1))
    scene.add_geometry(tick)


def main():
  args = parse_args()
  input_obj = os.path.abspath(args.input_obj)
  if not os.path.exists(input_obj):
    raise SystemExit(f"Input OBJ not found: {input_obj}")

  mesh = trimesh.load(input_obj)
  if hasattr(mesh, "geometry"):
    scene = mesh
  else:
    if args.scale != 1.0:
      mesh.apply_scale(args.scale)
    scene = trimesh.Scene(mesh)

  if args.show_ruler:
    add_ruler(scene, scene.bounds, args.ruler_length)

  if args.save:
    png = scene.save_image(resolution=(args.width, args.height), visible=True)
    out_path = os.path.abspath(args.save)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
      f.write(png)
    print(f"Saved render to: {out_path}")
    return

  scene.show()


if __name__ == "__main__":
  main()
