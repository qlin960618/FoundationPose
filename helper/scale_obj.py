#!/usr/bin/env python3
import argparse
import os

import trimesh


def parse_args():
  parser = argparse.ArgumentParser(description="Scale an OBJ mesh by a constant factor.")
  parser.add_argument("input_obj", type=str, help="Path to the input OBJ file.")
  parser.add_argument("--scale", type=float, required=True, help="Scale factor to apply to vertex coordinates.")
  parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Optional output OBJ path. Defaults to <input_stem>_scaled.obj",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  input_obj = os.path.abspath(args.input_obj)
  if not os.path.exists(input_obj):
    raise SystemExit(f"Input OBJ not found: {input_obj}")

  output_obj = args.output
  if output_obj is None:
    stem, _ = os.path.splitext(input_obj)
    output_obj = f"{stem}_scaled.obj"
  output_obj = os.path.abspath(output_obj)

  mesh = trimesh.load(input_obj, force="mesh")
  print(type(mesh.visual).__name__)
  mesh.apply_scale(args.scale)
  os.makedirs(os.path.dirname(output_obj), exist_ok=True)
  mesh.export(output_obj)

  print(f"Input:  {input_obj}")
  print(f"Scale:  {args.scale}")
  print(f"Output: {output_obj}")
  print(f"Extents after scaling: {mesh.extents}")


if __name__ == "__main__":
  main()
