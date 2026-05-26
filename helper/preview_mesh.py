#!/usr/bin/env python3
import argparse
import os

from PIL import Image
import trimesh


def load_mesh(mesh_file: str):
  mesh_file = os.path.abspath(mesh_file)
  mesh = trimesh.load(mesh_file, process=False)

  # Match the repo's YCB-style mesh loading path: a .ply may carry UVs while
  # the texture image sits next to it as a .png.
  if isinstance(mesh, trimesh.Trimesh) and mesh_file.lower().endswith(".ply"):
    tex_file = os.path.splitext(mesh_file)[0] + ".png"
    if os.path.exists(tex_file) and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
      image = Image.open(tex_file)
      material = trimesh.visual.texture.SimpleMaterial(image=image)
      mesh.visual = trimesh.visual.TextureVisuals(
        uv=mesh.visual.uv,
        image=image,
        material=material,
      )

  return mesh


def flatten_to_single_mesh(mesh_or_scene):
  if isinstance(mesh_or_scene, trimesh.Trimesh):
    return mesh_or_scene

  if isinstance(mesh_or_scene, trimesh.Scene):
    meshes = [geom for geom in mesh_or_scene.dump() if isinstance(geom, trimesh.Trimesh)]
    if not meshes:
      raise SystemExit("No mesh geometry found in scene")
    return trimesh.util.concatenate(meshes)

  raise SystemExit(f"Unsupported geometry type: {type(mesh_or_scene).__name__}")


def exterior_shell(mesh: trimesh.Trimesh, pitch: float):
  if pitch <= 0:
    raise SystemExit("--pitch must be > 0")

  # Convert the mesh into a filled voxel volume, then extract only the outer
  # surface. This removes internal geometry, but the result is approximate and
  # depends on the voxel pitch.
  voxels = mesh.voxelized(pitch)
  filled = voxels.fill()
  shell = filled.marching_cubes
  return shell


def solid_from_voxels(mesh: trimesh.Trimesh, pitch: float):
  if pitch <= 0:
    raise SystemExit("--pitch must be > 0")

  # This fills only enclosed volume. If the mesh is an open container, the
  # inside is still connected to the outside, so it will not become solid
  # unless the opening is capped first.
  return mesh.voxelized(pitch).fill().marching_cubes


def main():
  parser = argparse.ArgumentParser(description="Preview a mesh and its material in an interactive window.")
  parser.add_argument("mesh_file", type=str, help="Path to mesh file, for example .obj or .ply")
  parser.add_argument("--merge", action="store_true", help="Flatten a multi-mesh scene into one mesh")
  parser.add_argument("--convex-hull", action="store_true", help="Preview only the convex hull of the merged mesh")
  parser.add_argument("--outer-shell", action="store_true", help="Remove internal structure and keep only the exterior shell")
  parser.add_argument("--solid", action="store_true", help="Fill enclosed volume and preview the resulting solid")
  parser.add_argument("--pitch", type=float, default=0.002, help="Voxel size for --outer-shell or --solid, in mesh units")
  parser.add_argument("--export", type=str, default=None, help="Optional output path for the processed mesh")
  args = parser.parse_args()

  if not os.path.exists(args.mesh_file):
    raise SystemExit(f"Mesh not found: {os.path.abspath(args.mesh_file)}")

  processed = load_mesh(args.mesh_file)
  if args.merge or args.convex_hull or args.outer_shell or args.solid:
    processed = flatten_to_single_mesh(processed)
  if args.convex_hull:
    processed = processed.convex_hull
  if args.solid:
    processed = solid_from_voxels(processed, args.pitch)
  if args.outer_shell:
    processed = exterior_shell(processed, args.pitch)

  if args.export:
    out_path = os.path.abspath(args.export)
    out_dir = os.path.dirname(out_path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    processed.export(out_path)
    print(f"Saved mesh to: {out_path}")

  if isinstance(processed, trimesh.Scene):
    scene = processed
  else:
    scene = trimesh.Scene(processed)

  scene.show()


if __name__ == "__main__":
  main()
