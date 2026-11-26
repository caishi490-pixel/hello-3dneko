# SPDX-License-Identifier: GPL-3.0-or-later
"""Fingerprint raw Blender mesh data and scene relationships without editing them."""

import argparse
import hashlib
import json
import os
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import bpy
import numpy as np


def script_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def digest(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def put_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def save_new_blend(path: Path) -> None:
    """Publish a complete file, atomically refusing existing destinations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".gre-", dir=path.parent) as staging:
        staged = Path(staging) / path.name
        status = bpy.ops.wm.save_as_mainfile(
            filepath=str(staged), check_existing=False, compress=True, relative_remap=False, copy=True,
        )
        if status != {"FINISHED"}:
            raise RuntimeError(f"Save failed: {path}")
        os.link(staged, path)


def mesh_object(name: str) -> bpy.types.Object:
    obj = bpy.context.scene.objects.get(name)
    if obj is None or obj.type != "MESH":
        raise ValueError(f"Mesh object not found in the active scene: {name}")
    return obj


def stable_ids(mesh: bpy.types.Mesh, name: str, domain: str) -> np.ndarray:
    attribute = mesh.attributes.get(name)
    if attribute is None or (attribute.data_type, attribute.domain) != ("INT", domain):
        raise ValueError(f"Expected integer {domain} attribute {name}")
    ids = np.empty(len(attribute.data), dtype=np.int32)
    attribute.data.foreach_get("value", ids)
    if np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} must contain unique positive IDs")
    return ids


def coordinates(mesh: bpy.types.Mesh) -> np.ndarray:
    values = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", values)
    return values.reshape(-1, 3)


def float_hash(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()


def fingerprint(obj: bpy.types.Object) -> dict:
    mesh = obj.data
    vid = stable_ids(mesh, "gre_vid", "POINT").tolist()
    fid = stable_ids(mesh, "gre_fid", "FACE").tolist()
    topology = {
        "vertices": vid,
        "edges": [[vid[index] for index in edge.vertices] for edge in mesh.edges],
        "faces": [[face_id, [vid[index] for index in face.vertices]] for face_id, face in zip(fid, mesh.polygons)],
        "loops": [vid[loop.vertex_index] for loop in mesh.loops],
    }
    layers = []
    for layer in mesh.uv_layers:
        values = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        layer.uv.foreach_get("vector", values)
        layers.append((layer.name, layer.active_render, float_hash(values)))
    scene = bpy.context.scene
    relations = [
        {
            "name": item.name,
            "type": item.type,
            "parent": item.parent.name if item.parent else None,
            "parent_type": item.parent_type,
            "parent_bone": item.parent_bone,
            "collections": sorted(collection.name for collection in item.users_collection),
            "matrix_world": [list(row) for row in item.matrix_world],
        }
        for item in sorted(scene.objects, key=lambda item: item.name)
    ]
    materials = {
        "slots": [(slot.link, slot.material.name if slot.material else None) for slot in obj.material_slots],
        "faces": [face.material_index for face in mesh.polygons],
    }
    result = {
        "object": obj.name,
        "vertex_count": len(vid),
        "edge_count": len(mesh.edges),
        "face_count": len(fid),
        "loop_count": len(mesh.loops),
        "uv_layers": [layer.name for layer in mesh.uv_layers],
        "unit_system": scene.unit_settings.system,
        "unit_scale": scene.unit_settings.scale_length,
        "matrix_world": [list(row) for row in obj.matrix_world],
        "topology_hash": digest(topology),
        "uv_hash": digest({"active_index": mesh.uv_layers.active_index, "layers": layers}),
        "material_hash": digest(materials),
        "object_relationship_hash": digest(relations),
        "position_hash": float_hash(coordinates(mesh)),
    }
    result["source_hash"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="Table")
    args = parser.parse_args(script_args())
    print(json.dumps(fingerprint(mesh_object(args.object)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
