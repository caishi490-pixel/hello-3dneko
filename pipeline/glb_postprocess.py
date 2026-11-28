"""Repair zero-length FLOAT VEC4 tangents in GLB 2.0 without changing valid entries."""

import json
import math
import struct
from pathlib import Path

from .sandbox import ShortPathWorkspace


_HEADER = struct.Struct("<4sII")
_CHUNK = struct.Struct("<I4s")


def _read_glb(source):
    data = Path(source).read_bytes()
    if len(data) < _HEADER.size or _HEADER.unpack_from(data) != (b"glTF", 2, len(data)):
        raise ValueError("Expected a complete GLB 2.0 file")
    offset = _HEADER.size
    chunks = []
    while offset < len(data):
        size, kind = _CHUNK.unpack_from(data, offset)
        offset += _CHUNK.size
        if size % 4 or offset + size > len(data):
            raise ValueError("Invalid GLB chunk length")
        chunks.append((kind, data[offset:offset + size]))
        offset += size
    if not chunks or chunks[0][0] != b"JSON":
        raise ValueError("GLB JSON chunk missing")
    return chunks


def _accessor_values(document, binary, index, width):
    accessor = document["accessors"][index]
    if accessor["componentType"] != 5126 or accessor["type"] != f"VEC{width}":
        raise ValueError(f"Expected FLOAT VEC{width} accessor {index}")
    fmt = struct.Struct("<" + "f" * width)
    values = [(0.0,) * width] * accessor["count"]
    offsets = []
    views = document.get("bufferViews", [])

    def view_data(reference, item_size, count, *, use_stride=False):
        view = views[reference["bufferView"]]
        if view.get("buffer", 0) != 0:
            raise ValueError("External buffers cannot be repaired inside a GLB")
        stride = view.get("byteStride", item_size) if use_stride else item_size
        start = view.get("byteOffset", 0) + reference.get("byteOffset", 0)
        end = start + stride * max(0, count - 1) + (item_size if count else 0)
        if stride < item_size or start < 0 or end > min(len(binary), view.get("byteOffset", 0) + view["byteLength"]):
            raise ValueError(f"GLB accessor {index} lies outside its buffer view")
        return range(start, start + stride * count, stride)

    if "bufferView" in accessor:
        offsets = list(view_data(accessor, fmt.size, len(values), use_stride=True))
        values = [fmt.unpack_from(binary, offset) for offset in offsets]
    if sparse := accessor.get("sparse"):
        index_fmt = struct.Struct({5121: "<B", 5123: "<H", 5125: "<I"}[sparse["indices"]["componentType"]])
        indices = view_data(sparse["indices"], index_fmt.size, sparse["count"])
        sparse_values = view_data(sparse["values"], fmt.size, sparse["count"])
        for index_offset, value_offset in zip(indices, sparse_values):
            values[index_fmt.unpack_from(binary, index_offset)[0]] = fmt.unpack_from(binary, value_offset)
        offsets = []
    return values, offsets


def _fallback_tangent(normal, handedness):
    length = math.hypot(*normal)
    if length == 0.0 or not math.isfinite(length):
        return (1.0, 0.0, 0.0, handedness)
    n = [v / length for v in normal]
    axis = min(range(3), key=lambda i: abs(n[i]))
    tangent = [float(i == axis) - n[axis] * v for i, v in enumerate(n)]
    return (*[v / math.hypot(*tangent) for v in tangent], handedness)


def repair_zero_tangents(source, output=None):
    source = Path(source)
    output = Path(output) if output is not None else source
    chunks = _read_glb(source)
    document = json.loads(chunks[0][1])
    binary = bytearray(next((chunk for kind, chunk in chunks if kind == b"BIN\0"), b""))
    uses = {}
    for mesh in document.get("meshes", []):
        for primitive in mesh["primitives"]:
            attributes = primitive["attributes"]
            if "TANGENT" in attributes:
                uses.setdefault(attributes["TANGENT"], attributes.get("NORMAL"))
    scanned = repaired = 0
    for index, normal_index in uses.items():
        values, offsets = _accessor_values(document, binary, index, 4)
        invalid = [i for i, (x, y, z, w) in enumerate(values)
                   if not math.isfinite(x + y + z + w) or math.hypot(x, y, z) <= 1e-8 or abs(w) <= 1e-8]
        scanned += len(values)
        if not invalid:
            continue
        normals = _accessor_values(document, binary, normal_index, 3)[0] if normal_index is not None else []
        for i in invalid:
            values[i] = _fallback_tangent(normals[i] if i < len(normals) else (0.0, 0.0, 1.0),
                                          -1.0 if values[i][3] < 0 else 1.0)
            if offsets:
                struct.pack_into("<4f", binary, offsets[i], *values[i])
        if not offsets:
            accessor = document["accessors"][index]
            views = document.setdefault("bufferViews", [])
            accessor.pop("sparse", None)
            accessor.pop("byteOffset", None)
            accessor["bufferView"] = len(views)
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": 16 * len(values), "target": 34962})
            binary.extend(b"".join(struct.pack("<4f", *value) for value in values))
        accessor = document["accessors"][index]
        for name, bound in (("min", min), ("max", max)):
            if name in accessor:
                accessor[name] = [bound(value[i] for value in values) for i in range(4)]
        repaired += len(invalid)

    if repaired:
        document["buffers"][0]["byteLength"] = len(binary)
        meta = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        meta += b" " * (-len(meta) % 4)
        out_chunks = [(b"JSON", meta), (b"BIN\0", binary)]
        out_chunks.extend((kind, data) for kind, data in chunks[1:] if kind != b"BIN\0")
        contents = b"".join(_CHUNK.pack(len(data), kind) + data for kind, data in out_chunks)
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            pending = Path(folder) / "packed.glb"
            pending.write_bytes(_HEADER.pack(b"glTF", 2, _HEADER.size + len(contents)) + contents)
            ShortPathWorkspace.publish_file(pending, output)
    elif source.resolve() != output.resolve():
        ShortPathWorkspace.publish_file(source, output)
    return {"tangents_scanned": scanned, "zero_tangents_repaired": repaired, "zero_tangents_remaining": 0}
