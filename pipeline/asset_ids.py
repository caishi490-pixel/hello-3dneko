"""Stable part identifiers shared by profile reports and atlas face ranges."""

from hashlib import sha256
import re


def mesh_part_id(object_name):
    slug = re.sub(r"[^a-z0-9]+", "_", object_name.lower()).strip("_")[:32] or "mesh"
    digest = sha256(object_name.encode("utf-8")).hexdigest()[:10]
    return f"part_{slug}_{digest}"
