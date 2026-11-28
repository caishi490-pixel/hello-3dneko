"""Blender worker helpers for compute devices and export hygiene."""

import bpy


CYCLES_DEVICES = ("OPTIX", "CUDA", "HIP", "ONEAPI", "CPU")


def _enable_gpu(preferences, backend):
    try:
        preferences.compute_device_type = backend
        preferences.refresh_devices()
        devices = list(preferences.devices)
    except (TypeError, RuntimeError, ValueError):
        return False
    for device in devices:
        device.use = device.type == backend
    return any(device.use for device in devices)


def cycles_devices(preferences):
    if preferences is None:
        return ["CPU"]
    previous = preferences.compute_device_type
    try:
        return [backend for backend in CYCLES_DEVICES[:-1] if _enable_gpu(preferences, backend)] + ["CPU"]
    finally:
        preferences.compute_device_type = previous
        preferences.refresh_devices()


def configure_cycles(scene, preferences=None, *, preferred="AUTO"):
    scene.render.engine = "CYCLES"
    if preferences is None:
        addon = bpy.context.preferences.addons.get("cycles")
        preferences = addon.preferences if addon else None
    scene.cycles.device = "CPU"
    order = CYCLES_DEVICES[:-1]
    if preferred in order:
        order = (preferred, *(device for device in order if device != preferred))
    if preferences is not None and preferred != "CPU":
        for backend in order:
            if _enable_gpu(preferences, backend):
                scene.cycles.device = "GPU"
                return backend
    return "CPU"


def junk_objects(objects):
    objects = list(objects)
    shapes = {bone.custom_shape for obj in objects if obj.type == "ARMATURE" for bone in obj.pose.bones}
    return [
        obj for obj in objects
        if obj in shapes or (
            obj.type == "MESH" and obj.parent is None
            and obj.name.split(".")[0].casefold() in {"icosphere", "棱角球"}
            and len(obj.data.vertices) <= 128 and not obj.data.materials
        )
    ]


def remove_export_junk():
    junk = junk_objects(bpy.context.scene.objects)
    names = [obj.name for obj in junk]
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            for bone in obj.pose.bones:
                if bone.custom_shape in junk:
                    bone.custom_shape = None
    for obj in junk:
        bpy.data.objects.remove(obj, do_unlink=True)
    return names
