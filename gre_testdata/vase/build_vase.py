# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the P4 native vase, marked/unmarked plans and reproducible input."""

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gre import blender_io as io
from gre.snapshot import write_json


N, STEPS, INNER, HEIGHT = 96, 64, 36, .260
BELT = (28, 41)


def surface(t, angle, relief=True):
    center = np.array([.014*t**1.3 + .003*np.sin(np.pi*t), -.008*t + .002*np.sin(2*np.pi*t)*t])
    radius = (.057 + .032*np.exp(-((t-.43)/.29)**2) - .014*np.exp(-((t-.82)/.14)**2)
              + .006*np.exp(-((t-1)/.09)**2)) * (1 + .042*np.cos(2*angle+.5*t))
    height = (.0042*np.exp(-((t-.531)/.049)**4) * (np.maximum(0, np.cos(12*angle+.6*np.sin(2*np.pi*t)))**3))
    radial = np.column_stack((np.cos(angle), np.sin(angle)))
    return center + (radius + (height if relief else 0))[:, None]*radial


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    angles = np.arange(N) * (2*np.pi/N)
    vertices, faces, loops = [], [], []
    regions = {name: [] for name in ('body', 'relief', 'rim', 'foot', 'inside')}

    def ring(t, outer):
        xy = surface(t, angles, relief=outer)
        if not outer:
            xy -= .006*np.column_stack((np.cos(angles), np.sin(angles)))
        indices = list(range(len(vertices), len(vertices)+N))
        vertices.extend(np.column_stack((xy, np.full(N,t*HEIGHT))).tolist())
        return indices

    def face(indices, uv, name):
        faces.append(tuple(indices))
        loops.extend(uv)
        regions[name].append(len(faces))

    def bridge(bottom, edge, name, v0, v1, *, reverse=False):
        for i in range(N):
            j = (i+1) % N
            ids = [bottom[i], bottom[j], edge[j], edge[i]]
            coords = [(.04+.92*i/N,v0),(.04+.92*(i+1)/N,v0),(.04+.92*(i+1)/N,v1),(.04+.92*i/N,v1)]
            face(ids[::-1] if reverse else ids, coords[::-1] if reverse else coords, name)

    outer = [ring(j/STEPS, True) for j in range(STEPS+1)]
    inner = [ring(t, False) for t in np.linspace(.045, 1, INNER+1)]
    for row in range(STEPS):
        name = 'foot' if row < 4 else 'rim' if row >= STEPS-5 else 'relief' if BELT[0] <= row < BELT[1] else 'body'
        bridge(outer[row], outer[row+1], name, .04+.49*row/STEPS, .04+.49*(row+1)/STEPS)
    for row in range(INNER):
        bridge(inner[row], inner[row+1], 'inside', .57+.30*row/INNER, .57+.30*(row+1)/INNER, reverse=True)
    bridge(outer[-1], inner[-1], 'rim', .90, .93)
    for ids, x, flip, name in ((outer[0], .30, True, 'foot'), (inner[0], .69, False, 'inside')):
        coords = list(zip(x+.025*np.cos(angles), .96+.025*np.sin(angles)))
        face(ids[::-1] if flip else ids, coords[::-1] if flip else coords, name)

    mesh = bpy.data.meshes.new('VaseMesh')
    mesh.from_pydata(vertices, [], faces)
    if mesh.validate(verbose=True):
        raise RuntimeError('Vase construction required geometry repair')
    uv = mesh.uv_layers.new(name='VaseUV')
    uv.uv.foreach_set('vector', np.asarray(loops, dtype=np.float32).ravel())
    for name, domain, count in (('gre_vid','POINT',len(vertices)), ('gre_fid','FACE',len(faces))):
        attr = mesh.attributes.new(name=name, type='INT', domain=domain)
        attr.data.foreach_set('value', np.arange(1,count+1,dtype=np.int32))
    obj = bpy.data.objects.new('Vase',mesh)
    scene.collection.objects.link(obj)
    glaze = bpy.data.materials.new('Blue-green porcelain')
    glaze.diffuse_color = (.055, .25, .24, 1)
    if glaze.node_tree is None:
        glaze.use_nodes = True
    node = next(x for x in glaze.node_tree.nodes if x.type == 'BSDF_PRINCIPLED')
    node.inputs['Base Color'].default_value = glaze.diffuse_color
    node.inputs['Roughness'].default_value = .26
    mesh.materials.append(glaze)
    for poly in mesh.polygons:
        poly.use_smooth = len(poly.vertices) == 4
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.context.view_layer.update()
    return obj, regions, outer


def save(path):
    with TemporaryDirectory(prefix='.vase-', dir=path.parent) as temp:
        staged = Path(temp) / path.name
        io.save_copy(staged, path, path)
        os.link(staged,path)


def preview(path):
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0,0,-.001))
    def aim(obj, xyz):
        obj.rotation_euler = (Vector(xyz)-obj.location).to_track_quat('-Z','Y').to_euler()
    bpy.ops.object.camera_add(location=(.40,-.76,.37))
    scene.camera = bpy.context.object
    scene.camera.data.type = 'ORTHO'
    scene.camera.data.ortho_scale = .345
    aim(scene.camera, (0,0,.130))
    for loc, power, size in (((-.34,-.30,.45),22,.38), ((.22,.16,.30),18,.23), ((.25,-.22,.16),5,.24)):
        light = bpy.data.lights.new('Softbox','AREA')
        light.energy, light.size = power, size
        obj = bpy.data.objects.new('Softbox',light)
        obj.location = loc
        scene.collection.objects.link(obj)
        aim(obj,(0,0,.14))
    scene.world = bpy.data.worlds.new('Studio')
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 800, 920
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(path.resolve())
    bpy.ops.render.render(write_still=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--preview',type=Path)
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    root = args.output_dir.resolve()
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):
        p.error('Output directory must be empty')
    obj, regions, outer = build()
    clean = io.read_mesh(obj)
    save(root/'vase_clean.blend')
    common = {'schema_version':1,'asset_id':'gre_vase_p4','object_name':'Vase','unit_scale':clean.unit_scale,
              'matrix_world':clean.matrix_world.tolist(),'source_hash':clean.hashes['source_hash'],
              'transition_rings':2,'max_displacement_m':.00145,'feature_angle_deg':52,
              'feature_vertex_ids':[i+1 for i in outer[-1]],'annotator':'GRE P4 procedural fixture'}
    marked = [{'name':name,'face_ids':ids,'role':'editable' if name=='body' else 'protected'} for name,ids in regions.items()]
    unmarked = [{'name':name,'face_ids':ids if name!='body' else sorted(ids+regions['relief']),
                 'role':'editable' if name=='body' else 'protected'} for name,ids in regions.items() if name!='relief']
    write_json(root/'vase_marked.json',{**common,'feature_review_complete':True,'regions':marked})
    write_json(root/'vase_unmarked.json',{**common,'regions':unmarked})
    angles = np.arange(N)*(2*np.pi/N)
    ring_truth = []
    for row in range(BELT[0],BELT[1]+1):
        envelope = surface(row/STEPS,angles,relief=False)
        for i, vertex in enumerate(outer[row]):
            ring_truth.append([vertex+1,*envelope[i].tolist(),float(angles[i])])
    write_json(root/'vase_truth.json',{'source_hash':clean.hashes['source_hash'],'relief':ring_truth,
              'slices':[[i+1 for i in row] for row in outer], 'height_m':HEIGHT,'seed':args.seed,
              'noise_sigma_m':.00029,'belt_rows':list(BELT)})
    rng = np.random.default_rng(args.seed)
    patch = {}
    for row, ring_ids in enumerate(outer):
        if row <=4 or row>=STEPS-5 or BELT[0]<=row<=BELT[1]:
            continue
        noise = .00029*np.clip(rng.normal(size=N),-2.3,2.3)
        for i, vertex in enumerate(ring_ids):
            direction = np.array([np.cos(angles[i]),np.sin(angles[i]),0])
            patch[int(clean.vid[vertex])] = clean.V[vertex] + noise[i]*direction
    io.write_positions(obj,patch,clean.hashes['source_hash'])
    save(root/'vase_noisy.blend')
    if args.preview:
        io.open_blend(root/'vase_clean.blend')
        preview(args.preview)
    print(f'P4 VASE {len(clean.V)} vertices {len(clean.F)} faces | seed {args.seed}',flush=True)


if __name__=='__main__':
    main()
