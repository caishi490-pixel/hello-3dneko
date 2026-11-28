# SPDX-License-Identifier: GPL-3.0-or-later
"""Reopen the generated fixtures and final P4 result with Blender."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gre import blender_io as io
from gre.qa import INVARIANTS
from gre.snapshot import read_json


def read(path):
    io.open_blend(path.resolve())
    obj = io.mesh_object('Vase')
    mesh = io.read_mesh(obj)
    return mesh, io.qa_snapshot(obj,mesh)


def main():
    parser = argparse.ArgumentParser()
    for name in ('data','repeated','output'):
        parser.add_argument(f'--{name}',required=True,type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    for name in ('vase_clean.blend','vase_noisy.blend'):
        mesh,snapshot = read(args.data/name)
        _,repeat = read(args.repeated/name)
        assert snapshot.hashes == repeat.hashes and snapshot.V.tobytes() == repeat.V.tobytes()
        print(f'REGENERATED {name}: all geometry hashes and coordinates agree',flush=True)
    noisy,baseline = mesh,snapshot
    final,native = read(args.output/'marked'/'vase_bilateral.blend')
    assert all(baseline.hashes[key] == native.hashes[key] for key in INVARIANTS)
    proof = read_json(args.output/'metrics.json')['cases']['marked']['methods']['bilateral']
    assert native.hashes['position_hash'] == proof['position_hash']
    plan = read_json(args.data/'vase_marked.json')
    protected = {fid for region in plan['regions'] if region['role']=='protected' for fid in region['face_ids']}
    vertices = sorted({v for fid,face in zip(noisy.fid,noisy.F) if fid in protected for v in face})
    assert noisy.V[vertices].tobytes() == final.V[vertices].tobytes()
    truth = read_json(args.data/'vase_truth.json')
    index = {int(vid):i for i,vid in enumerate(final.vid)}
    ends = [[index[vid] for vid in row] for row in (truth['slices'][0],truth['slices'][-1])]
    assert noisy.V[ends].tobytes() == final.V[ends].tobytes()
    clean,_ = read(args.data/'vase_clean.blend')
    assert np.linalg.norm(final.V-clean.V) < np.linalg.norm(noisy.V-clean.V)
    print(f'DISK P4: {len(vertices)} annotated protected vertices and both end rings exact',flush=True)


if __name__=='__main__':
    main()
