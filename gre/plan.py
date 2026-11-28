# SPDX-License-Identifier: Apache-2.0
"""Strict, host-independent schema for the E3 manual plan."""

import json
import math
import re
from pathlib import Path


FIELDS = frozenset('schema_version asset_id source_hash region_face_ids editable_vertex_ids protected_vertex_ids '
                   'coordinate_space unit_scale semantic_class evidence_refs constraints algorithm parameters '
                   'max_displacement qa_profile random_seed approval_state'.split())
DEFAULTS = {
    'none': {}, 'plane': {'max_planarity_ratio': .005},
    'plane_family': {'ransac_distance_m': .004, 'min_coverage': .60},
    'laplacian': {'iterations': 1, 'lambda': .45},
    'taubin': {'iterations': 1, 'lambda': .45, 'mu': -.47},
    'bilateral': {'iterations': 1, 'sigma_s_h': .90, 'sigma_r_h': .12, 'relaxation': .28},
}
SEMANTICS = {'plane': {'none','plane'}, 'planar_assembly': {'none','plane_family'},
             'organic': {'none','laplacian','taubin','bilateral'}}
QA_LIMITS = {'min_angle_deg': (.1,30), 'uv_p95': (0,.02), 'uv_max': (0,.05),
             'area_limit': (0,.01), 'volume_limit': (0,.005), 'pair_limit': (1,500_000)}


class PlanError(ValueError):
    """A hand-authored plan failed schema or authorization checks."""


def read_strict(path: Path):
    def pairs(items):
        row = {}
        for key,value in items:
            if key in row:
                raise PlanError(f'Duplicate field: {key}')
            row[key] = value
        return row

    def constant(value):
        raise PlanError(f'Forbidden non-finite number: {value}')

    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):
            raise PlanError('Forbidden non-finite number')
        if isinstance(value,(dict,list)):
            for item in value.values() if isinstance(value,dict) else value:
                finite(item)

    try:
        result = json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=constant)
    except json.JSONDecodeError as error:
        raise PlanError(f'Invalid JSON: {error}') from error
    finite(result)
    return result


def identifier(value):
    return isinstance(value,str) and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',value) is not None


def _hash(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def _number(value, low, high, *, integer=False):
    return type(value) is int and low<=value<=high if integer else (
        type(value) in (int,float) and low<=value<=high and math.isfinite(value))


def validate(plan):
    if not isinstance(plan,dict) or plan.keys()!=FIELDS:
        raise PlanError('E3 plan has missing or unknown fields')
    if type(plan['schema_version']) is not int or plan['schema_version']!=1 or not identifier(plan['asset_id']) or not _hash(plan['source_hash']):
        raise PlanError('Invalid schema version, asset ID or source SHA-256')
    if plan['coordinate_space']!='object_local' or not _number(plan['unit_scale'],1e-12,1e12):
        raise PlanError('Plan requires object-local coordinates and explicit positive unit scale')
    for name in ('region_face_ids','editable_vertex_ids','protected_vertex_ids'):
        values = plan[name]
        if (not isinstance(values,list) or (name!='protected_vertex_ids' and not values)
                or any(not _number(v,1,2**31-1,integer=True) for v in values) or len(set(values))!=len(values)):
            raise PlanError(f'Empty or invalid stable IDs: {name}')
    if set(plan['editable_vertex_ids']).intersection(plan['protected_vertex_ids']):
        raise PlanError('Editable and protected vertices overlap')
    if (not _number(plan['random_seed'],0,2**64-1,integer=True)
            or plan['approval_state'] not in ('draft','approved')):
        raise PlanError('Invalid seed or approval state')
    evidence = plan['evidence_refs']
    if (not isinstance(evidence,dict) or not {'annotations_sha256','reference_sha256'}<=evidence.keys()
            or evidence.keys()-{'annotations_sha256','reference_sha256','truth_sha256'} or not all(map(_hash,evidence.values()))):
        raise PlanError('Expected pinned annotation and reference evidence hashes')
    budget = plan['max_displacement']
    if (not isinstance(budget,dict) or set(budget)!={'value','unit'} or budget['unit']!='m'
            or not _number(budget['value'],1e-12,1e6)):
        raise PlanError('Displacement budget requires explicit metres and a positive finite value')
    algo,semantic = plan['algorithm'],plan['semantic_class']
    if not isinstance(algo,str) or not isinstance(semantic,str) or algo not in SEMANTICS.get(semantic,set()):
        raise PlanError('Unknown algorithm or incompatible semantic class')
    params = plan['parameters']
    if not isinstance(params,dict) or params.keys()-DEFAULTS[algo].keys():
        raise PlanError('Unknown algorithm parameter')
    ranges = {'max_planarity_ratio':(1e-12,1),'ransac_distance_m':(1e-12,1), 'min_coverage':(.01,1),
              'iterations':(1,100),'lambda':(1e-9,.999),'mu':(-.999,-1e-9),'sigma_s_h':(1e-9,20),
              'sigma_r_h':(1e-9,20),'relaxation':(1e-9,1)}
    if any(not _number(value,*ranges[key],integer=key=='iterations') for key,value in params.items()):
        raise PlanError('Invalid algorithm parameter value')
    qa = plan['qa_profile']
    if (not isinstance(qa,dict) or qa.keys()-QA_LIMITS.keys()
            or any(not _number(value,*QA_LIMITS[key],integer=key=='pair_limit') for key,value in qa.items())):
        raise PlanError('Invalid, unknown or relaxed QA threshold')
    constraints = plan['constraints']
    if not isinstance(constraints,list) or (constraints and algo!='plane_family'):
        raise PlanError('Constraints must be an array for the plane-family solver')
    for row in constraints:
        if not isinstance(row,dict) or row.get('type') not in ('parallel','perpendicular','offset'):
            raise PlanError('Unknown constraint type')
        keys = {'type','a','b'} | ({'distance_m'} if row['type']=='offset' else set())
        if (row.keys()!=keys or not identifier(row['a']) or not identifier(row['b']) or row['a']==row['b']
                or not _number(row.get('distance_m',0),-1e6,1e6)):
            raise PlanError('Invalid constraint fields, names or distance in metres')
    return {**plan,'parameters':{**DEFAULTS[algo],**params}}


def bind(plan, *, source_hash, asset_id, unit_scale, evidence, source_vid, source_fid, allowed, faces, relations):
    plan = validate(plan)
    if plan['approval_state']!='approved':
        raise PlanError('Plan is not manually approved')
    if (plan['source_hash']!=source_hash or plan['asset_id']!=asset_id or plan['unit_scale']!=unit_scale
            or plan['evidence_refs']!=evidence):
        raise PlanError('Stale source, mismatched asset/units or changed evidence')
    vertices = set(plan['editable_vertex_ids'])
    protected = set(plan['protected_vertex_ids'])
    region = set(plan['region_face_ids'])
    if not region<=source_fid or not (vertices|protected)<=source_vid:
        raise PlanError('Stable vertex or face ID is out of range')
    if region!=faces or vertices!=allowed or vertices|protected!=source_vid:
        raise PlanError('Plan must match the full authorized footprint and lock every other vertex')
    if plan['algorithm']=='plane_family' and plan['constraints']!=relations:
        raise PlanError('Constraints lack matching annotation authorization')
    return plan
