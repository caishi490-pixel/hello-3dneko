# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender-side execution of one frozen P5 transaction."""

import ctypes
import html
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import bpy
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gre import blender_io as io
from gre import p3_cli as family, p4_cli as organic
from gre.cli import select_faces
from gre.denoise import bilateral_vertex, laplacian, pack_neighbors, robust_normals, taubin
from gre.geometry import bbox_diagonal, closed_locked_rim, distance_statistics, face_geometry, partition_region
from gre.plan import DEFAULTS, PlanError, bind, read_strict
from gre.plane import DegeneratePlane, fit_locked_rim, fit_plane, project_region
from gre.qa import CandidatePatch, INVARIANTS, QAProfile, evaluate
from gre.relations import Infeasible, project_junctions, solve_planes
from gre.snapshot import commit, file_hash, load_snapshot, rollback, save_snapshot, write_json


def peak_memory():
    if os.name=='nt':
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('page_faults',wintypes.DWORD)] + [
                (name,ctypes.c_size_t) for name in ('peak','working','paged_peak','paged','nonpaged_peak',
                                                    'nonpaged','pagefile','pagefile_peak')]

        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD)
        query.restype = wintypes.BOOL
        data = Counters()
        data.cb = ctypes.sizeof(data)
        return int(data.peak) if query(handle(),ctypes.byref(data),data.cb) else None
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)


def open_mesh(path, name):
    io.open_blend(path)
    obj = io.mesh_object(name)
    mesh = io.read_mesh(obj)
    return mesh,io.qa_snapshot(obj,mesh)


def footprint(source, annotation, semantic, V, reference):
    if semantic=='organic':
        mask,h,_ = organic.selection(source,annotation,reference.hashes['source_hash'])
        declared = {fid for row in annotation['regions'] if row['role']=='editable' for fid in row['face_ids']}
        faces = np.flatnonzero(np.isin(source.fid,list(declared)))
        return SimpleNamespace(mask=mask,h=h,faces=faces,targets=np.arange(len(V)),relations=[])
    if semantic=='plane':
        faces = select_faces(annotation,'tabletop',source)
        regions,links,junctions = {'tabletop':faces},(),set()
    else:
        regions,links,junctions = family.parse_annotations(annotation,source)
        faces = np.unique(np.concatenate(list(regions.values())))
    graph,_,boundary,h = family.footprint(source,V,regions)
    core,ring,taper = partition_region(graph,boundary,annotation['transition_rings'])
    mask = np.zeros(len(V))
    mask[core],mask[ring] = 1,taper
    return SimpleNamespace(mask=mask,h=h,faces=faces,targets=np.array(sorted(graph)),core=core,ring=ring,
                           taper=taper,boundary=boundary,regions=regions,links=links,junctions=junctions,
                           relations=annotation.get('relations',[]))


def solve(V, source, area, annotation, plan):
    algorithm, params = plan['algorithm'],plan['parameters']
    if algorithm=='plane':
        triangles = np.isin(source.triangle_faces,area.faces)
        _,_,weights = face_geometry(V,source.triangles[triangles],source.triangle_faces[triangles],len(source.F))
        fit = fit_plane(V[area.targets],weights[area.targets])
        if fit[3]>params['max_planarity_ratio']:
            raise DegeneratePlane('not_planar')
        rim = closed_locked_rim(tuple(source.F[i] for i in area.faces),area.boundary)
        anchor = None if rim is None else fit_locked_rim(V[area.targets],weights[area.targets],V[rim],weights[rim],fit,area.h)
        normal,offset,*_ = anchor if anchor is not None else fit
        return project_region(V,area.core,area.ring,normal,offset,area.taper,passes=16 if anchor is not None else 8),{}
    if algorithm=='plane_family':
        opts = SimpleNamespace(ransac_distance=params['ransac_distance_m'],min_coverage=params['min_coverage'])
        found,observations = family.detect_regions(V,source,area.regions,opts,plan['random_seed'])
        if found.keys()!=area.regions.keys():
            raise DegeneratePlane('insufficient_connected_plane_support')
        active = {name:ids[found[name].inliers] for name,ids in area.regions.items()}
        graph,member,boundary,_ = family.footprint(source,V,active)
        core,ring,taper = partition_region(graph,boundary,annotation['transition_rings'])
        if not len(core):
            raise DegeneratePlane('insufficient_connected_plane_support')
        support = family.with_locked_support(observations,member,boundary,V,opts.ransac_distance)
        planes,_ = solve_planes(support,area.links)
        return project_junctions(V,np.concatenate((core,ring)),np.concatenate((np.ones(len(core)),1-(1-taper)**8)),
                                 member,planes,area.junctions)
    rows = [[] for _ in V]
    for i,j in source.edges:
        rows[i].append(j)
        rows[j].append(i)
    adj = pack_neighbors(rows)
    mask = area.mask
    budget = plan['max_displacement']['value']
    if algorithm=='laplacian':
        result = laplacian(V,adj,mask,params['lambda'],params['iterations'],budget=budget)
    elif algorithm=='taubin':
        result = taubin(V,adj,mask,params['lambda'],params['mu'],params['iterations'],budget=budget)
    else:
        result = bilateral_vertex(V,lambda X:robust_normals(X,source.triangles,adj),adj,
                                  params['sigma_s_h']*area.h,params['sigma_r_h']*area.h,params['iterations'],
                                  mask*params['relaxation'],budget=budget)
    return result.V,{'iterations':result.iterations,'stopped_by_budget':result.stopped_by_budget,
                     'last_attempt_peak_m':result.proposed_peak_m}


def preview(folder, source, candidate, patch, label):
    i = {int(vid):k for k,vid in enumerate(source.vid)}
    indices = [i[vid] for vid in patch.vertex_ids.tolist()]
    np.savez_compressed(folder/'preview.npz',vid=patch.vertex_ids,before=source.V[indices],after=patch.positions,
                        unit_scale=source.unit_scale)
    # A projected native wireframe before/after, with displacement overlaid on the candidate.
    axes = np.array([[.91,-.28],[.41,.53],[0,.80]])
    points = (np.concatenate((source.V,candidate.V)) @ axes)
    lo,hi = points.min(axis=0),points.max(axis=0)
    scale = 260/max(1e-12,float(np.max(hi-lo)))
    points = (points-(lo+hi)/2)*scale
    points[:,1] *= -1
    before,after = np.split(points,2)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 355"><title>{html.escape(label)}</title>',
            '<rect width="680" height="355" fill="#152229"/><g fill="#eef6f7" font-family="sans-serif" font-size="17">',
            '<text x="25" y="28">Input</text><text x="365" y="28">Candidate</text></g>']
    # Regular edge decimation limits the renderer size without changing the actual mesh.
    edges = source.edges[::max(1,(len(source.edges)+1599)//1600)]
    d = np.linalg.norm(candidate.V-source.V,axis=1)
    denom = max(d.max(initial=0),1e-30)
    for view,x in ((before,170),(after,510)):
        segments = ' '.join(f'M{view[a,0]+x:.2f},{view[a,1]+178:.2f}L{view[b,0]+x:.2f},{view[b,1]+178:.2f}' for a,b in edges)
        body.append(f'<path d="{segments}" fill="none" stroke="#6e99a5" stroke-width=".55"/>')
    for k in indices[::max(1,(len(indices)+1199)//1200)]:
        c = int(80+175*d[k]/denom)
        body.append(f'<circle cx="{after[k,0]+510:.2f}" cy="{after[k,1]+178:.2f}" r="1.45" fill="rgb({c},161,81)"/>')
    body.append('</svg>')
    (folder/'preview.svg').write_text(''.join(body),encoding='utf-8')


def run(req, folder, audit):
    def phase(name):
        audit['phases'].append({'name':name,'utc':datetime.now(timezone.utc).isoformat()})
        write_json(folder/f'phase-{len(audit["phases"]):02}-{name}.json',audit['phases'][-1])

    root = Path(req['workspace'])
    if req['action']=='rollback':
        phase('rollback')
        try:
            audit['rollback'] = rollback(root,req['audit_id'],req['expected_current_hash'])
        except (ValueError,FileNotFoundError) as error:
            audit.update(decision='rejected',failure_reason=str(error),failure_category='data_fidelity')
            return
        receipt = read_strict(root/'audit'/f"{req['audit_id']}.json")
        frozen,_ = load_snapshot(root,receipt['asset_id'],receipt['source_file_sha256'])
        restored = root/'working'/receipt['working_file']
        _,mesh = open_mesh(restored,frozen.hashes['object'])
        if mesh.hashes!=frozen.hashes or mesh.V.tobytes()!=frozen.V.tobytes():
            raise RuntimeError('Reopened rollback did not match the frozen mesh')
        audit.update(asset_id=receipt['asset_id'],input_hash=receipt['source_file_sha256'],rollback_verified=True)
        audit['decision'] = 'accepted'
        return
    paths = {key:Path(req[key]) for key in ('source','annotations','reference','truth') if req.get(key)}
    if req.get('plan'):
        paths['plan'] = Path(req['plan'])
    fingerprints = {key:file_hash(path) for key,path in paths.items()}
    evidence = {key+'_sha256':digest for key,digest in fingerprints.items() if key in ('annotations','reference','truth')}
    supplied = read_strict(paths['plan']) if 'plan' in paths else None
    if supplied is not None:
        from gre.plan import validate
        supplied = validate(supplied)
    semantic = supplied['semantic_class'] if supplied else req['semantic']
    annotation = read_strict(paths['annotations'])
    name = annotation.get('object_name','Table')
    phase('analyze')
    reference,ref_snapshot = open_mesh(paths['reference'],name)
    source,baseline = open_mesh(paths['source'],name)
    if (annotation['source_hash']!=reference.hashes['source_hash']
            or any(baseline.hashes[key]!=ref_snapshot.hashes[key] for key in INVARIANTS)
            or source.unit_scale!=reference.unit_scale):
        raise PlanError('Clean reference, authorization and source topology do not match')
    V,clean = source.V*source.unit_scale,reference.V*source.unit_scale
    area = footprint(source,annotation,semantic,V,reference)
    vids,fids = set(source.vid.tolist()),set(source.fid.tolist())
    allowed = set(source.vid[area.mask>0].tolist())
    faces = set(source.fid[area.faces].tolist())
    if not allowed:
        raise PlanError('No editable vertices in the authorized footprint')
    cap = annotation.get('max_displacement_m')
    if cap is None:
        cap = min(area.h,.011*bbox_diagonal(V))
    audit.update(asset_id=annotation['asset_id'],input_hash=fingerprints['source'],annotations_hash=fingerprints['annotations'],
                 evidence=evidence,h_m=area.h,max_authorized_displacement_m=cap)
    audit['target_rms_before'] = distance_statistics((V-clean)[area.targets])['rms']
    save_snapshot(root,annotation['asset_id'],baseline,paths['source'],fingerprints['source'])
    phase('plan')
    algorithm = supplied['algorithm'] if supplied else req['algorithm']
    plan = supplied or dict(schema_version=1,asset_id=annotation['asset_id'],source_hash=fingerprints['source'],
                            region_face_ids=sorted(faces),editable_vertex_ids=sorted(allowed),protected_vertex_ids=sorted(vids-allowed),
                            coordinate_space='object_local',unit_scale=source.unit_scale,semantic_class=semantic,
                            evidence_refs=evidence,constraints=area.relations if algorithm=='plane_family' else [],
                            algorithm=algorithm,parameters=DEFAULTS[algorithm],max_displacement={'value':cap,'unit':'m'},
                            qa_profile={},random_seed=req.get('seed',42),approval_state='draft' if req['action']=='inspect' else 'approved')
    write_json(folder/'plan.json',plan)
    audit.update(algorithm=algorithm,params=plan['parameters'],seed=plan['random_seed'])
    if req['action']=='inspect':
        audit.update(decision='no_action',failure_reason=None)
        return
    plan = bind(plan,source_hash=fingerprints['source'],asset_id=annotation['asset_id'],unit_scale=source.unit_scale,
                evidence=evidence,source_vid=vids,source_fid=fids,allowed=allowed,faces=faces,relations=area.relations)
    budget = plan['max_displacement']['value']
    if budget>cap:
        raise PlanError('Plan exceeds the authorized displacement budget')
    phase('solve')
    points = V
    if algorithm!='none' and not (algorithm in ('plane','plane_family') and np.array_equal(V[area.targets],clean[area.targets])):
        try:
            points,audit['solver'] = solve(V,source,area,annotation,plan)
        except (Infeasible,DegeneratePlane) as error:
            category = 'segmentation' if str(error)=='insufficient_connected_plane_support' else (
                'estimation' if isinstance(error,DegeneratePlane) else 'solver')
            audit.update(decision='rejected',failure_reason=str(error),failure_category=category)
            return
    # Build the exact float32 proposal without filtering or silently clipping any attempted member.
    if points is V:
        stored = source.V
    else:
        intended = np.any(points!=V,axis=1)
        stored = source.V.copy()
        stored[intended] = (points[intended]/source.unit_scale).astype(np.float32)
    modified = np.any(stored.view(np.uint64)!=source.V.view(np.uint64),axis=1)
    phase('validate')
    proposal = CandidatePatch(source.vid[modified],stored[modified],{key:value for key,value in baseline.hashes.items() if key!='position_hash'})
    profile = QAProfile(frozenset(allowed),budget,target_vids=tuple(source.vid[area.targets].tolist()),clean_positions=clean,
                        **plan['qa_profile'])
    qa = evaluate(baseline,proposal,profile)
    audit['qa'] = qa
    decision = qa['status'] if modified.any() else 'no_action'
    reason = None
    if qa['hard_failures']:
        decision,reason = 'rejected',','.join(qa['hard_failures'])
        audit['failure_category'] = 'design_intent_conflict' if req.get('design_conflict') else 'data_fidelity'
    elif decision=='needs_review':
        reason = ','.join(qa['review_reasons'])
        audit['failure_category'] = 'data_fidelity'
    if semantic=='organic':
        if 'truth' not in paths:
            if modified.any() and decision=='accepted':
                decision,reason = 'needs_review','independent_feature_reference_missing'
                audit['failure_category'] = 'semantic'
        else:
            truth = read_strict(paths['truth'])
            if truth['source_hash']!=reference.hashes['source_hash']:
                raise PlanError('Truth does not match the clean mesh')
            measured = organic.independent_metrics(V,stored*source.unit_scale,clean,truth,source.vid)
            audit['features'] = measured
            audit['feature_preservation'] = measured['relief_retention']
            if modified.any() and decision!='rejected':
                if annotation.get('feature_review_complete') is not True:
                    decision,reason = 'needs_review','feature_annotation_review_missing'
                    audit['failure_category'] = 'semantic'
                elif measured['centerline_max_change_from_clean_m']>.00035 or measured['silhouette_support_p95_m']>.00055:
                    decision,reason = 'needs_review','intentional_contour_deviation'
                    audit['failure_category'] = 'design_intent_conflict'
    if algorithm=='plane_family' and modified.any():
        target_vertices = {region:np.array(sorted({v for i in ids for v in source.F[i]})) for region,ids in area.regions.items()}
        audit['measured_relations'] = family.measure(stored*source.unit_scale,target_vertices,area.links,{})['relations']
        if decision=='accepted' and any(row['deviation_deg']>=.1 or row.get('offset_error_m',0)>=.001 for row in audit['measured_relations']):
            decision,reason = 'needs_review','measured_relation_out_of_tolerance'
            audit['failure_category'] = 'solver'
    audit.update(decision=decision,failure_reason=reason)
    if decision=='rejected':
        return
    phase('preview')
    copy = SimpleNamespace(V=stored)
    preview(folder,source,copy,proposal,req['case_id'])
    if decision!='accepted':
        return
    if any(file_hash(path)!=fingerprints[key] for key,path in paths.items()):
        raise PlanError('Input changed before candidate publication')
    with TemporaryDirectory(prefix='.gre-p5-',dir=folder) as scratch:
        stage = Path(scratch)/'candidate.blend'
        io.open_blend(paths['source'])
        io.write_positions(io.mesh_object(name),dict(zip(proposal.vertex_ids.tolist(),proposal.positions)),source.hashes['source_hash'])
        io.save_copy(stage,paths['source'],folder/'candidate.blend')
        _,observed = open_mesh(stage,name)
        if observed.V.tobytes()!=stored.tobytes() or any(observed.hashes[key]!=baseline.hashes[key] for key in INVARIANTS):
            raise RuntimeError('Native candidate round trip failed integrity')
        os.link(stage,folder/'candidate.blend')
        audit['candidate_sha256'] = file_hash(stage)
        audit['candidate_position_hash'] = observed.hashes['position_hash']
        if req.get('commit') or req.get('transaction_test'):
            phase('commit')
            if any(file_hash(path)!=fingerprints[key] for key,path in paths.items()):
                raise PlanError('Input changed before commit')
            receipt = commit(root,annotation['asset_id'],fingerprints['source'],stage,req['case_id']+'.blend','accepted')
            audit['commit'] = receipt
            working = root/'working'/receipt['working_file']
            _,checked = open_mesh(working,name)
            if checked.hashes!=observed.hashes or checked.V.tobytes()!=observed.V.tobytes():
                raise RuntimeError('Committed file differs from checked candidate')
            audit['commit_verified'] = True
            if req.get('transaction_test'):
                phase('rollback')
                audit['rollback'] = rollback(root,receipt['audit_id'],receipt['committed_file_sha256'])
                _,restored = open_mesh(working,name)
                if file_hash(working)!=fingerprints['source'] or restored.hashes!=baseline.hashes or restored.V.tobytes()!=baseline.V.tobytes():
                    raise RuntimeError('Rollback did not restore the original file and mesh')
                audit['rollback_verified'] = True


def main():
    request_path = Path(sys.argv[sys.argv.index('--')+1])
    req = read_strict(request_path)
    folder = request_path.parent
    audit = {'case_id':req['case_id'],'decision':'solver_failed','phases':[],
             'versions':{'blender':bpy.app.version_string,'python':sys.version.split()[0],'numpy':np.__version__}}
    write_json(folder/'environment.json',audit['versions'])
    try:
        run(req,folder,audit)
    except (PlanError,io.Unsupported) as error:
        audit.update(decision='unsupported' if isinstance(error,io.Unsupported) else 'rejected',failure_reason=str(error),
                     failure_category='semantic')
    audit['peak_memory'] = peak_memory()
    write_json(folder/'worker_result.json',audit)


if __name__=='__main__':
    main()
