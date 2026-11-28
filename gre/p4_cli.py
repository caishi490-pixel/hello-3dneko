# SPDX-License-Identifier: Apache-2.0
"""Run the P4 vase benchmark and publish only an approved bilateral candidate."""

import argparse
import os
import sys
from pathlib import Path
from tempfile import mkstemp
from time import perf_counter

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gre import blender_io as io
from gre.denoise import FilterResult, bilateral_vertex, laplacian, pack_neighbors, robust_normals, sharp_vertices, taubin
from gre.geometry import adjacency, distance_statistics, face_edges, face_geometry, partition_region
from gre.qa import CandidatePatch, INVARIANTS, QAProfile, evaluate
from gre.report import write_report
from gre.snapshot import file_hash, read_json, write_json


def selection(source, plan, clean_hash):
    if (plan.get('schema_version') != 1 or plan.get('object_name') != 'Vase' or plan.get('source_hash') != clean_hash
            or plan.get('unit_scale') != source.unit_scale or plan.get('matrix_world') != source.matrix_world.tolist()):
        raise ValueError('Plan does not match the approved reference, units or transform')
    budget, rings, angle = (plan[key] for key in ('max_displacement_m','transition_rings','feature_angle_deg'))
    if (type(rings) is not int or rings < 0 or type(budget) not in (float,int) or not np.isfinite(budget) or budget <= 0
            or type(angle) not in (float,int) or not np.isfinite(angle) or not 0 < angle < 180):
        raise ValueError('Invalid displacement budget, transition ring count or feature angle')
    seen, selected, names = set(), set(), set()
    source_fids = set(source.fid.tolist())
    for row in plan['regions']:
        ids = row['face_ids']
        name, role = row['name'], row['role']
        if (not isinstance(name,str) or not name or name in names or role not in {'editable','protected','unknown'}
                or not isinstance(ids,list) or not ids or any(type(fid) is not int or fid not in source_fids for fid in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError('Invalid or overlapping face annotations')
        seen.update(ids)
        names.add(name)
        if role == 'editable':
            selected.update(ids)
    if not selected:
        raise ValueError('No approved editable polygons')
    fid = source.fid.tolist()
    faces = tuple(face for i,face in zip(fid,source.F) if i in selected)
    edges = face_edges(faces)
    V = source.V * source.unit_scale
    graph,h = adjacency(V,edges)
    outside = {v for i,face in zip(fid,source.F) if i not in selected for v in face}
    outside.update(v for edge in source.edges if tuple(sorted(edge)) not in edges for v in edge)
    normals,_,_ = face_geometry(V,source.triangles,source.triangle_faces,len(source.F))
    features = sharp_vertices(source.F,normals,angle)
    index = {int(vid):i for i,vid in enumerate(source.vid)}
    declared = plan.get('feature_vertex_ids',[])
    if any(type(vid) is not int or vid not in index for vid in declared):
        raise ValueError('Unknown declared feature vertex ID')
    features.update(index[vid] for vid in declared)
    boundary = graph.keys() & (outside | features)
    core,blend,taper = partition_region(graph,boundary,rings)
    mask = np.zeros(len(V))
    mask[core], mask[blend] = 1, taper
    return mask,h,features


def independent_metrics(before, after, clean, truth, vids):
    index = {int(vid):i for i,vid in enumerate(vids)}
    rings = np.array([[index[vid] for vid in row] for row in truth['slices']],dtype=np.intp)
    band = np.array([index[item[0]] for item in truth['relief']],dtype=np.intp)
    baseline = np.array([item[1:] for item in truth['relief']])
    angles = baseline[:,2]
    normal = np.column_stack((np.cos(angles),np.sin(angles)))
    relief = np.sum((clean[band,:2]-baseline[:,:2])*normal,axis=1)
    error = np.sum((after[band,:2]-clean[band,:2])*normal,axis=1)
    feature_loss = float(np.linalg.norm(error)/np.linalg.norm(relief))
    sample = np.arange(16)*2*np.pi/16
    direction = np.column_stack((np.cos(sample),np.sin(sample)))

    def contour(X):
        return (X[rings,:2]@direction.T).max(axis=1)

    line = after[rings,:2].mean(axis=1)
    clean_line = clean[rings,:2].mean(axis=1)
    return {'clean_error_m':distance_statistics(after-clean), 'input_error_m':distance_statistics(before-clean),
            'relief_retention':max(0.,1-feature_loss), 'relief_relative_error':feature_loss,
            'relief_coordinates_exact':before[band].tobytes()==after[band].tobytes(),
            'centerline_max_change_from_clean_m':float(np.linalg.norm(line-clean_line,axis=1).max()),
            'clean_top_base_eccentricity_m':float(np.linalg.norm(clean_line[-1]-clean_line[0])),
            'top_base_eccentricity_m':float(np.linalg.norm(line[-1]-line[0])),
            'silhouette_support_p95_m':float(np.percentile(np.abs(contour(after)-contour(clean)),95)),
            'silhouette_support_max_m':float(np.max(np.abs(contour(after)-contour(clean))))}


def publish(source, dest, folder, object_name, before, reference, native, fingerprints, no_images):
    if os.path.lexists(dest):
        raise FileExistsError(dest)
    fd, temp = mkstemp(prefix='.gre-p4-',suffix='.blend',dir=dest.parent)
    os.close(fd)
    staged = Path(temp)
    try:
        io.save_copy(staged,source,dest)
        io.open_blend(staged)
        obj = io.mesh_object(object_name)
        data = io.read_mesh(obj)
        checked = io.qa_snapshot(obj,data)
        if checked.hashes != native.hashes or checked.V.tobytes() != native.V.tobytes():
            raise RuntimeError('Native disk round trip changed the candidate')
        if not no_images:
            io.render_previews(obj,before,data,reference,folder,qa=True,fit_frame=True)
        if any(file_hash(path) != digest for path,digest in fingerprints.items()):
            raise RuntimeError('Input changed before publishing the P4 result')
        checksum = file_hash(staged)
        os.link(staged,dest)
        return checksum
    finally:
        staged.unlink(missing_ok=True)


def run(args):
    files = [path.resolve(strict=True) for path in (args.source,args.reference,args.marked,args.unmarked,args.truth)]
    source_file,ref_file,*_ = files
    fingerprints = {path:file_hash(path) for path in files}
    io.open_blend(ref_file)
    ref_obj = io.mesh_object('Vase')
    reference = io.read_mesh(ref_obj)
    clean = io.qa_snapshot(ref_obj,reference)
    truth = read_json(files[4])
    if truth['source_hash'] != reference.hashes['source_hash']:
        raise ValueError('Evaluation truth does not match the clean mesh')
    io.open_blend(source_file)
    obj = io.mesh_object('Vase')
    source = io.read_mesh(obj)
    baseline = io.qa_snapshot(obj,source)
    if any(baseline.hashes[key]!=clean.hashes[key] for key in INVARIANTS):
        raise ValueError('Reference and source have different native geometry structure')
    V = source.V * source.unit_scale
    ref = reference.V * reference.unit_scale
    rows = [[] for _ in V]
    for left,right in source.edges:
        rows[left].append(right)
        rows[right].append(left)
    adj = pack_neighbors(rows)
    result = {'stage':'P4','input_sha256':{path.name:digest for path,digest in fingerprints.items()}, 'cases':{}}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for label,file in zip(('marked','unmarked'),files[2:4]):
        plan = read_json(file)
        mask,h,edges = selection(source,plan,reference.hashes['source_hash'])
        cap = plan['max_displacement_m']
        editable = mask > 0
        profile = QAProfile(frozenset(source.vid[editable].tolist()),cap,target_vids= tuple(source.vid.tolist()),clean_positions=ref)
        case = {'h_m':h,'budget_m':cap,'locked_vertices':int((~editable).sum()),'feature_vertices':len(edges),
                'feature_review_complete':plan.get('feature_review_complete') is True,'methods':{}}
        algorithms = {
            'none': lambda:FilterResult(V,0,False,0.),
            'laplacian':lambda:laplacian(V,adj,mask,.45,args.iters,budget=cap),
            'taubin':lambda:taubin(V,adj,mask,.45,-.47,args.iters,budget=cap),
            'bilateral':lambda:bilateral_vertex(V,lambda X:robust_normals(X,source.triangles,adj),adj,
                                                args.sigma_s_h*h,args.sigma_r_h*h,args.iters,
                                                args.bilateral_relaxation*mask,budget=cap),
        }
        component = args.output_dir/label
        component.mkdir()
        for method,filter_mesh in algorithms.items():
            start = perf_counter()
            filtered = filter_mesh()
            solver_time = perf_counter()-start
            stored = source.V.copy()
            stored[editable] = (filtered.V[editable]/source.unit_scale).astype(np.float32)
            changed = np.any(stored.view(np.uint64) != source.V.view(np.uint64),axis=1)
            io.open_blend(source_file)
            obj = io.mesh_object('Vase')
            if changed.any():
                io.write_positions(obj,dict(zip(source.vid[changed].tolist(),stored[changed])),source.hashes['source_hash'])
            candidate = io.read_mesh(obj)
            native = io.qa_snapshot(obj,candidate)
            start = perf_counter()
            qa = evaluate(baseline,CandidatePatch.between(baseline,native),profile)
            qa_time = perf_counter()-start
            measured = independent_metrics(V,native.V*source.unit_scale,ref,truth,source.vid)
            disposition = qa['status'] if changed.any() else 'no_action'
            reasons = []
            if changed.any() and not case['feature_review_complete']:
                reasons.append('feature_annotation_review_missing')
            if measured['centerline_max_change_from_clean_m'] > .00035 or measured['silhouette_support_p95_m'] > .00055:
                reasons.append('intentional_contour_deviation')
            if reasons and disposition == 'accepted':
                disposition = 'needs_review'
            row = {'status':disposition,'p4_review_reasons':reasons,'p2':qa,'metrics':measured,
                   'iterations':filtered.iterations,'stopped_by_budget':filtered.stopped_by_budget,
                   'last_attempt_peak_m':filtered.proposed_peak_m,'solver_s':solver_time,'qa_s':qa_time,
                   'position_hash':native.hashes['position_hash']}
            case['methods'][method] = row
            output = component/method
            output.mkdir()
            if method == 'bilateral' and disposition == 'accepted':
                result_file = component/'vase_bilateral.blend'
                row['output_sha256'] = publish(source_file,result_file,output,obj.name,source,reference,native,
                                               fingerprints,args.no_images)
            write_report(output,{**qa,'status':disposition,'p2_status':qa['status'],'p4_review_reasons':reasons})
        result['cases'][label] = case
    if any(file_hash(path)!=digest for path,digest in fingerprints.items()):
        raise RuntimeError('Inputs changed during P4')
    result['parameters'] = {'iterations':args.iters, 'laplacian_lambda':.45, 'taubin_lambda':.45, 'taubin_mu':-.47,
                            'sigma_s_h':args.sigma_s_h,'sigma_r_h':args.sigma_r_h,'bilateral_relaxation':args.bilateral_relaxation}
    write_json(args.output_dir/'metrics.json',result)
    for label,case in result['cases'].items():
        for name, row in case['methods'].items():
            m = row['metrics']
            print(f"P4 {label:8s} {name:10s} {row['status']:12s} RMS {1000*m['clean_error_m']['rms']:.6f} mm | relief {m['relief_retention']:.6%}",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','reference','marked','unmarked','truth','output-dir'):
        p.add_argument(f'--{key}',required=True,type=Path)
    p.add_argument('--sigma-s-h',type=float,default=.90)
    p.add_argument('--sigma-r-h',type=float,default=.12)
    p.add_argument('--iters',type=int,default=1)
    p.add_argument('--bilateral-relaxation',type=float,default=.28)
    p.add_argument('--no-images',action='store_true')
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if (not np.isfinite([args.sigma_s_h,args.sigma_r_h]).all() or min(args.sigma_s_h,args.sigma_r_h)<=0
            or args.iters<1 or not 0 < args.bilateral_relaxation <= 1):
        p.error('Invalid bilateral parameters')
    args.output_dir = args.output_dir.resolve()
    run(args)


if __name__=='__main__':
    main()
