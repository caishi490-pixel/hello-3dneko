# SPDX-License-Identifier: Apache-2.0
"""P5 subprocess coordinator: inspect, run a manual plan, batch and rollback."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from gre.plan import SEMANTICS, PlanError, identifier, read_strict


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('asset_id input_hash annotations_hash algorithm params seed time peak_memory target_rms_before target_rms_after '
          'max_displacement topology_equal uv_hash_equal uv_stretch_p95 uv_stretch_max intersections_new feature_preservation '
          'decision failure_reason').split()
CATEGORIES = {'segmentation':'分割','semantic':'语义','estimation':'估计','solver':'求解','data_fidelity':'数据保真',
              'missed_detection':'检测漏报','design_intent_conflict':'设计意图冲突'}


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def write(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as file:
        file.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def enumerate_cases(data, catalog):
    index = read_strict(catalog)
    if (not isinstance(index,dict) or index.keys()!={'schema_version','families'}
            or type(index['schema_version']) is not int or index['schema_version']!=1 or not isinstance(index['families'],list)):
        raise PlanError('Invalid P5 catalog schema')
    required = {'semantic','reference','sources','annotations','algorithms','seed'}
    coverage, names, result = set(),set(),[]

    def resolve(reference):
        if not isinstance(reference,str):
            raise PlanError('Expected data-relative paths in the catalog')
        path = (data/reference).resolve()
        if not path.is_relative_to(data) or Path(reference).is_absolute():
            raise PlanError('Catalog paths must remain inside gre_testdata')
        return str(path)

    for row in index['families']:
        if not isinstance(row,dict) or not required<=row.keys() or row.keys()-required-{'transaction_source','truth','design_conflict'}:
            raise PlanError('Missing or unknown P5 catalog fields')
        semantic = row['semantic']
        for key in ('sources','annotations','algorithms'):
            if not isinstance(row[key],list) or not row[key] or any(not isinstance(value,str) for value in row[key]):
                raise PlanError(f'Catalog {key} requires a nonempty string array')
        if (not isinstance(semantic,str) or semantic not in SEMANTICS or not set(row['algorithms'])<=SEMANTICS[semantic]
                or type(row['seed']) is not int or not 0<=row['seed']<2**64
                or type(row.get('design_conflict',False)) is not bool
                or ('transaction_source' in row and (semantic!='planar_assembly' or row['transaction_source'] not in row['sources']))):
            raise PlanError('Invalid catalog algorithm, seed or transaction case')
        for source,annotation,algorithm in itertools.product(row['sources'],row['annotations'],row['algorithms']):
            references = {'source':source,'annotations':annotation,'reference':row['reference']}
            if 'truth' in row:
                references['truth'] = row['truth']
            case = re.sub(r'[^a-zA-Z0-9_-]','_',f'{Path(source).stem}--{Path(annotation).stem}--{algorithm}')
            if not identifier(case) or case.lower() in names:
                raise PlanError(f'Invalid or duplicate catalog case: {case}')
            names.add(case.lower())
            paths = {name:resolve(value) for name,value in references.items()}
            coverage.add(Path(paths['source']))
            result.append(dict(action='run',case_id=case,semantic=semantic,algorithm=algorithm,seed=row['seed'],
                               transaction_test=source==row.get('transaction_source') and algorithm=='plane_family',
                               design_conflict=row.get('design_conflict',False),fixture_paths=references,**paths))
    on_disk = {path.resolve() for path in data.rglob('*.blend')}
    if not on_disk<=coverage:
        raise PlanError('Unregistered .blend fixtures: '+', '.join(sorted(str(x.relative_to(data)) for x in on_disk-coverage)))
    return result


def run_child(command, log, timeout):
    start = perf_counter()
    try:
        with log.open('xb') as stream:
            result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,check=False)
        failure = None if result.returncode==0 else f'worker_exit_{result.returncode}'
    except subprocess.TimeoutExpired:
        failure = 'worker_timeout'
    except OSError as error:
        failure = f'worker_start_error: {error}'
    return perf_counter()-start,failure


def row(audit):
    checks = audit.get('qa',{}).get('checks',{})
    equal = checks.get('integrity',{}).get('invariant_equal',{})
    errors = checks.get('target_error',{}).get('clean_reference_m',{})
    layers = checks.get('uv_stretch',{}).get('layers',{}).values()

    def uv(key):
        return max((item[key] for item in layers if item.get(key) is not None),default=None)

    return {**{key:audit.get(key) for key in FIELDS},
            'target_rms_before':errors.get('before',{}).get('rms',audit.get('target_rms_before')),
            'target_rms_after':errors.get('after',{}).get('rms'),
            'max_displacement':checks.get('displacement',{}).get('maximum_m'),
            'topology_equal':equal.get('topology_hash'),'uv_hash_equal':equal.get('uv_hash'),
            'uv_stretch_p95':uv('p95_log'),'uv_stretch_max':uv('max_log'),
            'intersections_new':checks.get('self_intersection',{}).get('new_pairs')}


def attempt(blender, root, req, timeout):
    case = req['case_id']
    if not identifier(case):
        raise PlanError('Invalid case ID')
    folder = root/'runs'/case
    folder.mkdir(parents=True,exist_ok=False)
    audit_dir = root/'audit'
    audit_dir.mkdir(exist_ok=True)
    job = {**req,'workspace':str(root)}
    write(folder/'request.json',job)
    command = [str(blender),'-b','--factory-startup','-t','4','--python-exit-code','1','--python',
               str(ROOT/'gre/pipeline_worker.py'),'--',str(folder/'request.json')]
    elapsed,failure = run_child(command,folder/'worker.log',timeout)
    worker = folder/'worker_result.json'
    audit = {'case_id':case,'algorithm':req.get('algorithm'),'seed':req.get('seed'),
             'peak_memory':None,'units':{'length':'m','time':'s','peak_memory':'bytes'},'command':command,
             'source':req.get('source'),'annotations':req.get('annotations'),'reference':req.get('reference'),
             'fixture_paths':req.get('fixture_paths'), 'host':{'python':platform.python_version(),'os':platform.system()}}
    for key,path in (('input_hash',req.get('source')),('annotations_hash',req.get('annotations'))):
        try:
            audit[key] = sha(Path(path)) if path else None
        except OSError:
            audit[key] = None
    if failure is None:
        try:
            result = read_strict(worker)
            if not isinstance(result,dict) or result.get('decision') not in {'accepted','rejected','needs_review','no_action','solver_failed','unsupported'}:
                raise PlanError('Invalid worker decision')
            audit.update(result)
        except (ValueError,OSError) as error:
            failure = f'invalid_worker_result: {error}'
    if failure is not None:
        # A killed worker may leave a partially written diagnostic. Keep the case and continue.
        unreadable = []

        def diagnostic(path):
            try:
                return read_strict(path)
            except (ValueError,OSError):
                unreadable.append(path.name)
                return None

        audit.update(decision='solver_failed',failure_category='solver',failure_reason=failure,
                     phases=[value for path in sorted(folder.glob('phase-*.json')) if (value:=diagnostic(path)) is not None],
                     versions=diagnostic(folder/'environment.json'))
        audit['unreadable_diagnostics'] = unreadable
    audit.update(run_id=uuid4().hex,time=elapsed,paths={'run':f'runs/{case}','log':f'runs/{case}/worker.log'})
    if audit['peak_memory'] is None:
        audit['peak_memory_unavailable_reason'] = 'Worker resource counter was unavailable or did not complete'
    record = row(audit)
    audit['summary_row'] = record
    write(audit_dir/f"{audit['run_id']}.json",audit)
    category = audit.get('failure_category')
    if category:
        archive = root/'failures'/category
        archive.mkdir(parents=True,exist_ok=True)
        write(archive/f"{audit['run_id']}.json",{'category':category,'category_zh':CATEGORIES[category],
              'audit':f"audit/{audit['run_id']}.json",'fixture_paths':req.get('fixture_paths'),
              'plan':f'runs/{case}/plan.json' if (folder/'plan.json').exists() else None,
              'preview':f'runs/{case}/preview.svg' if (folder/'preview.svg').exists() else None,
              'log':f'runs/{case}/worker.log','decision':audit['decision'],'reason':audit['failure_reason']})
    print(f"{case}: {audit['decision']} ({elapsed:.2f}s) {audit.get('failure_reason') or ''}",flush=True)
    return audit


def summarize(root, audits):
    rows = [{'case_id':a['case_id'],'run_id':a['run_id'],**a['summary_row']} for a in audits]
    counts = Counter(a['decision'] for a in audits)
    category_counts = Counter(a.get('failure_category') for a in audits)
    stats = {'cases':len(audits),'inputs':len({a['input_hash'] for a in audits if a.get('input_hash')}),
             'decisions':dict(counts),'failures_by_category':{kind:category_counts[kind] for kind in CATEGORIES},
             'verified_commits':sum(a.get('commit_verified',False) is True for a in audits),
             'verified_rollbacks':sum(a.get('rollback_verified',False) is True for a in audits),
             'peak_memory_bytes':max((a['peak_memory'] for a in audits if a['peak_memory'] is not None),default=None),
             'worker_time_seconds':sum(a['time'] for a in audits)}
    for kind in CATEGORIES:
        (root/'failures'/kind).mkdir(parents=True,exist_ok=True)
    write(root/'failures/categories.json',CATEGORIES)
    write(root/'metrics.json',{'schema_version':1,'units':{'time':'s','peak_memory':'bytes','length':'m'},
                               'statistics':stats,'rows':rows})
    with (root/'results.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=['case_id','run_id',*FIELDS])
        writer.writeheader()
        writer.writerows({**r,'params':json.dumps(r['params'],sort_keys=True)} for r in rows)
    write_summary(root,stats,rows)


def write_summary(root, stats, rows):
    counts = stats['decisions']
    lines = ['# GRE MVP 总结','',
             f"在 {stats['inputs']} 份原生 Blender 输入上实测 {stats['cases']} 个相容算法与标注组合。"
             + '状态分布为 '+ '、'.join(f'`{name}` {count} 组' for name,count in sorted(counts.items()))+'。','',
             f"已复核 {stats['verified_commits']} 次提交重开和 {stats['verified_rollbacks']} 次原件回滚。"
             f"所有工作进程累计时间 {stats['worker_time_seconds']:.2f} 秒。",'']
    if stats['peak_memory_bytes'] is not None:
        lines.extend((f"单进程峰值内存 {stats['peak_memory_bytes']/2**20:.2f} MiB。",''))
    lines.extend(('## 代表结果','',
                  '| 资产与算法 | 独立参考 RMS 前 → 后（mm） | 结果 | 说明 |',
                  '| --- | ---: | --- | --- |'))

    def number(value):
        return f'{1000*value:.6f}' if value is not None else 'N/A'

    for item in rows:
        if item['algorithm'] not in ('plane','plane_family','bilateral') or not item['target_rms_before']:
            continue
        description = item['failure_reason'] or ('P2 独立验收通过' if item['decision']=='accepted' else '')
        lines.append(f"| `{item['case_id']}` | {number(item['target_rms_before'])} → {number(item['target_rms_after'])} | "
                     f"`{item['decision']}` | {description.replace('|','/')} |")
    lines.extend(('', '实测数字来自本目录 `metrics.json` 的 `statistics` 与 `rows`。RMS 为对照干净参考的稳定 ID 顶点误差。'
                  '内存按 Windows 工作集高水位或 POSIX 常驻集高水位记账。提前拒绝或工人异常的未测字段保留为空。','',
                  '## 失败归档','',
                  '| 类别 | 归档数 |','| --- | ---: |'))
    for kind,label in CATEGORIES.items():
        lines.append(f"| {label} | {stats['failures_by_category'][kind]} |")
    lines.extend(('', '`failures/` 的记录链接到对应审计、冻结计划、可用预览和运行日志。零条表示本批未归档此类样本。','',
                  '## 范围与下一阶段','',
                  '- 已接通分析、手工计划绑定、固定拓扑求解、独立门禁、预览及显式提交与回滚。批跑复用木桌平面、柜子平面族和花瓶局部滤波。',
                  '- 验证范围为程序化 Blender 夹具和单网格。真实 AI 资产、跨对象碰撞、跨 Blender 版本和人工盲评仍待验证。可选水杯圆柱拟合留待后续。',
                  '- Agent 语义规划应先输出可审阅的区域、约束与依据，由相同严格 E3 校验器复核。',
                  '- 杯子先做圆柱、圆锥模型竞争，再对轴向覆盖与歧义设硬拒绝。',
                  '- 外部 Worker 可沿当前快照与补丁协议拆分。先测集中 I/O、峰值内存和大网格耗时，再调整进程粒度。','',
                  '```powershell',"python -m gre.pipeline batch --blender $env:HELLO_3DNEKO_BLENDER_EXE --output-dir ./gre_p5_reproduced",'```',''))
    with (root/'GRE_MVP_总结.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    for name in ('inspect','run','batch','rollback'):
        p = actions.add_parser(name)
        p.add_argument('--blender',required=True,type=Path)
        p.add_argument('--output-dir',required=True,type=Path)
        p.add_argument('--timeout',type=float,default=60)
        if name=='batch':
            p.add_argument('--data',type=Path,default=ROOT/'gre_testdata')
            p.add_argument('--catalog',type=Path)
        else:
            p.add_argument('--case-id',required=True)
            if name=='rollback':
                p.add_argument('--audit-id',required=True)
                p.add_argument('--expected-current-hash',required=True)
            else:
                for key in ('source','annotations','reference'):
                    p.add_argument(f'--{key}',required=True,type=Path)
                p.add_argument('--truth',type=Path)
                if name=='run':
                    p.add_argument('--plan',required=True,type=Path)
                    p.add_argument('--commit',action='store_true')
                else:
                    p.add_argument('--semantic',required=True,choices=SEMANTICS)
                    p.add_argument('--algorithm',required=True)
    args = parser.parse_args()
    if not 0<args.timeout<=86400:
        parser.error('Timeout must be finite and between 0 and 86400 seconds')
    blender,root = args.blender.resolve(strict=True),args.output_dir.resolve()
    if args.action=='batch':
        if root.exists() and any(root.iterdir()):
            parser.error('Batch output directory must be empty')
        data = args.data.resolve(strict=True)
        cases = enumerate_cases(data,args.catalog or data/'p5_catalog.json')
        root.mkdir(parents=True,exist_ok=True)
        reports = [attempt(blender,root,req,args.timeout) for req in cases]
        summarize(root,reports)
        return int(any(item['decision']=='solver_failed' for item in reports))
    req = {key: (str(value.resolve()) if isinstance(value,Path) else value)
           for key,value in vars(args).items() if key not in {'blender','output_dir','timeout'}}
    if args.action=='inspect' and args.algorithm not in SEMANTICS[args.semantic]:
        parser.error('Incompatible algorithm and semantic class')
    report = attempt(blender,root,req,args.timeout)
    return int(report['decision'] in {'rejected','unsupported','solver_failed'})


if __name__=='__main__':
    raise SystemExit(main())
