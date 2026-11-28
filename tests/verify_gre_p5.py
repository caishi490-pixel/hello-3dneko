# SPDX-License-Identifier: Apache-2.0
"""Verify provenance, audit, native artifacts and optionally numerical replay."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gre.pipeline import CATEGORIES, FIELDS, enumerate_cases, read_strict, sha, write_summary


def verify(root,data):
    metrics = read_strict(root/'metrics.json')
    stats,rows = metrics['statistics'],metrics['rows']
    cases = enumerate_cases(data,data/'p5_catalog.json')
    assert {r['case_id'] for r in rows} == {case['case_id'] for case in cases}
    assert stats['cases']==len(rows)==31 and stats['inputs']==9
    assert Counter(r['decision'] for r in rows)==stats['decisions']=={'accepted':3,'needs_review':13,'no_action':13,'rejected':2}
    defects = Counter()
    transactions = []
    with (root/'results.csv').open(encoding='utf-8-sig',newline='') as file:
        csv_rows = list(csv.DictReader(file))
    assert len(csv_rows)==len(rows) and list(csv_rows[0])==['case_id','run_id',*FIELDS]
    for case,row,csv_row in zip(cases,rows,csv_rows):
        run_id = row['run_id']
        audit = read_strict(root/'audit'/f'{run_id}.json')
        assert audit['summary_row']=={key:row[key] for key in FIELDS}
        assert csv_row['case_id']==case['case_id']==row['case_id'] and csv_row['run_id']==run_id
        assert row['input_hash']==sha(Path(case['source'])) and row['annotations_hash']==sha(Path(case['annotations']))
        folder = root/audit['paths']['run']
        assert (folder/'plan.json').is_file() and (root/audit['paths']['log']).is_file()
        assert set(audit['versions'])=={'blender','python','numpy'} and row['peak_memory']>0 and row['time']>0
        phases = [phase['name'] for phase in audit['phases']]
        assert phases[:3]==['analyze','plan','solve']
        if row['decision']=='accepted':
            assert phases[3:5]==['validate','preview']
            assert row['topology_equal'] is row['uv_hash_equal'] is True
            assert row['intersections_new']==0 and audit['qa']['checks']['locked']['changed']==0
            assert sha(folder/'candidate.blend')==audit['candidate_sha256']
            assert (folder/'preview.svg').is_file() and (folder/'preview.npz').is_file()
        else:
            assert not (folder/'candidate.blend').exists()
        if audit.get('commit'):
            transactions.append(audit)
        category = audit.get('failure_category')
        if category:
            defects[category] += 1
            record = read_strict(root/'failures'/category/f'{run_id}.json')
            assert record['reason']==row['failure_reason'] and record['category']==category
            assert (root/record['audit']).is_file() and (root/record['log']).is_file()
    assert stats['failures_by_category']=={key:defects[key] for key in CATEGORIES}
    assert stats['verified_commits']==stats['verified_rollbacks']==len(transactions)==1
    trial = transactions[0]
    assert trial['commit_verified'] and trial['rollback_verified']
    assert sha(root/'working'/trial['commit']['working_file'])==trial['input_hash']==trial['rollback']['restored_file_sha256']
    with TemporaryDirectory() as folder:
        write_summary(Path(folder),stats,rows)
        assert (Path(folder)/'GRE_MVP_总结.md').read_bytes()==(root/'GRE_MVP_总结.md').read_bytes()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--repeat',type=Path)
    args = parser.parse_args()
    original = verify(args.root.resolve(),args.data.resolve())
    if args.repeat:
        replay = verify(args.repeat.resolve(),args.data.resolve())
        variable = {'run_id','time','peak_memory'}
        strip = lambda r: {key:value for key,value in r.items() if key not in variable}
        assert [strip(r) for r in original['rows']]==[strip(r) for r in replay['rows']]
        for a,b in zip(original['rows'],replay['rows']):
            left = read_strict(args.root/'audit'/f"{a['run_id']}.json")
            right = read_strict(args.repeat/'audit'/f"{b['run_id']}.json")
            assert left.get('candidate_position_hash')==right.get('candidate_position_hash')
        print('P5 REPLAY: numerical rows, decisions and released position fingerprints match',flush=True)
    print('P5 ACCEPTANCE: PASS',flush=True)


if __name__=='__main__':
    main()
