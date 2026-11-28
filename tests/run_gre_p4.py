# SPDX-License-Identifier: Apache-2.0
"""Run the noisy and clean P4 experiments, regenerate inputs, and verify disk outputs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser()
    for name in ('blender','numpy-python','data','output-dir'):
        parser.add_argument(f'--{name}',required=True,type=Path)
    parser.add_argument('--no-images',action='store_true')
    args = parser.parse_args()
    data,out = args.data.resolve(strict=True),args.output_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory must be new or empty')
    out.mkdir(parents=True,exist_ok=True)
    inputs = sorted(path for path in data.glob('vase_*') if path.suffix in {'.blend','.json'})
    start = {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}

    def call(command):
        print(subprocess.list2cmdline(command),flush=True)
        subprocess.run(command,cwd=root,check=True)

    call([str(args.numpy_python.resolve(strict=True)),'-m','unittest','discover','-s','tests','-p','test_gre_p4.py','-v'])
    base = [str(args.blender.resolve(strict=True)),'-b','--factory-startup','-t','4','--python-exit-code','1','--python']
    for name in ('noisy','clean'):
        command = [*base,'gre/p4_cli.py','--','--source',str(data/f'vase_{name}.blend'),
                   '--reference',str(data/'vase_clean.blend'),'--marked',str(data/'vase_marked.json'),
                   '--unmarked',str(data/'vase_unmarked.json'),'--truth',str(data/'vase_truth.json'),
                   '--output-dir',str(out/name)]
        if name=='clean' or args.no_images:
            command.append('--no-images')
        call(command)
    noisy = json.loads((out/'noisy'/'metrics.json').read_text(encoding='utf-8'))['cases']
    marked,unmarked = (noisy[name]['methods'] for name in ('marked','unmarked'))
    for methods in (marked,unmarked):
        assert set(methods) == {'none','laplacian','taubin','bilateral'}
        assert methods['bilateral']['metrics']['clean_error_m']['rms'] < methods['none']['metrics']['clean_error_m']['rms']
        for row in methods.values():
            checks = row['p2']['checks']
            assert checks['integrity']['status'] == 'pass' and checks['locked']['changed']==0
    assert marked['bilateral']['status']=='accepted' and marked['bilateral']['metrics']['relief_coordinates_exact']
    assert unmarked['bilateral']['metrics']['relief_retention'] > unmarked['taubin']['metrics']['relief_retention']
    assert unmarked['bilateral']['status']=='needs_review' and not list((out/'noisy'/'unmarked').glob('*.blend'))
    outline = marked['bilateral']['metrics']
    assert outline['top_base_eccentricity_m'] == outline['clean_top_base_eccentricity_m'] > .01
    assert outline['centerline_max_change_from_clean_m'] < .00035 and outline['silhouette_support_p95_m'] < .00055
    assert not list((out/'clean').rglob('*.blend')), 'A change in the clean control was published'
    if not args.no_images:
        for name in ('before','after','wireframe','uv_checker','displacement'):
            assert (out/'noisy'/'marked'/'bilateral'/f'{name}.png').stat().st_size > 0

    with TemporaryDirectory(prefix='gre-p4-regenerate-') as folder:
        call([*base,'gre_testdata/vase/build_vase.py','--','--output-dir',folder,'--seed','42'])
        call([*base,'tests/gre_p4_disk.py','--','--data',str(data),'--repeated',folder,'--output',str(out/'noisy')])
    assert all(hashlib.sha256(path.read_bytes()).hexdigest()==sha for path,sha in start.items())
    print('P4 ACCEPTANCE: PASS',flush=True)


if __name__=='__main__':
    main()
