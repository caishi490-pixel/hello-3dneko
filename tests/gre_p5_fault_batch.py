# SPDX-License-Identifier: Apache-2.0
"""Run a real two-child batch where the first Blender worker cannot read its input."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blender',required=True,type=Path)
    parser.add_argument('--data',required=True,type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix='p5-isolation-') as folder:
        base = Path(folder)
        data = base/'fixtures'
        data.mkdir()
        for filename in ('table_clean.blend','table_annotations.json'):
            shutil.copyfile(args.data/'table'/filename,data/filename)
        catalog = {'schema_version':1,'families':[{'semantic':'plane','reference':'table_clean.blend',
                    'sources':['missing.blend','table_clean.blend'],'annotations':['table_annotations.json'],
                    'algorithms':['none'],'seed':42}]}
        (data/'p5_catalog.json').write_text(json.dumps(catalog),encoding='utf-8')
        out = base/'results'
        command = [sys.executable,'-m','gre.pipeline','batch','--blender',str(args.blender.resolve()),
                   '--data',str(data),'--output-dir',str(out)]
        result = subprocess.run(command,cwd=repo,check=False)
        rows = json.loads((out/'metrics.json').read_text(encoding='utf-8'))['rows']
        assert result.returncode==1 and len(rows)==2
        assert rows[0]['decision']=='solver_failed' and rows[0]['failure_reason'].startswith('worker_exit_')
        assert rows[1]['decision']=='no_action'
        print('P5 FAULT BATCH: failed Blender child was archived and the next asset completed',flush=True)


if __name__=='__main__':
    main()
