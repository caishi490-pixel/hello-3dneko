# SPDX-License-Identifier: Apache-2.0
"""Validate the E3 input and process-isolation behavior without Blender."""

import copy
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gre.pipeline import row, run_child
from gre.plan import PlanError, bind, read_strict, validate


def manual():
    return dict(schema_version=1,asset_id='fixture',source_hash='a'*64,region_face_ids=[1],editable_vertex_ids=[1,2],
                protected_vertex_ids=[3,4],coordinate_space='object_local',unit_scale=1.0,semantic_class='plane',
                evidence_refs={'annotations_sha256':'b'*64,'reference_sha256':'c'*64},constraints=[],algorithm='plane',
                parameters={},max_displacement={'value':.001,'unit':'m'},qa_profile={},random_seed=42,approval_state='approved')


def binding(plan):
    return bind(plan,source_hash='a'*64,asset_id='fixture',unit_scale=1.0,evidence=manual()['evidence_refs'],
                source_vid={1,2,3,4},source_fid={1,2},allowed={1,2},faces={1},relations=[])


class PlanTest(unittest.TestCase):
    def test_manual_schema_inserts_known_defaults(self):
        self.assertEqual(binding(manual())['parameters']['max_planarity_ratio'],.005)

    def test_rejects_unknown_fields_bad_numbers_missing_units_and_empty_region(self):
        variations = [{'surprise':1},{'parameters':{'unknown':1}},{'unit_scale':None},{'region_face_ids':[]},
                      {'max_displacement':{'value':.1}},{'max_displacement':{'value':float('nan'),'unit':'m'}},
                      {'random_seed':True},{'qa_profile':{'uv_max':1}},{'editable_vertex_ids':[1,3]}]
        for change in variations:
            with self.subTest(change=change), self.assertRaises(PlanError):
                validate({**manual(),**change})

    def test_against_frozen_source_and_footprint(self):
        for change in ({'editable_vertex_ids':[1,99]},{'protected_vertex_ids':[3]}, {'region_face_ids':[2]},
                       {'source_hash':'d'*64},{'approval_state':'draft'}):
            with self.subTest(change=change), self.assertRaises(PlanError):
                binding({**manual(),**change})
        altered = copy.deepcopy(manual())
        altered['evidence_refs']['reference_sha256'] = 'd'*64
        with self.assertRaises(PlanError):
            binding(altered)

    def test_json_parser_rejects_duplicate_keys_and_non_finite_encodings(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'test.json'
            for message in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e999}','{"a":Infinity}'):
                path.write_text(message,encoding='utf-8')
                with self.subTest(message=message), self.assertRaises(PlanError):
                    read_strict(path)

    def test_timed_out_and_crashed_children_leave_next_child_runnable(self):
        with TemporaryDirectory() as folder:
            parent = Path(folder)
            jobs = [('import time; time.sleep(30)',.2,'worker_timeout'),('raise SystemExit(7)',5,'worker_exit_7'),
                    ('print("next-child-ok")',5,None)]
            for i,(code,seconds,expected) in enumerate(jobs):
                log = parent/f'{i}.log'
                _,reason = run_child([sys.executable,'-c',code],log,seconds)
                self.assertEqual(reason,expected)
            self.assertIn('next-child-ok',(parent/'2.log').read_text())

    def test_unmeasured_values_are_not_reported_as_zero(self):
        result = row({'decision':'solver_failed','failure_reason':'worker_timeout'})
        self.assertIsNone(result['target_rms_after'])
        self.assertIsNone(result['intersections_new'])
        self.assertIsNone(result['peak_memory'])
