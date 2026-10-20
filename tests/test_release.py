import ast
import contextlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def script(self, task, name, *args):
        subprocess.run([sys.executable, str(ROOT/'src'/task/name), *map(str, args)], check=True, capture_output=True, text=True)

    def test_filter_preserves_list_and_constant_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            records=[dict(right_response=['a','b'], indiv_score=[1,1]),
                     dict(right_response=['c'], indiv_score=[2]),
                     dict(right_response=[], indiv_score=[]),
                     dict(right_response=['d','e'], indiv_score=[0,2])]
            (d/'input.json').write_text(json.dumps(records))
            for task in ['MedQA','GSM8K']:
                self.script(task, 'step5_filter_data.py', '--data_path', d/'input.json', '--save_path', d/'out.json')
                output=json.loads((d/'out.json').read_text())
                self.assertEqual(output[0]['right_response'], ['a','b'])
                self.assertEqual(output[0]['indiv_score'], [1,1])
                self.assertEqual(output[1]['indiv_score'], [2])
                self.assertEqual(output[3]['right_response'], ['e'])

    def test_split_keeps_recovered_candidates_and_bounds_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            record=dict(instruction='Synthetic', input='context', system='system', label='A',
                        rejection_sampling=['##Answer:\nB']*10+[['##Answer:\nA']*5])
            (d/'input.json').write_text(json.dumps([record]))
            self.script('MedQA','step3_splict_right_wrong.py',d,'input',d/'split.json')
            output=json.loads((d/'split.json').read_text())[0]
            self.assertEqual(len(output['right_response']),5)
            self.assertEqual(len(output['wrong_response']),5)
            self.script('MedQA','step6_dpo_dataset_prepare.py',d,'split',d/'pairs.json')
            pairs=json.loads((d/'pairs.json').read_text())
            self.assertEqual(len(pairs),5)
            self.assertTrue(all(pair['input']=='context' for pair in pairs))

    def test_refinement_prompts_do_not_accumulate(self):
        # Load only pure function definitions; GPU modules are intentionally not imported.
        source=ast.parse((ROOT/'src/MedQA/step2_verify-then-exit_sampling.py').read_text())
        functions=ast.Module(body=[node for node in source.body if isinstance(node,ast.FunctionDef)],type_ignores=[])
        scope={'re':re,'torch':types.SimpleNamespace(no_grad=contextlib.nullcontext)}
        exec(compile(functions,'sampling-functions','exec'),scope)
        captured=[]
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                captured.append(list(messages))
                return 'prompt'
        llm=types.SimpleNamespace(generate=lambda *a,**k:[])
        record=dict(instruction='Question',input='Context',system='System',label='A',
                    rejection_sampling=['The answer is A. ##Answer:\nB']*2)
        result=scope['inference'](llm,Tokenizer(),None,record,5)
        self.assertEqual(result[:2],record['rejection_sampling'])
        self.assertEqual([len(messages) for messages in captured],[3,3])
        self.assertIn('Context',captured[0][1]['content'])

    def test_plans_cover_both_tasks_and_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            for task in ['MedQA','GSM8K']:
                for mode in ['sft','dpo']:
                    out=Path(directory)/(task+mode)
                    result=subprocess.run([sys.executable,str(ROOT/'run.py'),'--base-model','base-model',
                        '--train-data','absent.json','--seed-data','absent-seed.json','--task',task,'--mode',mode,
                        '--llamafactory','absent-framework','--output',str(out),'--iterations','1','--dry-run'],
                        capture_output=True,text=True,check=True)
                    self.assertFalse(out.exists())
                    self.assertIn('step4_cal_indiv_score.py',result.stdout)
                    if mode=='sft':
                        self.assertIn('--dataset seed,hsir_sft_1',result.stdout)
                    else:
                        self.assertIn('--dataset hsir_dpo_1',result.stdout)
                        self.assertNotIn('--dataset seed,hsir_dpo_1',result.stdout)


if __name__=='__main__':
    unittest.main()
