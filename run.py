"""HSIR seed SFT, verify-then-exit sampling, scoring and iterative training."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from release_utils import Runner, common_arguments, validate_arguments

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common_arguments(p)
    p.add_argument('--task', choices=['MedQA', 'GSM8K'], required=True)
    p.add_argument('--seed-data', required=True, help='SFT seed JSON array; see docs/DATA.md')
    p.add_argument('--preference-seed', help='Optional seed pairs for DPO (chosen/rejected required)')
    p.add_argument('--template', default='qwen')
    p.add_argument('--sft-lr', type=float, default=1e-5)
    p.add_argument('--dpo-lr', type=float, default=1e-6)
    a = p.parse_args()
    validate_arguments(a)
    r = Runner(a, ROOT)
    r.start()
    source = ROOT/'src'/a.task
    prepared = r.out/'prepared_data'
    if not a.dry_run:
        prepared.mkdir(exist_ok=True)
    seed_columns = ({'prompt': 'instruction', 'query': 'input', 'response': 'output', 'system': 'system'}
                    if a.task == 'MedQA' else {'prompt': 'problem', 'response': 'reasoning_solution', 'system': 'system'})
    preference_columns = {'prompt': 'instruction', 'system': 'system', 'chosen': 'chosen', 'rejected': 'rejected'}
    if a.task == 'MedQA':
        preference_columns['query'] = 'input'
    r.register('seed', a.seed_data, seed_columns)
    if a.preference_seed:
        r.register('preference_seed', a.preference_seed, preference_columns, ranking=True)
    current = r.out/'models'/'seed'
    r.train(a.base_model, 'seed', current, 'sft', a.sft_lr)

    def shards(script, data, folder, count=None):
        commands = []
        gpus = a.gpus.split(',')
        for i, gpu in enumerate(gpus):
            if script.startswith('step4'):
                args = ['--model_name_or_path', current, '--data_path', data, '--save_path', folder/f'part{i}.json',
                        '--start_idx', i, '--end_idx', len(gpus)]
                filename = f'part{i}.json'
            else:
                args = [current, data, folder, 1.0, i, len(gpus), count]
                filename = f'part_{i}.json'
            command = [str(x) for x in [a.python, source/script, *args]]
            commands.append((gpu, command, filename))
            print(f'CUDA_VISIBLE_DEVICES={gpu} '+shlex.join(command), flush=True)
        if not a.dry_run:
            folder.mkdir(parents=True, exist_ok=False)
            processes = []
            try:
                for gpu, command, _ in commands:
                    processes.append(subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)))
                failures = [proc.wait() for proc in processes]
                if any(failures):
                    raise RuntimeError(f'Shard workers failed: {failures}; outputs were not merged')
            finally:
                for proc in processes:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait()
            merged = []
            for _, _, filename in commands:
                merged.extend(json.loads((folder/filename).read_text(encoding='utf-8')))
            (folder/'merged.json').write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')
        return folder/'merged.json'

    for iteration in range(1, a.iterations+1):
        work = r.out/f'iteration_{iteration}'
        samples = shards('step1_rejection_sampling.py', a.train_data, work/'sampling', 10)
        verified = shards('step2_verify-then-exit_sampling.py', samples, work/'verified', 5)
        split = work/'right_and_wrong.json'
        r.script(f'{a.task}/step3_splict_right_wrong.py', verified.parent, verified.stem, split)
        scored = shards('step4_cal_indiv_score.py', split, work/'scores')
        filtered = work/'filtered.json'
        r.script(f'{a.task}/step5_filter_data.py', '--data_path', scored, '--save_path', filtered, '--indiv_score_thre', -0.5)
        dataset = f'hsir_{a.mode}_{iteration}'
        training_file = prepared/f'{dataset}.json'
        filename = 'step6_sft_data_prepare.py' if a.mode == 'sft' else 'step6_dpo_dataset_prepare.py'
        r.script(f'{a.task}/{filename}', filtered.parent, filtered.stem, training_file)
        r.register(dataset, training_file, seed_columns if a.mode == 'sft' else preference_columns, ranking=a.mode=='dpo')
        init = a.base_model if a.mode == 'sft' else current
        mixture = f'seed,{dataset}' if a.mode == 'sft' else (f'preference_seed,{dataset}' if a.preference_seed else dataset)
        current = r.out/'models'/dataset
        r.train(init, mixture, current, a.mode, a.sft_lr if a.mode=='sft' else a.dpo_lr)


if __name__ == '__main__':
    main()
