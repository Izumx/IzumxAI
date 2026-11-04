"""Verify MBPP train references and combine with short numeric replay.

MBPP by Google Research / Austin et al. (2021), CC BY 4.0.
Only official train is used for gradients; validation stays separate.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import build_labeled_example, load_tokenizer
from .tasks import SYSTEM_CODE, SYSTEM_RU

ALLOWED_IMPORTS={'math','re','collections','itertools','functools','heapq','bisect','operator',
                 'string','typing','statistics','datetime','calendar','array','copy','decimal','fractions'}
FORBIDDEN_CALLS={'open','eval','exec','compile','__import__','input','breakpoint','exit','quit'}


def verify_reference(code, setup, tests):
    if not tests:
        return False
    try:
        tree=ast.parse(setup+'\n'+code+'\n'+'\n'.join(tests))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node,ast.Import) and any(n.name.split('.')[0] not in ALLOWED_IMPORTS for n in node.names):
            return False
        if isinstance(node,ast.ImportFrom) and (node.module or '').split('.')[0] not in ALLOWED_IMPORTS:
            return False
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return False
    with tempfile.TemporaryDirectory() as td:
        script=Path(td)/'reference.py'
        script.write_text(setup+'\n'+code+'\n'+'\n'.join(tests),encoding='utf-8')
        try:
            result=subprocess.run([sys.executable,'-I',str(script)],cwd=td,capture_output=True,timeout=5)
            return result.returncode==0
        except subprocess.TimeoutExpired:
            return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay',default='runs/stage2-20261103/curriculum.jsonl')
    ap.add_argument('--out',default='data/distill/stage3-smel')
    ap.add_argument('--tokenizer',default='models/izumxai-2b')
    ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--max-len',type=int,default=512)
    args=ap.parse_args()
    from datasets import load_dataset
    tokenizer=load_tokenizer(args.tokenizer)
    rng=random.Random(args.seed)
    human_raw=Path('.cache/eval/HumanEval.jsonl.gz').read_bytes()
    human=[json.loads(line) for line in gzip.decompress(human_raw).decode().splitlines() if line.strip()]
    excluded_names={row['entry_point'] for row in human}
    for path in Path('data/eval').glob('stage2_holdout.jsonl'):
        excluded_names.update(json.loads(line).get('function','') for line in path.read_text(encoding='utf-8').splitlines())
    skipped=Counter(); fingerprints={}; groups={}; used_names=set()
    for split in ('train','validation'):
        dataset=load_dataset('google-research-datasets/mbpp','full',split=split)
        fingerprints[split]=dataset._fingerprint
        accepted=[]
        for row in dataset:
            code=row['code']; setup=row['test_setup_code']; tests=row['test_list']+row['challenge_test_list']
            try:
                functions=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)]
            except SyntaxError:
                skipped['syntax']+=1; continue
            names={n.name for n in functions}
            if names & excluded_names or (split=='validation' and names & used_names):
                skipped['evaluation_name_overlap']+=1; continue
            calls={n.func.id for test in row['test_list'] for n in ast.walk(ast.parse(test))
                   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
            function=next((n for n in functions if n.name in calls),None)
            if function is None:
                skipped['no_entrypoint']+=1; continue
            prompt=(row['text']+'\n\nImplement '+function.name+'('+ast.unparse(function.args)+'). '
                    'Include required helpers and imports.\nExample:\n'+row['test_list'][0])
            messages=[{'role':'system','content':SYSTEM_CODE},{'role':'user','content':prompt},
                      {'role':'assistant','content':'```python\n'+(setup+'\n'+code).strip()+'\n```'}]
            if build_labeled_example(tokenizer,messages,args.max_len) is None:
                skipped['overlength']+=1; continue
            if not verify_reference(code,setup,tests):
                skipped['reference_test_failure_or_scope']+=1; continue
            accepted.append({'messages':messages,'task_type':'code','verified':True,
                             'source':'google-research-datasets/mbpp','source_split':split,
                             'source_task_id':row['task_id'],'license':'CC-BY-4.0',
                             'reference_test_count':len(tests)})
            if split=='train': used_names.update(names)
        groups[split]=accepted
        print(f'MBPP {split}: accepted {len(accepted)}/{len(dataset)}',flush=True)
    rng.shuffle(groups['validation'])
    groups['validation']=groups['validation'][:32]
    replay={}
    for line in Path(args.replay).read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        if row.get('task_type') not in {'math','logic'} or not row.get('verified') or row.get('expected') is None:
            continue
        prompt=next(m['content'] for m in row['messages'] if m['role']=='user')
        normalized=' '.join(prompt.split()).casefold()
        replay.setdefault(normalized,{'messages':[{'role':'system','content':SYSTEM_RU},
            {'role':'user','content':prompt},{'role':'assistant','content':'Ответ: '+str(row['expected'])}],
            'task_type':row['task_type'],'verified':True,'source':'verified_numeric_replay',
            'expected':str(row['expected'])})
    for kind in ('math','logic'):
        rows=[r for r in replay.values() if r['task_type']==kind]
        rng.shuffle(rows)
        if len(rows)<16: raise ValueError(f'insufficient replay {kind}')
        groups['validation'].extend(rows[:12])
        groups['train'].extend(rows[12:])
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    seen=set()
    for split in ('train','validation'):
        clean=[]
        for row in groups[split]:
            prompt=next(m['content'] for m in row['messages'] if m['role']=='user')
            key=' '.join(prompt.split()).casefold()
            if key in seen:
                if split=='validation': skipped['cross_split_duplicate']+=1
                continue
            seen.add(key); clean.append(row)
        groups[split]=clean
        name='val' if split=='validation' else 'train'
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in clean),encoding='utf-8')
    meta={'dataset':'google-research-datasets/mbpp','config':'full','license':'CC-BY-4.0',
          'source_url':'https://huggingface.co/datasets/google-research-datasets/mbpp',
          'attribution':'Google Research; Austin et al., Program Synthesis with Large Language Models (2021)',
          'changes':'Filtered official splits by execution, length and evaluation function-name overlap; added signature/example wrapper and numeric replay',
          'fingerprints':fingerprints,'seed':args.seed,'max_len':args.max_len,'skipped':dict(skipped),
          'counts':{s:dict(Counter(r['task_type'] for r in rows)) for s,rows in groups.items()},
          'humaneval_gzip_sha256':hashlib.sha256(human_raw).hexdigest()}
    (out/'data_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
