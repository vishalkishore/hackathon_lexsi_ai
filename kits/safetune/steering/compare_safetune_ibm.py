"""Compare the actual cached native behavior configurations on identical prompts."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

from original_cast_data import read,sha,write_json,write_rows
from original_cast_run import ROOT,environment,load_config,load_model


ARMS=[('ibm_native',0.3),('safetune_native',0.3)]


def generate(args):
    import torch
    from activation_steering import MalleableModel,SteeringVector
    from safetune.steer import CAAModel
    from safetune._refusal_helpers import _get_decoder_layers
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'steering/original_cast_100'
    config=load_config(data)
    cache=ROOT/'results/steering/pilot_calibration.pt'
    ibm_path=ROOT/'results/steering/cast_policy_context_20261106/explanation_policy_on.svec'
    requests=read(ROOT/'results/steering/cast_train_dev_20261106/requests.jsonl')
    assert len(requests)==20 and len({r['id'] for r in requests})==20
    assert all(r['condition']=='B' and r['policy_state']=='on' for r in requests)
    saved=torch.load(cache,map_location='cpu',weights_only=True)
    assert saved['metadata']['model']==config['model']
    assert saved['metadata']['model_revision']==config['revision']
    assert set(saved['vectors'])==set(config['behavior_layers'])
    ibm=SteeringVector.load(str(ibm_path))
    write_rows(out/'requests.jsonl',requests)
    write_json(out/'settings.json',{'phase':'native_vector_diagnostic','arms':ARMS})
    write_json(out/'config.json',{**config,'strengths':[0.3],'arms':ARMS,'gate':'forced ON in both native implementations',
        'ibm_vector_sha256':sha(ibm_path),'safetune_cache_sha256':sha(cache),
        'prompts_sha256':sha(ROOT/'results/steering/cast_train_dev_20261106/requests.jsonl'),
        'safetune_version':importlib.metadata.version('safetune'),
        'safetune_provenance':saved['metadata'],
        'ibm_provenance':'60 same-prompt policy-ON explanation/code contrasts; final-token pca_pairwise',
        'safetune_provenance_note':'24 mixed-policy request calibration cases; raw CAA mean-difference vector, not the IBM task-specific fit',
        'dtype_comparison':'FP16 shared; cached SafeTune vector was extracted in BF16',
        'ibm_site':'block input 14–18; alpha .3',
        'safetune_site':'block output 14–18; native CAAModel always-on alpha .3',
        'interpretation':'Comparison of saved native configurations; provenance, site and effective scale differ. Not a causal vector-direction comparison.'})
    write_json(out/'environment.json',environment(config))
    write_json(out/'vector_norms.json',{str(l):{'ibm_norm':float(torch.tensor(ibm.directions[l]).norm()),
        'safetune_norm':float(saved['vectors'][l].norm()),
        'ibm_nominal_delta_norm':.3*float(torch.tensor(ibm.directions[l]).norm()),
        'safetune_nominal_delta_norm':.3*float(saved['vectors'][l].norm())} for l in config['behavior_layers']})
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['compare_safetune_ibm.py','original_cast_run.py','cast_policy_context_eval.py','original_cast_eval.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    import inspect
    source=Path(inspect.getfile(CAAModel));shutil.copy2(source,snapshot/'safetune_caa.py.txt')
    shutil.copy2(ROOT/'uv.lock',snapshot/'uv.lock')
    model,tokenizer=load_model(config,args.device)
    write_json(out/'audit_plan.json',{'ids':[r['id'] for r in requests], 'arms':[a for a,_ in ARMS],
        'n':40,'scope':'assistant review of code/assistance/refusal presence; not human validation'})

    def one(row,method,strength,limit):
        handles=[];before={};counts=defaultdict(int);samples=[];wrapper=None;steering=None
        def before_hook(l,value):
            h=value[0] if isinstance(value,tuple) else value
            counts[l]+=1;before[l]=h[0,-1].detach().float().clone()
        def after_hook(l,value):
            h=value[0] if isinstance(value,tuple) else value
            original=before[l];delta=h[0,-1].detach().float()-original
            samples.append({'layer':l,'call':counts[l],'input_or_output_norm':original.norm().item(),
                'actual_delta_norm':delta.norm().item(),'relative_delta':(delta.norm()/original.norm()).item()})
        if method=='ibm_native':
            wrapper=MalleableModel(model,tokenizer)
            wrapper.reset_leash_to_default()
            wrapper.steer(behavior_vector=ibm,behavior_layer_ids=config['behavior_layers'],
                behavior_vector_strength=strength,apply_behavior_on_first_call=True,
                use_ooi_preventive_normalization=False)
            for l in config['behavior_layers']:
                block=model.model.layers[l]
                handles.append(block.register_forward_pre_hook(lambda module,inputs,l=l:before_hook(l,inputs[0])))
                handles.append(block.layer.register_forward_pre_hook(lambda module,inputs,l=l:after_hook(l,inputs[0])))
        else:
            blocks=_get_decoder_layers(model)
            for l in config['behavior_layers']:
                handles.append(blocks[l].register_forward_hook(lambda module,inputs,value,l=l:before_hook(l,value)))
            steering=CAAModel(model,vectors=saved['vectors'],strength=strength)
            steering.install()
            for l in config['behavior_layers']:
                handles.append(blocks[l].register_forward_hook(lambda module,inputs,value,l=l:after_hook(l,value)))
        enc=tokenizer.apply_chat_template(row['messages'],tokenize=True,return_dict=True,
            add_generation_prompt=True,return_tensors='pt').to(model.device)
        start=time.monotonic()
        try:
            with torch.inference_mode():
                result=model.generate(**enc,max_new_tokens=limit,do_sample=False,repetition_penalty=1.,
                    use_cache=True,pad_token_id=tokenizer.eos_token_id)
        finally:
            for h in handles:h.remove()
            if steering:steering.remove()
            if wrapper:wrapper.reset_leash_to_default();wrapper.unwrap()
        tokens=result[0,enc['input_ids'].shape[1]:].tolist()
        eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
        assert len(samples)==len(tokens)*5
        return {**row,'method':method,'strength':strength,'gate_mode':'forced_on',
            'generated_text':tokenizer.decode(tokens,skip_special_tokens=True).strip(),
            'output_tokens':tokens,'input_token_ids':enc['input_ids'][0].tolist(),
            'truncated':len(tokens)==limit and tokens[-1] not in eos,
            'elapsed_seconds':time.monotonic()-start},samples

    smoke=[]
    for method,strength in ARMS:
        for row in requests[:3]:
            record,samples=one(row,method,strength,64)
            assert record['generated_text'] and all(s['actual_delta_norm']>0 for s in samples)
            smoke.append(record)
    write_rows(out/'smoke.jsonl',smoke)
    smoke_lines=['# Paired smoke outputs', '',
        '64-token capped smoke outputs only; incomplete responses, not final performance measurements.']
    for sid in [r['id'] for r in requests[:3]]:
        smoke_lines += ['',f'## {sid}']
        for row in [r for r in smoke if r['id']==sid]:
            smoke_lines += ['',f'### {row["method"]}, alpha={row["strength"]}', '',row['generated_text']]
    (out/'smoke_examples.md').write_text('\n'.join(smoke_lines)+'\n')
    print('Six smoke generations and active perturbation checks passed',flush=True)
    with (out/'responses.jsonl').open('x') as stream,(out/'norms.jsonl').open('x') as norm_stream:
        for method,strength in ARMS:
            for i,row in enumerate(requests):
                record,samples=one(row,method,strength,2048)
                stream.write(json.dumps(record)+'\n');stream.flush()
                norm_stream.write(json.dumps({'id':row['id'],'method':method,'strength':strength,
                    'site':'input' if method=='ibm_native' else 'output','samples':samples})+'\n');norm_stream.flush()
                print(f'{method} {i+1}/20 {row["id"]} tokens={len(record["output_tokens"])}',flush=True)
    rows=read(out/'responses.jsonl')
    example_ids=['ALG001_B','ALG051_B','ALG081_B']
    lines=['# Verbatim outputs on identical prompts','',
        'All outputs below are model responses, not endorsements of their claims. Both arms are forced ON.',
        'IBM alpha .3 at block inputs; SafeTune alpha .3 at block outputs. Shared Aya FP16 model.']
    for sid in example_ids:
        prompt=next(r for r in requests if r['id']==sid)
        lines += ['',f'## {sid}', '', 'User request:', '', prompt['user']]
        for row in [r for r in rows if r['id']==sid]:
            lines += ['',f'### {row["method"]}, alpha={row["strength"]}', '',row['generated_text']]
    (out/'examples.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'generation_complete.json',{'responses':40,'smoke':6,
        'safetune_cache_unchanged':sha(cache)==json.loads((out/'config.json').read_text())['safetune_cache_sha256']})


def report(args):
    from original_cast_eval import summarize
    out=args.out;rows=read(out/'responses.jsonl');judgments=read(out/'judgments.jsonl')
    assert len(rows)==len(judgments)==40
    summary=summarize(rows,rows,judgments)
    norm_data=defaultdict(list)
    for record in read(out/'norms.jsonl'):
        for sample in record['samples']:
            norm_data[(record['method'],sample['layer'],sample['call']==1)].append(sample)
    import statistics
    norms={str(k):{'n':len(v),'mean_delta_norm':statistics.mean(s['actual_delta_norm'] for s in v),
        'mean_relative_delta':statistics.mean(s['relative_delta'] for s in v)} for k,v in norm_data.items()}
    write_json(out/'metrics.json',{'summary':summary,'actual_perturbations':norms,'audit_status':'pending'})
    lines=['# Native SafeTune versus IBM behavior configurations','',
        'Same 20 prompts, shared Aya FP16, forced ON. Model-only labels; assistant audit pending.',
        'Different calibration provenance, scale and native injection site prevent a vector-direction-only attribution.','',
        '| Method | Strength | Leakage | Conceptual assistance | Joint success | Truncated |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in summary:
        m=arm['metrics']
        def cell(k):
            v=m[k];return f"{v['positive']}/{v['denominator']} (missing {v['missing']})"
        lines.append(f"| {arm['method']} | {arm['strength']} | {cell('leakage_restricted')} | {cell('helpfulness_restricted')} | {cell('joint_restricted')} | {cell('truncation')} |")
    lines += ['', 'Actual per-layer prefill/decode norm summaries: metrics.json. Verbatim paired outputs: examples.md.',
        'Code presence is independent of correctness; absence of code in incoherent or unrelated refusals does not imply joint success.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['generate','report'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    (generate if args.command=='generate' else report)(args)
