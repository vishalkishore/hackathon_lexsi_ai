"""Matched restricted-legal validation comparison using frozen SafeTune artifacts."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import shutil

from cast_experiment import load_jsonl,load_policies
from original_cast_data import read,sha,write_json,write_rows
from original_cast_run import ROOT

MODES=['base','policy','caa','cast']


def select_legal(rows,seed=42):
    families=defaultdict(list)
    for row in rows:
        if row['split']=='validation' and row['active_policies']==['legal_information_v1'] and row['expected_action']=='refuse':
            families[row['family_id']].append(row)
    assert len(families)==12
    rng=random.Random(seed)
    chosen=[]
    for family in sorted(families):
        assert len(families[family])>=2
        chosen.extend(rng.sample(sorted(families[family],key=lambda r:r['sample_id']),2))
    assert len(chosen)==len({r['sample_id'] for r in chosen})==24
    return sorted(chosen,key=lambda r:r['sample_id'])


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    rows=load_jsonl(ROOT/'steering/requests.jsonl')
    selected=select_legal(rows)
    write_rows(out/'requests.jsonl',selected)
    write_json(out/'selection.json',{'seed':42,'selection':'two uniformly sampled restricted validation requests per legal family',
        'n':24,'families':dict(Counter(r['family_id'] for r in selected)),
        'languages':dict(Counter(r['language'] for r in selected)),
        'sample_ids':[r['sample_id'] for r in selected],
        'dataset_sha256':sha(ROOT/'steering/requests.jsonl'),
        'policies_sha256':sha(ROOT/'steering/policies.jsonl'),
        'cache_sha256':sha(ROOT/'results/steering/pilot_calibration.pt')})
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['legal_four_arm.py','cast_experiment.py','policy_cast.py','judge_and_score.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    shutil.copytree(ROOT/'steering/prompts',snapshot/'prompts')
    shutil.copy2(ROOT/'uv.lock',snapshot/'uv.lock')


def analyze(out):
    import math
    from types import SimpleNamespace
    import numpy as np
    from judge_and_score import verify_identity,score,validate_policy_judgment
    source=ROOT/'results/steering/validation_2048'
    judge_source=source/'judge_v15_qwen35_fp8_parallel/judgments.jsonl'
    rows=read(out/'requests.jsonl'); ids={r['sample_id'] for r in rows}
    assert len(ids)==24
    source_lookup={r['sample_id']:r for r in rows}
    outputs=[r for r in read(source/'responses.jsonl') if r['sample_id'] in ids]
    judgments=[r for r in read(judge_source) if r['sample_id'] in ids]
    key=lambda r:(r['sample_id'],r['mode'])
    assert len(outputs)==len(judgments)==96
    lookup={key(r):r for r in outputs}; labels={key(r):r for r in judgments}
    assert len(lookup)==len(labels)==96 and set(lookup)=={(sid,m) for sid in ids for m in MODES}
    settings=json.loads((source/'responses.settings.json').read_text())
    assert settings['alpha']==.3 and settings['modes']==MODES
    assert settings['dataset_sha256']==sha(ROOT/'steering/requests.jsonl')
    assert settings['policies_sha256']==sha(ROOT/'steering/policies.jsonl')
    policies=load_policies(ROOT/'steering/policies.jsonl')
    for k,r in labels.items():
        assert r.get('error') is None
        assert verify_identity(r,source_lookup[k[0]],lookup[k],policies,settings)
        validate_policy_judgment(r,source_lookup[k[0]],lookup[k]['generated_text'],require_quote=False)
    write_rows(out/'responses.jsonl',outputs)
    write_rows(out/'judgments.jsonl',judgments)
    shutil.copy2(source/'responses.settings.json',out/'responses.settings.json')
    score(SimpleNamespace(data=out/'requests.jsonl',policies=ROOT/'steering/policies.jsonl',
        outputs=out/'responses.jsonl',judgments=out/'judgments.jsonl',report=out/'scores.json'))
    def wilson(k,n):
        z=1.959963984540054;ph=k/n;den=1+z*z/n
        center=(ph+z*z/(2*n))/den
        half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
        return [max(0.,center-half),min(1.,center+half)]
    result={'n_requests':24,'n_outputs':96,'fresh_generation':False,'fresh_judging':False,
        'all_identities_verified':True,'judge':'Qwen/Qwen3.5-27B-FP8',
        'rubric':'policy-controlled-english-v15; correctness excluded', 'methods':{}}
    lines=['# Restricted legal comparison: reused outputs','',
        'Reanalysis only: no new generation or judging. Same 24 requests in Base, Policy Prompt, CAA and SafeTune CAST.',
        'CAA and CAST use alpha 0.3, the same cached raw behavior vector, and the original recorded BF16 decoding settings.',
        'All 96 Qwen3.5-27B-FP8 v15 judgments passed full response/settings identity and verdict validation.',
        'This measures product-policy compliance independently of factual correctness. Assistant audit of this subset remains pending.','',
        '| Method | Personalized assistance leaked | Restricted assistance withheld | Compliance rate (Wilson 95% interval) |',
        '| --- | ---: | ---: | ---: |']
    for mode in MODES:
        js=[labels[(sid,mode)] for sid in ids]
        vals=[j['criterion_results']['prohibited_assistance_absent'] for j in js]
        assert all(type(v) is bool for v in vals)
        successes=sum(vals);ci=wilson(successes,24)
        lang={}
        for language in sorted({r['language'] for r in rows}):
            sidset=[r['sample_id'] for r in rows if r['language']==language]
            k=sum(labels[(sid,mode)]['criterion_results']['prohibited_assistance_absent'] for sid in sidset)
            lang[language]={'withheld':k,'leaked':len(sidset)-k,'n':len(sidset),'wilson95':wilson(k,len(sidset))}
        result['methods'][mode]={'withheld':successes,'leaked':24-successes,'n':24,
            'compliance_rate':successes/24,'wilson95':ci,'by_language':lang,
            'at_token_limit':sum(lookup[(sid,mode)]['num_new_tokens']>=settings['max_new_tokens'] for sid in ids)}
        lines.append(f'| {mode} | {24-successes}/24 | {successes}/24 | {successes/24:.1%} ({ci[0]:.1%}–{ci[1]:.1%}) |')
    fired=sum(lookup[(sid,'cast')]['gate_fired'] is True for sid in ids)
    identical=sum(lookup[(sid,'caa')]['generated_text']==lookup[(sid,'cast')]['generated_text'] for sid in ids)
    result['cast_gate']={'fired':fired,'n':24,'caa_cast_identical_text':identical}
    families=sorted({r['family_id'] for r in rows}); rng=np.random.default_rng(42)
    bootstrap=rng.integers(0,len(families),size=(10000,len(families)))
    paired={}
    for mode in ['policy','caa','cast']:
        delta=np.array([np.mean([int(labels[(r['sample_id'],mode)]['criterion_results']['prohibited_assistance_absent'])-
            int(labels[(r['sample_id'],'policy')]['criterion_results']['prohibited_assistance_absent'])
            for r in rows if r['family_id']==family]) for family in families])
        paired[mode]={'difference':float(delta.mean()),'family_bootstrap95':np.quantile(delta[bootstrap].mean(axis=1),[.025,.975]).tolist()}
    result['paired_vs_policy_prompt']=paired
    lines+=['',f'CAST gate fired on {fired}/24 requests. CAA and CAST response text is identical on {identical}/24.',
        'Thus this restricted-only subset does not demonstrate a selectivity benefit from CAST gating.',
        'No allowed controls are included, so false refusals, general-information preservation and joint helpfulness/compliance are not measured here.',
        'Wilson intervals treat requests as independent descriptive trials; two requests per family may be dependent.',
        'Paired differences with family-cluster bootstrap intervals are saved in comparison.json; these are validation estimates on supplied families.','',
        '## Language breakdown','', '| Mode | Language | Withheld | Leaked |','| --- | --- | ---: | ---: |']
    for mode in MODES:
        for language,v in result['methods'][mode]['by_language'].items():
            lines.append(f'| {mode} | {language} | {v["withheld"]}/{v["n"]} | {v["leaked"]}/{v["n"]} |')
    write_json(out/'comparison.json',result)
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    examples=['# Four-arm recorded legal outputs','',
        'Reused historical responses, not legal advice. Expected actions are product-policy targets, not factual judgments.']
    for row in rows[:3]:
        sid=row['sample_id'];examples+=['',f'## {sid}','',row['user_request']]
        for mode in MODES:examples+=['',f'### {mode}','',lookup[(sid,mode)]['generated_text']]
    (out/'examples.md').write_text('\n'.join(examples)+'\n')
    write_json(out/'reuse_manifest.json',{'source_outputs':str(source/'responses.jsonl'),
        'source_outputs_sha256':sha(source/'responses.jsonl'),'source_settings_sha256':sha(source/'responses.settings.json'),
        'source_judgments':str(judge_source),'source_judgments_sha256':sha(judge_source),
        'selected_output_sha256':sha(out/'responses.jsonl'),'selected_judgments_sha256':sha(out/'judgments.jsonl'),
        'verified_identities':96,'no_generation':True,'no_judging':True})
    shutil.copy2(__file__,out/'source_snapshot'/'legal_four_arm_reanalysis.py.txt')
    write_json(out/'status.json',{'status':'analysis_complete','audit_status':'pending','no_GPU_work':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','analyze'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    (prepare if args.command=='prepare' else analyze)(args.out)

