#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/admin/qwen38_soak_72h_20260820')
OUT=ROOT/'reasoning_contract_validation.json'
spec=importlib.util.spec_from_file_location('soak',ROOT/'run_soak.py')
soak=importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(soak)

def compact(response):
    if not response:
        return None
    choice=response.get('choices',[{}])[0]
    return {'message':choice.get('message',{}),'finish_reason':choice.get('finish_reason'),'timings':response.get('timings',{})}

def main():
    state={'started_at_utc':soak.utc(),'modes':{}}
    for quant in ('q4','q6'):
        log_path=ROOT/f'reasoning_contract_{quant}.log'
        with log_path.open('w',encoding='utf-8') as log:
            process=subprocess.Popen(soak.server_command(quant),stdout=log,stderr=subprocess.STDOUT,text=True)
            try:
                load=soak.wait_ready(process,log_path)
                preflight=soak.preflight(soak.MODELS[quant]['alias'])
                cases=[]
                for index in range(len(soak.REASONING)):
                    case=soak.classify('reasoning',index)
                    passed,detail,response,wall,attempts=soak.score_case(case,None,soak.MODELS[quant]['alias'])
                    cases.append({'index':index,'case_id':case['id'],'passed':passed,'detail':detail,'wall_seconds':wall,'attempts':attempts,'response':compact(response)})
                    print(f'{quant} reasoning {index+1}/5 passed={passed}',flush=True)
                state['modes'][quant]={'load_seconds':load,'preflight':preflight,'cases':cases,'passed':sum(c['passed'] for c in cases),'total':len(cases)}
            finally:
                soak.stop_server(process)
        OUT.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    state['completed_at_utc']=soak.utc()
    state['all_passed']=all(m['passed']==m['total'] for m in state['modes'].values())
    OUT.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    if not state['all_passed']:
        raise SystemExit(1)
if __name__=='__main__': main()
