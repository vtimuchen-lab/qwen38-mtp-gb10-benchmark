#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, subprocess, urllib.request
from pathlib import Path
ROOT=Path('/home/admin/qwen38_soak_72h_20260820')
OUT=ROOT/'long_contract_validation.json'
spec=importlib.util.spec_from_file_location('soak',ROOT/'run_soak.py')
soak=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(soak)

def tokenize(text):
    req=urllib.request.Request(f'http://127.0.0.1:{soak.PORT}/tokenize',data=json.dumps({'content':text,'add_special':False}).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=300) as response:
        return len(json.loads(response.read().decode())['tokens'])

def compact(response):
    if not response: return None
    choice=response.get('choices',[{}])[0]
    return {'message':choice.get('message',{}),'finish_reason':choice.get('finish_reason'),'timings':response.get('timings',{})}

def main():
    state={'started_at_utc':soak.utc(),'modes':{}}
    for quant in ('q4','q6'):
        log_path=ROOT/f'long_contract_{quant}.log'
        with log_path.open('w',encoding='utf-8') as log:
            process=subprocess.Popen(soak.server_command(quant),stdout=log,stderr=subprocess.STDOUT,text=True)
            try:
                load=soak.wait_ready(process,log_path); preflight=soak.preflight(soak.MODELS[quant]['alias']); cases=[]
                for index,category in enumerate(('long8','long32','long60')):
                    case=soak.classify(category,index)
                    raw_tokens=tokenize(case['messages'][0]['content'])
                    passed,detail,response,wall,attempts=soak.score_case(case,None,soak.MODELS[quant]['alias'])
                    guard=raw_tokens < soak.CTX_SIZE-512
                    cases.append({'category':category,'raw_tokens':raw_tokens,'context_guard':guard,'passed':passed,'detail':detail,'wall_seconds':wall,'attempts':attempts,'response':compact(response)})
                    print(f'{quant} {category} raw_tokens={raw_tokens} guard={guard} passed={passed}',flush=True)
                state['modes'][quant]={'load_seconds':load,'preflight':preflight,'cases':cases,'passed':sum(c['passed'] and c['context_guard'] for c in cases),'total':len(cases)}
            finally: soak.stop_server(process)
        OUT.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    state['completed_at_utc']=soak.utc(); state['all_passed']=all(m['passed']==m['total'] for m in state['modes'].values()); OUT.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    if not state['all_passed']: raise SystemExit(1)
if __name__=='__main__': main()
