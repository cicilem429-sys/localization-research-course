"""Offline checks; never import historical paid-agent scripts."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def near(a,b,tol=0.00011):
    assert abs(a-b)<=tol,(a,b)

def main():
    data=json.loads((ROOT/'final_validation_results.json').read_text())
    assert data['seeds']==[11001,11002,11003,11004,11005]
    assert len(data['seed_results'])==5
    assert sum(x['sample_count'] for x in data['seed_results'])==600
    for method,s in data['summary'].items():
        values=[x['metrics'][method]['macro_cep90'] for x in data['seed_results']]
        assert values==s['macro_cep90_per_seed']
        mean=sum(values)/5
        near(mean,s['macro_cep90_mean'])
        near(math.sqrt(sum((v-mean)**2 for v in values)/4),s['macro_cep90_std'])
        near(sum(x['metrics'][method]['mean_error'] for x in data['seed_results'])/5,s['mean_error_average'])
        for item in data['seed_results']:
            metrics=item['metrics'][method]
            assert len(metrics['per_case_cep90'])==12
            near(sum(metrics['per_case_cep90'].values())/12,metrics['macro_cep90'])
    for item in data['seed_results']:
        assert min(item['metrics'],key=lambda m:item['metrics'][m]['macro_cep90'])=='adaptive'
    for row in csv.DictReader((ROOT/'final_validation_summary.csv').open(encoding='utf-8-sig')):
        for key,value in row.items():
            if key!='method':near(float(value),data['summary'][row['method']][key],1e-10)
    changed=[]
    for entry in json.loads((ROOT/'docs/source_manifest.json').read_text()):
        if hashlib.sha256((ROOT/entry['file']).read_bytes()).hexdigest()!=entry['sha256']:changed.append(entry['file'])
    assert sorted(changed)==['final_statistical_validation.py','research_agent_v6.py'],changed
    files=list(ROOT.glob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    for path in files:ast.parse(path.read_text())
    checks={'saved_metrics':'pass','csv':'pass','source_hashes':'pass','syntax_files':len(files),'modified_source_files':changed}
    rerun=ROOT/'results/reproduced/final_validation_results.json'
    if rerun.exists():
        assert json.loads(rerun.read_text())==data,'Saved/reproduced JSON differs'
        checks['full_reproduced_json']='identical'
        checks['reproduced_sha256']=hashlib.sha256(rerun.read_bytes()).hexdigest()
    else:checks['full_reproduced_json']='not rerun in this checkout'
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    return checks

if __name__=='__main__':main()
