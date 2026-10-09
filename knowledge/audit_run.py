"""Audit saved exam payloads, label balance, phase ordering and grading."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from run_exam import INSTRUCTIONS,NOTATION,MODEL

ROOT=Path(__file__).resolve().parent

def main():
    directory=Path(json.loads((ROOT/'latest_run.json').read_text(encoding='utf-8'))['directory'])
    results=json.loads((directory/'results.json').read_text(encoding='utf-8'))
    questions=json.loads((ROOT/'questions.json').read_text(encoding='utf-8'))
    key=json.loads((ROOT/'answer_key.json').read_text(encoding='utf-8'))
    plan=results['plan']
    assert hashlib.sha256((ROOT/'questions.json').read_bytes()).hexdigest()==plan['questions_sha256']
    assert hashlib.sha256((ROOT/'answer_key.json').read_bytes()).hexdigest()==plan['answer_key_sha256']
    assert len(results['results'])==len(questions)==50
    assert len({r['id'] for r in results['results']})==50
    probabilities_checked=0
    starts={}
    for q,row in zip(questions,results['results']):
        assert q['id']==row['id']
        saved=json.loads((directory/f"{q['id']}.json").read_text(encoding='utf-8'))
        assert saved['status']==row['status']=='answered'
        assert saved['choice']==row['choice']
        assert row['correct']==(row['choice']==key[q['id']]['correct_choice'])
        assert row['correct_text']==q['options'][key[q['id']]['correct_choice']]
        assert len(saved['attempts'])==1
        attempt=saved['attempts'][0];payload=attempt['request'];response=attempt['response']
        assert payload['model']==MODEL
        assert payload['state']=={'question':q['question_en'],'notation':NOTATION}
        assert payload['questions']=={'action':{'type':'choice','instructions':INSTRUCTIONS,'criteria':q['options']}}
        assert response['choice']==row['choice'] and response['probabilities']==row['probabilities']
        assert 0<=response['confidence']<=1
        assert set(response['probabilities'])==set(q['options'])
        assert all(math.isfinite(v) and 0<=v<=1 for v in response['probabilities'].values())
        # API can round probabilities; allow up to one percent total rounding error.
        assert abs(sum(response['probabilities'].values())-1)<.01
        probabilities_checked+=1
        starts[q['id']]=(datetime.fromisoformat(attempt['started_at']),response['latency_seconds'])
    easy_last_end=max(starts[f'Q{i:02d}'][0].timestamp()+starts[f'Q{i:02d}'][1] for i in range(1,11))
    hard_first_start=min(starts[f'Q{i:02d}'][0].timestamp() for i in range(11,51))
    assert hard_first_start>=easy_last_end
    balance=Counter(key[q['id']]['correct_choice'] for q in questions[10:])
    assert balance=={'A':10,'B':10,'C':10,'D':10}
    assert sum(row['correct'] for row in results['results'])==results['overall']['correct']
    report={'passed':True,'payloads_checked':50,'probability_distributions_checked':probabilities_checked,
            'correct_answer_in_input':False,'worked_solution_in_input':False,'tools_supplied_to_jev':False,
            'phase_order_verified':True,'hard_correct_label_balance':balance,'valid_responses':50,'api_attempts':50,
            'grading_checked':True,'exam_and_key_hashes_verified':True}
    (directory/'run_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__': main()
