"""Run the frozen 10+40 exam through the existing JEV typed-choice API."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'car'))
from jev_client import MODEL,ENDPOINT,JevError,choose_action

INSTRUCTIONS=(
    'Solve the question in the supplied state and select its correct answer from the four candidate answers A,B,C,D. '
    'Read all assumptions and notation carefully. Use your own knowledge and reasoning; no external tools, '
    'calculator, search, code execution, worked solution, or feedback are supplied. The option order is arbitrary. '
    'If you cannot determine the answer, select unsure. Return the typed choice. '
    'The candidate descriptions are literal proposed answers, not additional premises.'
)
NOTATION='pi is the circle constant; i is the imaginary unit; ln is natural logarithm; sqrt is the nonnegative square root; c is light speed; hbar is reduced Planck constant. * denotes multiplication and ^ a power.'

def write(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)

def perform(q,output):
    state={'question':q['question_en'],'notation':NOTATION}
    assert set(state)=={'question','notation'}
    assert len(q['options'])==5 and set(q['options'])=={'A','B','C','D','unsure'}
    payload={'model':MODEL,'state':state,'questions':{'action':{'type':'choice','instructions':INSTRUCTIONS,'criteria':q['options']}}}
    attempts=[]
    for attempt in range(1,3):
        started=time.perf_counter()
        entry={'attempt':attempt,'started_at':datetime.now().astimezone().isoformat(),'request':payload}
        try:
            decision=choose_action(state=state,actions=q['options'],instructions=INSTRUCTIONS,timeout_seconds=40.)
            entry['response']=asdict(decision)
            attempts.append(entry)
            result={'id':q['id'],'status':'answered','choice':decision.choice,'answer_text':q['options'][decision.choice],
                    'confidence':decision.confidence,'probabilities':decision.probabilities,'cost_usd':decision.cost_usd,
                    'latency_seconds':decision.latency_seconds,'input_tokens':decision.input_tokens,'attempts':attempts}
            write(output/f"{q['id']}.json",result)
            return result
        except JevError as exc:
            entry.update(error=str(exc),elapsed_seconds=time.perf_counter()-started)
            attempts.append(entry)
            write(output/f"{q['id']}.json",{'id':q['id'],'status':'api_error','attempts':attempts})
    return {'id':q['id'],'status':'api_error','attempts':attempts}

def report(output,questions,key,results,elapsed,plan):
    ordered=sorted(results,key=lambda row:row['id'])
    by_id={q['id']:q for q in questions}
    for row in ordered:
        q=by_id[row['id']];gold=key[row['id']]
        row.update(difficulty=q['difficulty'],category=q['category'],title_he=q['title_he'],
                   correct_choice=gold['correct_choice'],correct_text=gold['correct_text'],
                   correct=row.get('choice')==gold['correct_choice'] if row['status']=='answered' else None)
    def summarize(group):
        valid=[r for r in group if r['status']=='answered']
        return {'questions':len(group),'answered':len(valid),'correct':sum(r['correct'] for r in valid),
                'incorrect':sum(not r['correct'] for r in valid),'unsure':sum(r['choice']=='unsure' for r in valid),
                'api_errors':len(group)-len(valid),'accuracy':sum(r['correct'] for r in valid)/len(valid) if valid else None}
    groups={name:summarize([r for r in ordered if r['difficulty']==name]) for name in ('simple_medium','very_hard')}
    categories={name:summarize([r for r in ordered if r['category']==name]) for name in sorted({q['category'] for q in questions})}
    valid=[r for r in ordered if r['status']=='answered']
    incorrect=[r for r in valid if not r['correct']]
    costs=[r['cost_usd'] for r in valid if r['cost_usd'] is not None]
    confidence={name:statistics.mean(r['confidence'] for r in valid if bool(r['correct'])==correct)
                for name,correct in [('correct_mean',True),('incorrect_mean',False)]
                if any(bool(r['correct'])==correct for r in valid)}
    summary={'model':MODEL,'overall':summarize(ordered),'difficulty':groups,'category':categories,
             'confidence':confidence,'wrong_with_reported_confidence_at_least_0_8':sum(r['confidence']>=.8 for r in incorrect),
             'known_cost_usd':sum(costs),'cost_complete':len(costs)==len(valid) and all(len(r['attempts'])==1 for r in ordered),
             'api_attempts':sum(len(r['attempts']) for r in ordered),'wall_seconds':elapsed,
             'mean_response_seconds':statistics.mean(r['latency_seconds'] for r in valid) if valid else None,
             'first_run_per_question':True,'plan':plan,'results':ordered,
             'limitations':['Author-assigned difficulty, no human participant study.','Multiple-choice selection, not proof generation.','Small mixed-topic sample, not IQ or a standardized benchmark.','Some classical puzzles may be familiar from training.','No tools or answer-key access for JEV.','Question language English; Hebrew translation for the user.']}
    write(output/'results.json',summary)
    lines=['# תוצאות מבחן JEV — 10+40','','מודל: `'+MODEL+'`. תשובה אחת לכל שאלה, בלי כלים, פתרונות או משוב. ניסיונות חוזרים הותרו רק בכשל API ונשמרו. שאלות באנגלית, עם תיעוד עברי.','','| קבוצה | שאלות | תשובות תקינות | נכונות | דיוק | לא יודע | כשל API |','|---|---:|---:|---:|---:|---:|---:|']
    for name,label in [('simple_medium','פשוטות–בינוניות'),('very_hard','קשות מאוד')]:
        g=groups[name];accuracy=f"{g['accuracy']:.1%}" if g['accuracy'] is not None else '—'
        lines.append(f"| {label} | {g['questions']} | {g['answered']} | {g['correct']} | {accuracy} | {g['unsure']} | {g['api_errors']} |")
    lines+=['','## לפי תחום','','| תחום | נכונות / תשובות תקינות | דיוק |','|---|---:|---:|']
    names={'mathematics':'מתמטיקה','physics':'פיזיקה','probability':'הסתברות','logic':'לוגיקה וחידות','geography':'גאוגרפיה'}
    for name,g in categories.items():
        accuracy=f"{g['accuracy']:.1%}" if g['accuracy'] is not None else '—'
        lines.append(f"| {names[name]} | {g['correct']}/{g['answered']} | {accuracy} |")
    lines+=['','## כל התשובות','','| שאלה | נושא | תשובת JEV | תשובת ייחוס | תוצאה | ביטחון מדווח |','|---|---|---|---|---|---:|']
    for row in ordered:
        flag='נכון' if row['correct'] else 'שגוי' if row['status']=='answered' else 'כשל API'
        conf=f"{row['confidence']:.1%}" if 'confidence' in row else '—'
        lines.append(f"| {row['id']} | {row['title_he']} | {row.get('answer_text','—')} | {row['correct_text']} | {flag} | {conf} |")
    lines+=['','## טעויות ופתרונות','','הפתרונות הבאים הם פתרונות הייחוס המקומיים, לא הסברים שחיבר JEV.']
    for row in incorrect:
        q=by_id[row['id']];gold=key[row['id']]
        lines+=['',f"### {q['id']} — {q['title_he']}",'',q['question_he'],'',f"**JEV:** {row['answer_text']}. **הנכון:** {gold['correct_text']}.",'',gold['solution_he']]
    lines+=['','## היקף ותנאים','',f"עלות ידועה: ${summary['known_cost_usd']:.6f}; ניסיונות API: {summary['api_attempts']}; זמן קיר: {elapsed:.2f} שניות.",'','הביטחון המדווח אינו הסתברות נכונות מכוילת. רמת הקושי היא הערכת המחבר; לא נבדקו בני אדם. זהו מבחן בחירה קטן, ללא דרישה להציג הוכחה, וחלק מהחידות קלאסיות. אין להסיק ממנו ציון IQ או יכולת כללית בכל התחומים. ניחוש אחיד מארבע התשובות המהותיות נותן 25%; כולל אפשרות ״לא יודע״, 20%.','','[כל 50 השאלות](../../questions_he.md) · [תשובות ופתרונות הייחוס](../../answer_key_he.md) · [אימות תשובות הייחוס](../../answer_key_verification.json)']
    (output/'results_he.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary

def main():
    questions=json.loads((ROOT/'questions.json').read_text(encoding='utf-8'))
    key=json.loads((ROOT/'answer_key.json').read_text(encoding='utf-8'))
    verified=json.loads((ROOT/'answer_key_verification.json').read_text(encoding='utf-8'))
    assert verified['all_passed'] and verified['questions_checked']==50
    assert len(questions)==50
    for q in questions:assert q['options'][key[q['id']]['correct_choice']]==key[q['id']]['correct_text']
    output=ROOT/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    plan={'model':MODEL,'endpoint':ENDPOINT,'questions_sha256':hashlib.sha256((ROOT/'questions.json').read_bytes()).hexdigest(),
          'answer_key_sha256':hashlib.sha256((ROOT/'answer_key.json').read_bytes()).hexdigest(),'workers':4,
          'format':'Four candidate answers plus unsure. Opaque labels A-D, balanced correct positions.',
          'stages':['First ten simple/medium questions; wait for completion.','Then forty very hard questions.'],
          'model_input_fields':['question','notation'],'ground_truth_sent_to_model':False,
          'feedback_sent_to_model':False,'retry_policy':'At most one retry only after an API error, never after a wrong answer.'}
    write(output/'plan.json',plan)
    started=time.perf_counter();results=[]
    for phase,subset in [('simple_medium',questions[:10]),('very_hard',questions[10:])]:
        print('START '+phase,flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(perform,q,output) for q in subset]):
                row=future.result();results.append(row)
                print(json.dumps({'id':row['id'],'status':row['status'],'choice':row.get('choice')}),flush=True)
        print('DONE '+phase,flush=True)
    summary=report(output,questions,key,results,time.perf_counter()-started,plan)
    write(ROOT/'latest_run.json',{'directory':str(output),'report':str(output/'results_he.md')})
    print(json.dumps({'overall':summary['overall'],'difficulty':summary['difficulty'],'report':str(output/'results_he.md'),'cost_usd':summary['known_cost_usd']}),flush=True)

if __name__=='__main__': main()
