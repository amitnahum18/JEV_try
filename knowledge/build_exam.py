"""Prepare the frozen public questions and separate private answer key."""
from pathlib import Path
import json
import random
from items_easy import add_items as easy
from items_math_probability import add_items as mathematical
from items_physics_logic import add_items as physical

ROOT = Path(__file__).resolve().parent
QUESTIONS = []

def add(identifier, category, title, english, hebrew, correct, distractors, solution, sources=()):
    assert len(distractors)==3 and len(set([correct,*distractors]))==4
    QUESTIONS.append(dict(id=identifier,difficulty='simple_medium' if len(QUESTIONS)<10 else 'very_hard',category=category,title_he=title,question_en=english,question_he=hebrew,correct_text=correct,distractors=distractors,solution_he=solution,sources=list(sources)))

def main():
    easy(add); mathematical(add); physical(add)
    assert len(QUESTIONS)==50
    assert [q['id'] for q in QUESTIONS]==[f'Q{i:02d}' for i in range(1,51)]
    rng=random.Random(27092026)
    positions=list(range(4))*2+[0,1]; rng.shuffle(positions)
    hard=list(range(4))*10; rng.shuffle(hard); positions+=hard
    public,key=[],{}
    for q,index in zip(QUESTIONS,positions):
        distractors=q['distractors'].copy(); rng.shuffle(distractors)
        answers=distractors[:index]+[q['correct_text']]+distractors[index:]
        options=dict(zip('ABCD',answers)); options['unsure']='I do not know / cannot determine the answer.'
        public.append({k:q[k] for k in ('id','difficulty','category','title_he','question_en','question_he')}|{'options':options})
        key[q['id']]={'correct_choice':'ABCD'[index],'correct_text':q['correct_text'],'solution_he':q['solution_he'],'sources':q['sources']}
    (ROOT/'questions.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'answer_key.json').write_text(json.dumps(key,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# מבחן JEV — 50 שאלות','','10 שאלות פשוטות–בינוניות ולאחריהן 40 ברמת אתגר גבוהה מאוד. רמת הקושי היא הערכת מחבר המבחן, לא תוצאה של ניסוי בבני אדם. פתרונות נשמרים בנפרד. השאלות נשלחות ל־JEV באנגלית; להלן גם תרגום עברי.']
    solutions=['# תשובות ייחוס ופתרונות','','קובץ זה אינו נשלח למודל. השאלות חוברו עבור הניסוי. מקורות מאמתים עקרונות פיזיקליים ועובדות גאוגרפיות; אינם נשלחים ל־JEV.']
    for q in public:
        lines+=['',f"## {q['id']} — {q['title_he']}",'',q['question_he'],'',f"**English sent to JEV:** {q['question_en']}",'']
        lines += [f'\u002d **{label}**: {answer}' for label,answer in q['options'].items()]
        a=key[q['id']]; solutions+=['',f"## {q['id']} — {a['correct_choice']}: {a['correct_text']}",'',a['solution_he']]
        if a['sources']: solutions+=['','מקורות: '+', '.join(f'[מקור {i+1}]({url})' for i,url in enumerate(a['sources']))]
    (ROOT/'questions_he.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'answer_key_he.md').write_text('\n'.join(solutions)+'\n',encoding='utf-8')
    print(json.dumps({'questions':50,'simple_medium':10,'very_hard':40,'output':str(ROOT)}))

if __name__=='__main__': main()
