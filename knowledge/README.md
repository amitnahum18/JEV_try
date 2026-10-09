# JEV knowledge and reasoning exam

50 questions: 10 simple/medium, then 40 demanding mathematics, probability,
physics and logic problems. This is a custom exploratory test; difficulty is
author-assigned and has not been measured with human participants.

JEV receives English questions and four literal candidate answers plus `unsure`.
The report and full questions also have Hebrew text. There is no calculator,
search, code execution, answer key, worked solution or prior-answer feedback
available to JEV. Local reference computations are for the evaluator only.

```powershell
.venv/Scripts/python.exe knowledge/build_exam.py
.venv/Scripts/python.exe knowledge/verify_key.py
.venv/Scripts/python.exe knowledge/run_exam.py
```

The first two commands make no model calls. The final command uses the existing
OpenRouter credentials and creates a fresh run directory. It completes the ten
initial questions before starting the forty hard ones. Four independent questions
can run concurrently; each answer comes from one JEV call. Incorrect answers are
never retried. An API error can be retried once, with both attempts preserved.

`questions.json` and `questions_he.md` contain questions/options without answers.
`answer_key.json` and `answer_key_he.md` contain reference answers and derivations.
`answer_key_verification.json` records the fifty local checks. Each run preserves
exact API payloads without authentication, returned choice/confidence/probabilities,
timing and known cost. `latest_run.json` points to the latest results.

All hard questions have a solvable, unique candidate answer under their stated
assumptions. `unsure` records abstention; it is not the correct answer to any item.
Choosing uniformly among the four substantive candidates yields 25% expected
accuracy (20% if uniform over all five choices). Correct labels are balanced in
the hard block. A single selection does not demonstrate a generated proof.
