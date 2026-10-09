from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.pgn


BASE = Path(__file__).resolve().parents[2]
VIDEO_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE / "recordings" / "jev_systems_competition_20260926_143733"
OLD_RUN_DIR = BASE / "recordings" / "ladder_20260926_1345"

competition = json.loads((RUN_DIR / "competition.json").read_text(encoding="utf-8"))
scenes: list[dict[str, object]] = []
cursor = 0.0


def add_scene(kind: str, duration: float, **data: object) -> None:
    global cursor
    scenes.append({"kind": kind, "start": round(cursor, 3), "duration": duration, **data})
    cursor += duration


add_scene("intro", 5.5)

for system in ("single", "team"):
    track = competition["systems"][system]
    meta = track["games"][0]
    game_dir = RUN_DIR / system
    decision_rows = json.loads((game_dir / meta["decisions"]).read_text(encoding="utf-8"))
    by_ply = {row["ply"]: row for row in decision_rows}
    with (game_dir / meta["pgn"]).open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise RuntimeError(f"Could not parse {meta['pgn']}")
    game_moves = list(game.mainline_moves())

    label = "JEV יחיד" if system == "single" else "צוות JEV"
    add_scene("match_title", 3.2, system=system, system_label=label, opponent=meta["opponent"])
    board = game.board()
    for ply, move in enumerate(game_moves, 1):
        san = board.san(move)
        board.push(move)
        row = by_ply.get(ply, {})
        is_jev = row.get("player") == "JEV"
        add_scene(
            "move",
            0.24,
            system=system,
            system_label=label,
            opponent=meta["opponent"],
            fen=board.fen(),
            san=san,
            ply=ply,
            mover=label if is_jev else meta["opponent"],
            confidence=row.get("confidence"),
            agreement=row.get("council_agreement"),
            tactical_review=row.get("tactical_review"),
            check=board.is_check(),
            move_number=board.fullmove_number - (1 if board.turn else 0),
        )
    add_scene(
        "game_result",
        4.2,
        system_label=label,
        opponent=meta["opponent"],
        result=meta["result"],
        status=meta["status"],
        plies=len(game_moves),
    )

old_summary = json.loads((OLD_RUN_DIR / "tournament.json").read_text(encoding="utf-8"))
old_results = [
    {"opponent": item["opponent"], "result": item["result"], "status": item["status"]}
    for item in old_summary["games"]
]
add_scene("old_baseline", 5.0, results=old_results)
add_scene("scoreboard", 9.0, comparison=competition["comparison"], winner=competition["winner"], old_results=old_results)
duration = round(cursor, 3)
scene_json = json.dumps(scenes, ensure_ascii=False, separators=(",", ":"))

html = r'''<!doctype html>
<html lang="he" data-resolution="landscape">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1920,height=1080">
<script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
<style>
*{box-sizing:border-box}html,body{margin:0;width:1920px;height:1080px;overflow:hidden;background:#101714}
canvas{display:block;width:1920px;height:1080px}
</style>
</head>
<body>
<canvas id="stage" data-composition-id="main" class="clip" data-start="0" data-duration="__DURATION__" data-width="1920" data-height="1080" width="1920" height="1080"></canvas>
<script>
const scenes=__SCENES__;
const DURATION=__DURATION__;
const canvas=document.getElementById('stage'),ctx=canvas.getContext('2d');
const glyph={K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
function backdrop(){ctx.fillStyle='#101714';ctx.fillRect(0,0,1920,1080);const g=ctx.createRadialGradient(960,450,80,960,540,1000);g.addColorStop(0,'#1c2a23');g.addColorStop(.65,'#121a16');g.addColorStop(1,'#0b100e');ctx.fillStyle=g;ctx.fillRect(0,0,1920,1080)}
function text(s,x,y,size,color='#f2efdf',align='center',weight='600',direction='rtl'){ctx.fillStyle=color;ctx.font=`${weight} ${size}px Arial, "Segoe UI", sans-serif`;ctx.textAlign=align;ctx.textBaseline='middle';ctx.direction=direction;ctx.fillText(String(s),x,y,1580)}
function eyebrow(s,y=285){text(s,960,y,25,'#a4bf75','center','700')}
function board(fen){const [placement]=fen.split(' '),ranks=placement.split('/'),left=248,top=176,size=720,cell=90;for(let r=0;r<8;r++)for(let f=0;f<8;f++){ctx.fillStyle=(r+f)%2===1?'#d8d5c4':'#70876a';ctx.fillRect(left+f*cell,top+r*cell,cell,cell)}for(let r=0;r<8;r++){let f=0;for(const ch of ranks[r]){if(/\d/.test(ch)){f+=Number(ch);continue}const px=left+f*cell,py=top+r*cell;ctx.fillStyle=ch===ch.toUpperCase()?'#fffdf1':'#202921';ctx.font='76px "Segoe UI Symbol","Arial Unicode MS",serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.direction='ltr';ctx.shadowColor=ch===ch.toUpperCase()?'#273127':'#cad1b9';ctx.shadowBlur=2;ctx.fillText(glyph[ch],px+cell/2,py+cell/2+1);ctx.shadowBlur=0;f++}}ctx.strokeStyle='#425444';ctx.lineWidth=3;ctx.strokeRect(left,top,size,size);ctx.fillStyle='#a8b49d';ctx.font='18px Arial';ctx.textAlign='center';ctx.textBaseline='middle';for(let i=0;i<8;i++){ctx.fillText(String(8-i),left-25,top+i*cell+cell/2);ctx.fillText('abcdefgh'[i],left+i*cell+cell/2,top+size+26)}}
function badge(label,value,y,color='#bad08b'){text(label,1735,y,23,'#9eac9f','right','500');text(value,1735,y+43,34,color,'right','700')}
function drawTableRow(y,system,opponent,result,moves,cost){ctx.fillStyle='#1a2820';ctx.fillRect(300,y-39,1320,78);ctx.strokeStyle='#3a5140';ctx.lineWidth=1;ctx.strokeRect(300,y-39,1320,78);text(system,1575,y,29,'#f2efdf','right','700');text(opponent,1220,y,26,'#c8d0c4','right','500');text(result,850,y,28,result==='הפסד'?'#e9a37f':'#bad08b','right','700');text(`${moves} מהלכים`,650,y,23,'#c8d0c4','right','500');text(`$${Number(cost).toFixed(4)}`,445,y,22,'#a4bf75','right','500','ltr')}
function draw(time){backdrop();const s=scenes.find(v=>time>=v.start&&time<v.start+v.duration)||scenes[scenes.length-1];
if(s.kind==='intro'){eyebrow('אתגר השחמט של JEV',355);text('שחקן אחד. צוות שלם.',960,475,82,'#f2efdf');text('מי יעמוד מול מודלים חזקים יותר?',960,590,48,'#bad08b');text('שתי מערכות, אותו יריב, אותם כללים',960,700,30,'#c6cfbf','center','500')}
else if(s.kind==='match_title'){eyebrow('שלב 1 · Qwen3.7 Plus',360);text(s.system_label,960,500,76);text('מול אותו מודל · JEV בלבן',960,610,32,'#c6cfbf','center','500')}
else if(s.kind==='move'){board(s.fen);text(`${s.system_label}  ·  ${s.opponent}`,1740,285,25,'#a4bf75','right','700');badge('תור',s.mover,405,'#f2efdf');text(s.san,1740,565,77,'#f4f1e5','right','700','ltr');text(`${s.move_number}${s.mover==='JEV יחיד'||s.mover==='צוות JEV'?'.':'...'}  ·  מהלך ${s.ply}`,1740,655,25,'#a5b397','right','500');if(s.check)badge('מצב','שח',740,'#e0ba70');if(s.confidence!==null&&s.confidence!==undefined)badge('ביטחון JEV',`${Math.round(s.confidence*100)}%`,790);if(s.agreement!==null&&s.agreement!==undefined)badge('הסכמת הצוות',`${Math.round(s.agreement*100)}%`,895);if(s.tactical_review)text('בדיקת טקטיקה הופעלה',1740,975,20,'#a4bf75','right','500')}
else if(s.kind==='game_result'){eyebrow(`${s.system_label} מול ${s.opponent}`,365);text(s.result==='1-0'?'ניצחון ל־JEV':s.result==='1/2-1/2'?'תיקו':'הפסד ל־JEV',960,510,70,s.result==='0-1'?'#e9a37f':'#bad08b');text(s.result==='0-1'?'0 — 1':s.result==='1-0'?'1 — 0':'½ — ½',960,655,96,'#f2efdf','center','700','ltr');text(`${s.plies} חצאי־מהלכים`,960,770,29,'#c6cfbf','center','500')}
else if(s.kind==='old_baseline'){eyebrow('בדיקה קודמת מול יריבים שאינם מודלי שפה',300);text('אקראי',1500,485,32);text('ניצחון ל־JEV · 1–0',840,485,30,'#bad08b');text('צייד־חומר',1500,610,32);text('תיקו · ½–½',840,610,30,'#e0ba70');text('Stockfish הוגדר בסולם, אך לא שוחק: התיקו עצר את ההתקדמות.',960,795,27,'#c6cfbf','center','500')}
else{eyebrow('טבלת תוצאות',215);text('תחרות ראש בראש · שלב ראשון',960,290,35,'#c6cfbf','center','500');text('מערכת',1575,385,22,'#a4bf75','right','700');text('יריב',1220,385,22,'#a4bf75','right','700');text('תוצאה',850,385,22,'#a4bf75','right','700');text('הישרדות',650,385,22,'#a4bf75','right','700');text('עלות JEV',445,385,22,'#a4bf75','right','700');const a=s.comparison[0],b=s.comparison[1];drawTableRow(470,'JEV יחיד','Qwen3.7 Plus','הפסד',a.jev_moves,a.jev_cost_usd);drawTableRow(570,'צוות JEV','Qwen3.7 Plus','הפסד',b.jev_moves,b.jev_cost_usd);text('תוצאה: תיקו בין המערכות — שתיהן נעצרו בשלב הראשון.',960,700,34,'#e0ba70');text('בבדיקה הקודמת JEV יחיד ניצח יריב אקראי וסיים בתיקו מול צייד־חומר.',960,795,26,'#c6cfbf','center','500');text('כלל הסולם: ממשיכים רק אחרי ניצחון.',960,880,25,'#a4bf75','center','600')}
}
window.__timelines=window.__timelines||{};const tl=gsap.timeline({paused:true,onUpdate:()=>draw(tl.time())});tl.to({}, {duration:DURATION},0);window.__timelines.main=tl;draw(0);tl.seek(0);
</script>
</body>
</html>
'''
html = html.replace("__DURATION__", str(duration)).replace("__SCENES__", scene_json)
(VIDEO_DIR / "index.html").write_text(html, encoding="utf-8")
print(f"Generated {len(scenes)} scenes, {sum(scene['kind'] == 'move' for scene in scenes)} plies, {duration:.1f}s")
