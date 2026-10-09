"""Render the actual eight recorded games, then their four-row result table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.pgn

VIDEO_DIR = Path(__file__).resolve().parent
CHESS_DIR = VIDEO_DIR.parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, default=CHESS_DIR / "recordings" / "non_llm_four_20260926")
args = parser.parse_args()
summary = json.loads((args.run_dir / "benchmark.json").read_text(encoding="utf-8"))
if not summary["completed"]:
    raise RuntimeError("The eight-game comparison is not complete")

scenes = []
cursor = 0.0


def add(kind: str, duration: float, **data) -> None:
    global cursor
    scenes.append({"kind": kind, "start": round(cursor, 4), "duration": duration, **data})
    cursor += duration


labels = {"single": "JEV יחיד", "team": "צוות JEV"}
outcome = {"win": "ניצחון", "draw": "תיקו", "loss": "הפסד"}
add("intro", 5)
table = []
for index, level_id in enumerate(("maia_1100", "maia_1500", "maia_1900", "stockfish"), 1):
    matches = [match for match in summary["games"] if match["level"]["id"] == level_id]
    name = matches[0]["opponent"].replace(" (maximum skill)", "")
    add("level", 2.4, level=index, name=name,
        description="מודל עצבי המדמה מהלכים אנושיים" if level_id != "stockfish" else "מנוע שחמט · מיומנות מרבית · תקציב חישוב קבוע")
    row = {"name": name}
    for system in labels:
        match = next(match for match in matches if match["system"] == system)
        with (args.run_dir / match["pgn"]).open(encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)
        if game is None or game.errors:
            raise RuntimeError(f"Invalid recorded PGN: {match['pgn']}")
        moves = list(game.mainline_moves())
        if len(moves) != match["plies"] or game.end().board().result(claim_draw=True) != match["result"]:
            raise RuntimeError(f"Recorded result mismatch: {match['pgn']}")
        records = json.loads((args.run_dir / system / f"{level_id}.json").read_text(encoding="utf-8"))["moves"]
        board = game.board()
        add("match", 1.8, label=labels[system], name=name, level=index)
        for ply, move in enumerate(moves, 1):
            record = records[ply - 1]
            if record["uci"] != move.uci():
                raise RuntimeError("PGN and decision log disagree")
            san = board.san(move)
            white = board.turn
            board.push(move)
            add("move", .26, label=labels[system], name=name, level=index,
                fen=board.fen(), san=san, ply=ply, total=len(moves), white=white,
                from_square=move.from_square, to_square=move.to_square,
                confidence=record.get("confidence"), agreement=record.get("role_agreement"),
                check=board.is_check(), mate=board.is_checkmate())
        add("result", 2.8, label=labels[system], name=name, fen=board.fen(),
            result=match["result"], outcome=outcome[match["status"]], plies=len(moves))
        row[system] = {"outcome": outcome[match["status"]], "plies": len(moves)}
    table.append(row)
add("scoreboard", 13, rows=table, scores=summary["scores"])
duration = round(cursor, 3)

template = r'''<!doctype html>
<html lang="he" data-resolution="landscape">
<head><meta charset="utf-8"><meta name="viewport" content="width=1920,height=1080">
<script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
<style>*{box-sizing:border-box}html,body{margin:0;width:1920px;height:1080px;overflow:hidden;background:#101714}canvas{display:block;width:1920px;height:1080px}</style></head>
<body><canvas id="stage" class="clip" data-composition-id="main" data-start="0" data-duration="__DURATION__" data-width="1920" data-height="1080" width="1920" height="1080"></canvas>
<script>
const scenes=__SCENES__,DURATION=__DURATION__;
const ctx=document.getElementById('stage').getContext('2d');
const glyph={K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
function txt(s,x,y,size=32,color='#f2efdf',align='center',direction='rtl',weight=600){ctx.fillStyle=color;ctx.font=`${weight} ${size}px Arial,"Segoe UI",sans-serif`;ctx.textAlign=align;ctx.textBaseline='middle';ctx.direction=direction;ctx.fillText(s,x,y)}
function background(){ctx.fillStyle='#101714';ctx.fillRect(0,0,1920,1080);ctx.fillStyle='#19251e';ctx.fillRect(0,0,1920,100);txt('JEV CHESS LAB',125,51,24,'#bad08b','left','ltr');txt('ארבעה יריבי שחמט · שתי מערכות JEV',1795,51,24,'#d4dccd','right')}
function board(fen,from=-1,to=-1){const x=140,y=220,size=88,rows=fen.split(' ')[0].split('/');for(let r=0;r<8;r++){let col=0;for(const symbol of rows[r]){if(/[1-8]/.test(symbol)){col+=Number(symbol);continue}col++}for(let c=0;c<8;c++){const sq=(7-r)*8+c;ctx.fillStyle=sq===from||sq===to?'#c7b869':(r+c)%2?'#77916f':'#e8dfca';ctx.fillRect(x+c*size,y+r*size,size,size)}}for(let r=0;r<8;r++){let c=0;for(const symbol of rows[r]){if(/[1-8]/.test(symbol)){c+=Number(symbol);continue}const px=x+(c+.5)*size,py=y+(r+.53)*size;ctx.direction='ltr';ctx.textAlign='center';ctx.textBaseline='middle';ctx.font='70px "Segoe UI Symbol","DejaVu Sans",serif';ctx.lineWidth=2;ctx.strokeStyle='#18221c';ctx.strokeText(glyph[symbol],px,py);ctx.fillStyle=symbol===symbol.toUpperCase()?'#fffdf1':'#18221c';ctx.fillText(glyph[symbol],px,py);c++}}for(let c=0;c<8;c++)txt('abcdefgh'[c],x+(c+.5)*size,y+size*8+26,23,'#a7b49e','center','ltr');for(let r=0;r<8;r++)txt(String(8-r),x-26,y+(r+.5)*size,23,'#a7b49e','center','ltr');txt('לבן: JEV',x,970,25,'#c6cfbf','left');txt('שחור: היריב',x+size*8,970,25,'#c6cfbf','right')}
function resultColor(value){return value==='ניצחון'?'#bad08b':value==='תיקו'?'#e0ba70':'#e9a37f'}
function draw(time){background();const s=scenes.find(scene=>time>=scene.start&&time<scene.start+scene.duration)||scenes[scenes.length-1];
if(s.kind==='intro'){txt('ארבעה יריבים שאינם מודלי שפה',960,375,67);txt('JEV יחיד מול צוות JEV',960,510,58,'#bad08b');txt('Maia 1100  ·  Maia 1500  ·  Maia 1900  ·  Stockfish',960,650,36,'#d4dccd','center','ltr');txt('משחקים מוקלטים בפועל · כל ארבע הרמות נבדקות',960,775,30,'#c6cfbf')}
else if(s.kind==='level'){txt(`יריב ${s.level} מתוך 4`,960,360,32,'#bad08b');txt(s.name,960,495,82,'#f2efdf','center','ltr');txt(s.description,960,640,35,'#c6cfbf');if(s.level<4)txt('מספר Maia הוא רמת האימון, לא דירוג משחק מובטח',960,765,27,'#a7b49e')}
else if(s.kind==='match'){txt(`יריב ${s.level} · ${s.name}`,960,390,38,'#bad08b');txt(s.label,960,525,85);txt('JEV משחק בלבן · ללא ייעוץ ממנוע שחמט',960,690,33,'#c6cfbf')}
else if(s.kind==='move'){board(s.fen,s.from_square,s.to_square);txt(`יריב ${s.level} / 4 · ${s.name}`,1720,245,31,'#bad08b','right');txt(s.label,1720,345,49,'#f2efdf','right');txt(s.white?'המהלך של JEV':'המהלך של היריב',1720,435,27,'#c6cfbf','right');txt(s.san,1720,555,89,'#f2efdf','right','ltr');txt(`חצי־מהלך ${s.ply} מתוך ${s.total}`,1720,675,29,'#a7b49e','right');if(s.confidence!==undefined)txt(`ביטחון JEV: ${Math.round(s.confidence*100)}%`,1720,765,29,'#bad08b','right');if(s.agreement!==undefined&&s.agreement!==null)txt(`הסכמת הצוות: ${Math.round(s.agreement*100)}%`,1720,835,28,'#bad08b','right');if(s.check)txt(s.mate?'מט':'שח',1720,935,52,'#e9a37f','right')}
else if(s.kind==='result'){board(s.fen);txt(`${s.label} · ${s.name}`,1720,300,34,'#bad08b','right');txt(`${s.outcome} ל־JEV`,1720,505,65,resultColor(s.outcome),'right');txt(s.result,1720,650,83,'#f2efdf','right','ltr');txt(`${s.plies} חצאי־מהלכים · משחק מלא`,1720,795,28,'#c6cfbf','right')}
else{txt('טבלת התוצאות',960,190,63);txt('יריב',1580,295,29,'#bad08b');txt('JEV יחיד',1030,295,29,'#bad08b');txt('צוות JEV',495,295,29,'#bad08b');s.rows.forEach((row,i)=>{const y=400+i*125;ctx.fillStyle=i%2?'#17231c':'#1d2b22';ctx.fillRect(265,y-47,1390,99);txt(row.name,1580,y,35,'#f2efdf','right','ltr');for(const [key,x] of [['single',1030],['team',495]]){txt(row[key].outcome,x,y-12,37,resultColor(row[key].outcome));txt(`${row[key].plies} חצאי־מהלכים`,x,y+28,22,'#c6cfbf')}});const a=s.scores.single,b=s.scores.team;txt(`יחיד: ${a.win} ניצחונות · ${a.draw} תיקו · ${a.loss} הפסדים    |    צוות: ${b.win} ניצחונות · ${b.draw} תיקו · ${b.loss} הפסדים`,960,900,30,'#e0ba70');txt('משחק אחד לכל מערכת מול כל יריב; זו אינה מדידת Elo.',960,969,25,'#c6cfbf');txt('Maia: ללא חיפוש · Stockfish: מיומנות 20, עד 200,000 צמתים למהלך',960,1020,23,'#a7b49e')}
}
window.__timelines=window.__timelines||{};
const tl=gsap.timeline({paused:true,onUpdate:()=>draw(tl.time())});tl.to({}, {duration:DURATION},0);window.__timelines.main=tl;draw(0);tl.seek(0);
</script></body></html>'''
html = template.replace("__SCENES__", json.dumps(scenes, ensure_ascii=False, separators=(",", ":"))).replace("__DURATION__", str(duration))
(VIDEO_DIR / "index.html").write_text(html, encoding="utf-8")
(VIDEO_DIR / "non_llm_timeline.json").write_text(json.dumps({"duration": duration, "scenes": scenes}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"duration_seconds": duration, "games": 8, "plies": sum(game["plies"] for game in summary["games"]), "scenes": len(scenes)}))
