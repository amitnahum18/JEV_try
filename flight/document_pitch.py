"""Build a transparent report from every saved pitch experiment, without API calls."""
from pathlib import Path
import hashlib
import json
from course import Config, generate_gates
from compare_axes import ROOT, write_json


def main():
    root = ROOT / "flight/recordings"
    paths = [root / "vertical_25_choices_preview_20260926/single_level1_seed44_232048_671155.json"]
    paths += sorted((root / "pitch_verification_20260927").glob("*/single*/episode.json"))
    paths += sorted((root / "pitch_confirmation_20260927").glob("*/episode.json"))
    paths += sorted((root / "pitch_v8_20260927").glob("*/episode.json"))
    records, rows = [], []
    for path in paths:
        r = json.loads(path.read_text(encoding="utf-8"))
        assert r["status"] != "flying", f"Trial is still running: {path}"
        cfg = Config(**r["config"])
        geometry_hash = hashlib.sha256(json.dumps(generate_gates(cfg), sort_keys=True).encode()).hexdigest()
        row = {"source": str(path), "seed": cfg.seed, "prompt": r["prompt_version"],
               "choices": cfg.joint_choice_count, "status": r["status"], "gates": r["gates_passed"],
               "seconds": r["simulated_seconds"], "cost_usd": r.get("known_cost_usd"),
               "course_hash": geometry_hash}
        records.append((path, r))
        rows.append(row)
    for seed in (44,45,46):
        assert len({row["course_hash"] for row in rows if row["seed"] == seed}) == 1
    latest = [(path,r) for path,r in records if "pitch_v8_20260927" in str(path)]
    assert len(latest) == 3
    selected_path, selected = max(latest, key=lambda item: (item[1]["status"] == "completed", item[1]["gates_passed"], item[1]["config"]["seed"] == 44))
    output = root / "pitch_v8_20260927"
    summary = {"all_attempts": rows, "latest_completed": sum(r["status"] == "completed" for _,r in latest),
               "latest_attempts": len(latest), "selected_source": str(selected_path),
               "selection": "Latest v8 trials: completed first, then gates passed, then original failed seed44. All trials preserved.",
               "limitations": "Three reused seeds, exploratory sample. Does not establish reliability or isolate causal effects. API waits pause physics. Pitch estimate is measured geometry, no action rollout/selection/override."}
    write_json(output / "verification_summary.json", summary)
    lines = ["# בדיקת JEV: פניות והפרשי גובה", "", "כל הניסויים השתמשו ב־JEV יחיד מול אותו מסלול לכל seed, באותה פיזיקה, מהירות, מרווחי מעבר ומרווח החלטה של 0.5 שניות. לא הוקלו המסגרות.", "", "| גרסת הנחיה | בחירות | Seed | מסגרות שעבר | תוצאה |", "|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| v{row['prompt']} | {row['choices']} | {row['seed']} | {row['gates']}/8 | {row['status']} |")
    lines += ["", "## התיקון", "", "הפספוס המקורי במסגרת 3 היה 0.08228 מטר מעל הגבול המותר. הנתונים התעדכנו למסגרת הנכונה מיד במעבר. LEVEL מכוון לזווית אף +2°, ולכן אינו משמר מהירות הנמכה קיימת. תוספת זוויות בלבד לא פתרה את הבעיה בניסוי שבוצע.", "", "תשע זוויות אף: +2, +4, +6, +12, +20, −3, −6, −10, −16 מעלות. כל אחת משולבת בחמש בחירות בנק: 0, ±25, ±40 — בסך הכול 45 בחירות. JEV מקבל אומדן גאומטרי לזווית אף המתאימה לפתח, המבוסס על חלון נתיב הטיסה ועל ההפרש הנמדד בין זווית האף לזווית נתיב הטיסה. v8 נותן לכל בחירה תנאים מספריים מפורשים. JEV בוחר דרך API; הקוד אינו בוחר או מחליף פעולה.", "", f"הגרסה האחרונה השלימה {summary['latest_completed']}/3 ניסויים. זהו מדגם קטן על seeds שכבר נבדקו; אינו מבטיח הצלחה בכל טיסה.", "", "## תיעוד", "", f"מקור הסרטון: `{selected_path.relative_to(ROOT).as_posix()}`.", "", "הסרטון מציג את השחזור הפיזיקלי המדויק ב־1x עם נתוני טיסה, פעולת JEV, הגאים, מסלול ותוצאה. זמני המתנה ל־API הוסרו, ובסוף יש שתי שניות עצירה. השחזור מאמת כל צעד, מספר מסגרות ותנוחה סופית. קבצי הבדיקות וה־API נשמרו בכל תיקיות הניסוי."]
    (output / "findings_he.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
