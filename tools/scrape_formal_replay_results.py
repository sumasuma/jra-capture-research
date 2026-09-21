from __future__ import annotations
import re, time, json
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup

DATES = [
    ("2026-09-12", 3),
    ("2026-09-13", 4),
    ("2026-09-19", 5),
]
COURSES = {
    "06": "中山",
    "09": "阪神",
}
MEETING = "04"
UA = {"User-Agent":"Mozilla/5.0 (compatible; JRAResearchAudit/1.0)"}

def sec(x):
    x=str(x).strip()
    if not x or x in {"--","---"}: return None
    m=re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)",x)
    if not m:return None
    return (int(m.group(1) or 0)*60)+float(m.group(2))

def class_code(title, meta):
    s=f"{title} {meta}"
    if "G1" in s or "Ｇ１" in s: return 195
    if "G2" in s or "Ｇ２" in s: return 179
    if "G3" in s or "Ｇ３" in s: return 147
    if re.search(r"\bL\b|\(L\)|（L）|リステッド",s): return 131
    if "オープン" in s or "OP" in s: return 115
    if "3勝" in s or "３勝" in s: return 67
    if "2勝" in s or "２勝" in s: return 43
    if "1勝" in s or "１勝" in s: return 23
    if "未勝利" in s: return 7
    if "新馬" in s: return 3
    return 0

def cell_text(tr, cls):
    td=tr.select_one("td."+cls)
    return td.get_text(" ",strip=True) if td else ""

def scrape_one(race_id12, date, venue):
    url=f"https://race.netkeiba.com/race/result.html?race_id={race_id12}"
    rr=requests.get(url,headers=UA,timeout=30)
    rr.raise_for_status()
    rr.encoding=rr.apparent_encoding or rr.encoding
    soup=BeautifulSoup(rr.text,"html.parser")
    title=(soup.select_one(".RaceName") or soup.select_one("h1"))
    title=title.get_text(" ",strip=True) if title else ""
    d1=soup.select_one(".RaceData01")
    meta=d1.get_text(" ",strip=True) if d1 else ""
    md=re.search(r"(芝|ダ|障)(\d+)m",meta)
    if not md:
        return {"audit":{"race_id12":race_id12,"url":url,"status":"NO_DISTANCE","title":title,"meta":meta},"rows":[]}
    surface=md.group(1); distance=int(md.group(2))
    if surface=="障" or "障害" in title or "障害" in meta:
        return {"audit":{"race_id12":race_id12,"url":url,"status":"SKIP_OBSTACLE","title":title,"meta":meta},"rows":[]}
    gm=re.search(r"馬場:([^\s/]+)",meta)
    going=gm.group(1) if gm else ""
    table=soup.select_one("table.RaceTable01") or soup.select_one("table")
    if table is None:
        return {"audit":{"race_id12":race_id12,"url":url,"status":"NO_TABLE","title":title,"meta":meta},"rows":[]}
    rows=[]
    result_rows=table.select("tr.HorseList")
    if not result_rows:
        result_rows=[tr for tr in table.find_all("tr") if tr.select_one('a[href*="/horse/"]')]
    for tr in result_rows:
        # netkeiba occasionally changes/omits td class names. Keep the
        # semantic-class path, with a stable positional fallback.
        tds=tr.find_all("td", recursive=False)
        def pos(i):
            return tds[i].get_text(" ",strip=True) if 0 <= i < len(tds) else ""

        rank=cell_text(tr,"Rank") or pos(0)
        rank=str(rank).strip()
        if not rank.isdigit(): continue
        finish=int(rank)

        no=cell_text(tr,"Num.Txt_C") or cell_text(tr,"Num")
        if not str(no).strip().isdigit():
            no=pos(2)
        if not str(no).strip().isdigit(): continue
        horse_no=int(str(no).strip())

        hi=tr.select_one("td.Horse_Info")
        if hi is None and len(tds) > 3:
            hi=tds[3]
        ha=hi.select_one('a[href*="/horse/"]') if hi else None
        if ha is None: continue
        horse_name=ha.get_text(" ",strip=True)
        hm=re.search(r"/horse/(\\d+)",ha.get("href",""))
        if not hm: continue
        horse_id=hm.group(1)

        barei=cell_text(tr,"Barei") or pos(4)
        am=re.search(r"(\\d+)",barei)
        age=int(am.group(1)) if am else None

        wt=cell_text(tr,"Weight") or pos(5)
        wtm=re.search(r"(\\d+(?:\\.\\d+)?)",wt)
        carried=float(wtm.group(1)) if wtm else None

        jk=cell_text(tr,"Jockey") or pos(6)
        tm=cell_text(tr,"Time") or pos(7)
        actual=sec(tm)
        if actual is None: continue
        rows.append({
            "race_id16":date.replace("-","")+race_id12[4:],
            "date":date,
            "horse_id":horse_id,
            "horse_name":horse_name,
            "horse_no":horse_no,
            "finish":finish,
            "actual_time":actual,
            "venue":venue,
            "surface":surface,
            "distance":distance,
            "track_state":going,
            "jockey":jk,
            "carried_weight":carried,
            "age":age,
            "class_code":class_code(title,meta),
            "source_url":url,
        })
    status="PASS" if len(rows)>=6 and sum(x["finish"]==1 for x in rows)==1 else "SKIP_INVALID_FIELD"
    return {"audit":{"race_id12":race_id12,"url":url,"status":status,"title":title,"meta":meta,"valid_finishers":len(rows)},"rows":rows if status=="PASS" else []}

def main():
    out=[]; audits=[]
    for date,day in DATES:
        for course,venue in COURSES.items():
            for race_no in range(1,13):
                rid=f"2026{course}{MEETING}{day:02d}{race_no:02d}"
                try:
                    r=scrape_one(rid,date,venue)
                except Exception as e:
                    r={"audit":{"race_id12":rid,"status":"ERROR","error":repr(e)},"rows":[]}
                audits.append(r["audit"]);out.extend(r["rows"])
                print(r["audit"],flush=True)
                time.sleep(.25)
    df=pd.DataFrame(out)
    if not df.empty:
        df=df.sort_values(["date","race_id16","finish","horse_no"])
    Path("artifacts").mkdir(exist_ok=True)
    df.to_csv("artifacts/formal_replay_results_20260912_13_19.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv("artifacts/formal_replay_scrape_audit.csv",index=False,encoding="utf-8-sig")
    summary={
        "rows":int(len(df)),
        "races":int(df.race_id16.nunique()) if len(df) else 0,
        "dates":sorted(df.date.unique().tolist()) if len(df) else [],
        "obstacle_skips":sum(a.get("status")=="SKIP_OBSTACLE" for a in audits),
        "errors":sum(a.get("status")=="ERROR" for a in audits),
        "invalid_fields":sum(a.get("status")=="SKIP_INVALID_FIELD" for a in audits),
        "market_fields_saved":False,
    }
    Path("artifacts/formal_replay_scrape_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":
    main()
