import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(os.environ.get("JRA_WORKDIR", "/tmp/jra"))
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
INPUT.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get("UPLOAD_TOKEN", "")

app = FastAPI(title="JRA Capture Research Worker")


def auth(x_upload_token: str | None):
    if TOKEN and x_upload_token != TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/health")
def health():
    return {"ok": True, "workdir": str(ROOT)}


@app.post("/upload")
async def upload(file: UploadFile = File(...), x_upload_token: str | None = Header(default=None)):
    auth(x_upload_token)
    target = INPUT / (file.filename or "dataset.zip")
    h = hashlib.sha256()
    size = 0
    with target.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            size += len(chunk)
    return {"ok": True, "path": str(target), "bytes": size, "sha256": h.hexdigest()}


@app.post("/inspect")
def inspect_dataset(x_upload_token: str | None = Header(default=None)):
    auth(x_upload_token)
    zips = sorted(INPUT.glob("*.zip"))
    if not zips:
        raise HTTPException(status_code=404, detail="no zip uploaded")
    zp = zips[-1]

    rows = []
    csv_names = []
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        csv_names = names
        for name in names[:80]:
            try:
                with z.open(name) as fh:
                    df = pd.read_csv(fh, encoding="cp932", nrows=5)
                rows.append({
                    "name": name,
                    "ncols": int(df.shape[1]),
                    "columns": [str(c) for c in df.columns[:40]],
                })
            except Exception as e:
                rows.append({"name": name, "error": repr(e)})

    result = {
        "zip": zp.name,
        "csv_count": len(csv_names),
        "sample_files": rows,
    }
    out = OUTPUT / "dataset_inspection.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


@app.get("/artifact/{name}")
def artifact(name: str, x_upload_token: str | None = Header(default=None)):
    auth(x_upload_token)
    p = OUTPUT / name
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(p)
