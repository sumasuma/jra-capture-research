import hashlib
import json
import os
import shutil
import urllib.request
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
DATA_URL = os.environ.get("DATA_URL", "")

app = FastAPI(title="JRA Capture Research Worker")


def auth(x_upload_token: str | None):
    if TOKEN and x_upload_token != TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


def bootstrap_data():
    if not DATA_URL:
        return
    target = INPUT / "central.zip"
    if target.exists() and target.stat().st_size > 0:
        return
    tmp = INPUT / "central.zip.part"
    urllib.request.urlretrieve(DATA_URL, tmp)
    tmp.replace(target)


@app.on_event("startup")
def startup():
    bootstrap_data()


@app.get("/health")
def health():
    files = [
        {"name": p.name, "bytes": p.stat().st_size}
        for p in sorted(INPUT.glob("*"))
        if p.is_file()
    ]
    return {"ok": True, "workdir": str(ROOT), "input_files": files}


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


def inspect_zip(path: Path):
    out = {"zip": path.name, "bytes": path.stat().st_size, "members": []}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist()[:200]:
            out["members"].append({"name": info.filename, "bytes": info.file_size})
    return out


@app.post("/inspect")
def inspect_dataset(x_upload_token: str | None = Header(default=None)):
    auth(x_upload_token)
    zips = sorted(INPUT.glob("*.zip"))
    if not zips:
        raise HTTPException(status_code=404, detail="no zip uploaded")
    zp = zips[-1]

    result = inspect_zip(zp)
    nested = []
    with zipfile.ZipFile(zp) as z:
        for name in z.namelist():
            if name.lower().endswith(".zip"):
                target = INPUT / Path(name).name
                if not target.exists():
                    with z.open(name) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                nested.append(inspect_zip(target))
    result["nested_zips"] = nested

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
