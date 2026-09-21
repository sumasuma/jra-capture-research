import hashlib
import json
import os
import shutil
import threading
import urllib.request
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from train_v2 import run_training

ROOT = Path(os.environ.get("JRA_WORKDIR", "/data/jra"))
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
INPUT.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get("UPLOAD_TOKEN", "")
DATA_URL = os.environ.get("DATA_URL", "")
AUTO_TRAIN = os.environ.get("AUTO_TRAIN", "0") == "1"
EXPECTED_BYTES = int(os.environ.get("DATA_EXPECTED_BYTES", "0") or 0)

app = FastAPI(title="JRA Capture Research Worker")
BOOTSTRAP_ERROR = None
TRAIN_THREAD = None


def auth(x_upload_token: str | None):
    if TOKEN and x_upload_token != TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


def file_sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_data():
    global BOOTSTRAP_ERROR
    BOOTSTRAP_ERROR = None
    if not DATA_URL:
        return
    target = INPUT / "central.zip"
    if target.exists() and target.stat().st_size > 0:
        print(
            f"[bootstrap] existing central.zip bytes={target.stat().st_size} sha256={file_sha(target)}",
            flush=True,
        )
        return
    tmp = INPUT / "central.zip.part"
    try:
        print("[bootstrap] downloading dataset", flush=True)
        urllib.request.urlretrieve(DATA_URL, tmp)
        size = tmp.stat().st_size
        sha = file_sha(tmp)
        print(f"[bootstrap] downloaded bytes={size} sha256={sha}", flush=True)
        if EXPECTED_BYTES and size != EXPECTED_BYTES:
            raise RuntimeError(f"dataset byte mismatch expected={EXPECTED_BYTES} got={size}")
        with zipfile.ZipFile(tmp) as z:
            members = len(z.namelist())
        print(f"[bootstrap] zip_valid members={members}", flush=True)
        tmp.replace(target)
    except Exception as exc:
        BOOTSTRAP_ERROR = repr(exc)
        print(f"[bootstrap:error] {exc!r}", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def maybe_start_training():
    global TRAIN_THREAD
    if not AUTO_TRAIN or BOOTSTRAP_ERROR:
        return
    status = OUTPUT / "status.json"
    if status.exists():
        try:
            s = json.loads(status.read_text(encoding="utf-8"))
            version = (s.get("summary") or {}).get("version")
            if s.get("complete") and version == "JRA_ADULT_DIRT_RUNTIME_V2_2":
                print("[train] v2.2 already complete; not restarting", flush=True)
                return
        except Exception:
            pass
    if TRAIN_THREAD and TRAIN_THREAD.is_alive():
        return
    TRAIN_THREAD = threading.Thread(target=run_training, args=(ROOT,), daemon=True)
    TRAIN_THREAD.start()
    print("[train] background training started", flush=True)


@app.on_event("startup")
def startup():
    bootstrap_data()
    maybe_start_training()


@app.get("/health")
def health():
    files = [
        {"name": p.name, "bytes": p.stat().st_size}
        for p in sorted(INPUT.glob("*"))
        if p.is_file()
    ]
    return {
        "ok": True,
        "workdir": str(ROOT),
        "input_files": files,
        "bootstrap_error": BOOTSTRAP_ERROR,
        "auto_train": AUTO_TRAIN,
        "training_alive": bool(TRAIN_THREAD and TRAIN_THREAD.is_alive()),
    }


@app.get("/status")
def status():
    p = OUTPUT / "status.json"
    if not p.exists():
        return {"phase": "NOT_STARTED", "complete": False}
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/summary")
def summary():
    p = OUTPUT / "SUMMARY.json"
    if not p.exists():
        return {"ready": False}
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/download/runtime")
def download_runtime():
    p = OUTPUT / "JRA_ADULT_DIRT_RUNTIME_V2_2.zip"
    if not p.exists():
        raise HTTPException(status_code=404, detail="runtime not ready")
    return FileResponse(p, media_type="application/zip", filename=p.name)


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
    with zipfile.ZipFile(zp) as z:
        members = [{"name": i.filename, "bytes": i.file_size} for i in z.infolist()[:200]]
    result = {"zip": zp.name, "bytes": zp.stat().st_size, "members": members}
    p = OUTPUT / "dataset_inspection.json"
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


@app.get("/artifact/{name}")
def artifact(name: str, x_upload_token: str | None = Header(default=None)):
    auth(x_upload_token)
    p = OUTPUT / name
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(p)
