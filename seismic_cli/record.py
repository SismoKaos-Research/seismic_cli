"""A provenance record written into every dataset this CLI generates.

A dataset directory used to hold tensors and a manifest and nothing saying how
they were made. The two datasets behind the deployed detector and magnitude
regressor could only be traced back through shell history and old session
logs, and one of them could not be traced at all.

`recorded` wraps a command so that it writes `<output dir>/dataset.json`:

- the command, argv and every resolved parameter, defaults included, because a
  default that changes later silently changes what re-running argv produces
- the git commit of this package and whether the tree was dirty, or the
  commit pip recorded when it was installed from git
- versions of the libraries that shape the tensors (torch, torchaudio, numpy,
  scipy, obspy)
- a fingerprint of every input path: file count, total size and a hash over
  the sorted relative names and sizes. The contents are not hashed, which
  keeps this cheap on 100k-file directories; a changed file keeps its name
  but usually not its size.
- what came out: manifest rows per split, tensor files per split, the
  manifest's sha256 and the shapes of one sample tensor

The record is written with status "started" before the command runs and
rewritten at the end with "ok", "failed" or "empty", so an interrupted build
still leaves evidence of what it was. "empty" means the command returned
without writing anything; most generators report their failures that way, as
a printed [ERROR] and a normal return, so the wrapper then exits with status 1
instead of 0. An earlier record in the same directory is
kept as `dataset.<its start time>.json`, not overwritten.
"""
import functools
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

RECORD = "dataset.json"
_PACKAGE = "download-labels"
_LIBRARIES = ("torch", "torchaudio", "numpy", "scipy", "obspy", "pandas")


def _git(*args):
    """Runs git in this package's checkout; None outside one."""
    try:
        out = subprocess.run(("git",) + args, capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _source():
    """Where this package's code came from: a checkout, or a pip-recorded commit."""
    commit = _git("rev-parse", "HEAD")
    if commit is not None:
        return {"git_commit": commit,
                # A build from an edited tree cannot be reproduced from its SHA.
                "git_dirty": bool(_git("status", "--porcelain"))}
    src = {"git_commit": None, "git_dirty": None}
    try:
        direct = metadata.distribution(_PACKAGE).read_text("direct_url.json")
        if direct:
            d = json.loads(direct)
            src["installed_from"] = d.get("url")
            src["git_commit"] = d.get("vcs_info", {}).get("commit_id")
    except Exception:
        pass
    return src


def _versions():
    """Versions of this package and the libraries that shape the tensors."""
    out = {}
    for name in (_PACKAGE,) + _LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _fingerprint(path):
    """Identity of one input path, without reading file contents.

    Returns None for a value that is not an existing path.
    """
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        st = p.stat()
        h = hashlib.sha256()
        # Catalogues and station tables are small and are where a silent edit
        # hurts most, so hash their content; large files by name and size.
        if st.st_size <= 64 * 2**20:
            h.update(p.read_bytes())
            how = "content"
        else:
            h.update(f"{p.name}\t{st.st_size}".encode())
            how = "name+size"
        return {"path": str(p.resolve()), "kind": "file", "bytes": st.st_size,
                "sha256": h.hexdigest(), "hashed": how}
    entries = []
    for root, _, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                entries.append((str(fp.relative_to(p)), fp.stat().st_size))
            except OSError:
                continue
    entries.sort()
    h = hashlib.sha256()
    for rel, size in entries:
        h.update(f"{rel}\t{size}\n".encode())
    return {"path": str(p.resolve()), "kind": "dir", "files": len(entries),
            "bytes": sum(s for _, s in entries), "sha256": h.hexdigest(),
            "hashed": "relative names + sizes"}


def _outputs(out_dir):
    """Summary of what a generator left in its output directory."""
    out_dir = Path(out_dir)
    res = {}
    manifest = out_dir / "manifest.csv"
    if manifest.exists():
        res["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        try:
            import pandas as pd
            m = pd.read_csv(manifest)
            res["manifest_rows"] = len(m)
            res["manifest_columns"] = list(m.columns)
            if "split" in m.columns:
                res["rows_per_split"] = {str(k): int(v) for k, v in m["split"].value_counts().items()}
            if "label" in m.columns:
                res["rows_per_label"] = {str(k): int(v) for k, v in m["label"].value_counts().items()}
        except Exception as e:
            res["manifest_error"] = repr(e)
    files = {}
    sample = None
    for split in ("train", "val", "test"):
        d = out_dir / split
        if d.is_dir():
            pts = sorted(d.glob("*.pt"))
            files[split] = len(pts)
            sample = sample or (pts[0] if pts else None)
    if files:
        res["tensor_files_per_split"] = files
    if sample is not None:
        try:
            import torch
            t = torch.load(sample, weights_only=False, map_location="cpu")
            items = t.items() if isinstance(t, dict) else [("tensor", t)]
            res["sample_shapes"] = {k: list(v.shape) for k, v in items if hasattr(v, "shape")}
            res["sample_dtypes"] = {k: str(v.dtype) for k, v in items if hasattr(v, "dtype")}
        except Exception as e:
            res["sample_error"] = repr(e)
    return res


def _wrote_anything(out_dir, record_name):
    """Whether a command left any file in `out_dir` besides its own records."""
    stem = Path(record_name).stem
    for root, _, files in os.walk(out_dir):
        for f in files:
            if not (Path(root) == Path(out_dir) and f.startswith(stem + ".")):
                return True
    return False


def _plain(v):
    """JSON-safe form of a parameter value."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return str(v)


def _write(path, rec):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    tmp.replace(path)


def recorded(command, out_param="output_dir", record_name=RECORD):
    """Decorates a CLI command so that it writes a provenance record.

    Apply below `@app.command(...)`. The signature is preserved, so typer
    builds the same options.

    Args:
        command: The command's CLI name, as the user types it.
        out_param: The parameter holding the output directory.
        record_name: File name of the record inside that directory.
    """
    def wrap(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def run(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = {k: _plain(v) for k, v in bound.arguments.items()}
            out_dir = Path(params[out_param])
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / record_name
            if path.exists():
                try:
                    old = json.loads(path.read_text()).get("started_utc", "")
                    stamp = old.replace(":", "").replace("-", "")[:15] or str(int(time.time()))
                except Exception:
                    stamp = str(int(time.time()))
                path.replace(path.with_name(f"{path.stem}.{stamp}.json"))

            inputs = {}
            for k, v in params.items():
                if k == out_param or not isinstance(v, str) or not v:
                    continue
                fp = _fingerprint(v)
                if fp is not None:
                    inputs[k] = fp
            rec = {
                "command": f"seismic-cli {command}",
                "status": "started",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "argv": sys.argv,
                "params": params,
                "inputs": inputs,
                **_source(),
                "versions": _versions(),
                "host": os.uname().nodename,
                "python": sys.version.split()[0],
            }
            _write(path, rec)
            t0 = time.time()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                rec.update(status="failed", error=repr(e),
                           traceback=traceback.format_exc(limit=8))
                raise
            else:
                rec["status"] = "ok" if _wrote_anything(out_dir, record_name) else "empty"
            finally:
                rec["ended_utc"] = datetime.now(timezone.utc).isoformat()
                rec["duration_s"] = round(time.time() - t0, 1)
                try:
                    rec["outputs"] = _outputs(out_dir)
                except Exception as e:
                    rec["outputs"] = {"error": repr(e)}
                _write(path, rec)
            if rec["status"] == "empty":
                # The generators report most failures as a printed [ERROR] and a
                # plain return, which exits 0 with nothing written; a script
                # chaining steps would carry on to train on an empty directory.
                print(f"seismic-cli {command}: nothing was written to {out_dir}; "
                      f"exiting with status 1", file=sys.stderr)
                raise SystemExit(1)
            return result
        return run
    return wrap
