"""Repository hygiene: no data files, absolute local paths, e-mail addresses, tokens or network calls in the tree.

A pattern scan cannot establish consent or disclosure permission.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
DATA_SUFFIXES = {".npz", ".npy", ".mat", ".set", ".fdt", ".edf", ".bdf", ".fif", ".vhdr", ".vmrk", ".eeg", ".csv",
                 ".tsv", ".asc", ".pkl", ".pt", ".pth", ".h5", ".hdf5", ".png", ".jpg", ".jpeg", ".pdf", ".zip"}
PATTERNS = {
    "absolute local path": re.compile(r"(?<![\w.])/(?:home|media|mnt|Users|run/media|tmp|var|opt|srv)/"),
    "windows path": re.compile(r"[A-Za-z]:\\\\(?:Users|Data)"),
    "e-mail address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})\b"),
    "participant label": re.compile(r"\bsub-\d{2,}\b"),
}
ALLOWED_EMAILS = {"noreply@anthropic.com"}


def tracked_like_files():
    """Files Git tracks or would track (ignored outputs of the user's own runs are not part of the repository)."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=str(ROOT),
                             capture_output=True, check=True).stdout.decode()
        for rel in filter(None, out.split("\0")):
            p = ROOT / rel
            if p.is_file():
                yield p
    except (OSError, subprocess.CalledProcessError):  # not a Git checkout (e.g. a source archive)
        for p in ROOT.rglob("*"):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS):
                yield p


def test_no_data_or_binary_files():
    found = [str(p.relative_to(ROOT)) for p in tracked_like_files() if p.suffix.lower() in DATA_SUFFIXES]
    assert found == []


def test_no_private_patterns_in_text():
    hits = []
    for p in tracked_like_files():
        if p.suffix.lower() in DATA_SUFFIXES or p.name == "test_repo_hygiene.py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for name, pat in PATTERNS.items():
            for m in pat.finditer(text):
                if name == "e-mail address" and m.group(0) in ALLOWED_EMAILS:
                    continue
                hits.append((str(p.relative_to(ROOT)), name))
    assert hits == []


def test_package_makes_no_network_calls():
    src = ROOT / "src" / "eareeg_decoder"
    bad = re.compile(r"^\s*(?:import|from)\s+(?:urllib|requests|http|socket|ftplib|smtplib|httpx|aiohttp)\b", re.M)
    offenders = [p.name for p in src.glob("*.py") if bad.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
