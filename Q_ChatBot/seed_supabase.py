import os
from pathlib import Path
from typing import List, Dict, Iterable
from supabase import create_client

KNOW_FILE = Path("knowledge.txt")
BATCH_SIZE = 500

def parse_qas(path: Path) -> List[Dict[str, str]]:
    """Parse Q:/A: alternating lines into [{'q':..., 'a':...}, ...]."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.resolve()}")
    rows: List[Dict[str, str]] = []
    buf_q = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("Q:"):
            buf_q = ln[2:].strip()
        elif ln.startswith("A:") and buf_q is not None:
            rows.append({"q": buf_q, "a": ln[2:].strip()})
            buf_q = None
    return rows

def chunks(lst: List[Dict[str, str]], n: int) -> Iterable[List[Dict[str, str]]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def main():
    url = os.environ.get("SUPABASE_URL")
    srk = os.environ.get("SUPABASE_SERVICE_ROLE")
    if not url or not srk:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE in env")

    sb = create_client(url, srk)
    rows = parse_qas(KNOW_FILE)
    if not rows:
        print("No Q/A pairs found in knowledge.txt"); return

    total = 0
    for batch in chunks(rows, BATCH_SIZE):
        # upsert so re-running the script won’t duplicate entries
        sb.table("knowledge").upsert(batch, on_conflict="q").execute()
        total += len(batch)
    print(f"✅ Seeded/updated {total} Q/A rows into public.knowledge")

if __name__ == "__main__":
    main()
