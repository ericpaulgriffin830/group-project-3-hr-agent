# Deployment notes — Render free tier

**For the whole team.** Measured, not assumed — the commands are here so anyone can
re-run them. Written 2026-09-20 by Chris.

This is **input for Eric's `render.yaml` and `deployed.md`, and for Rob's embedding
swap** — not a replacement for either. Neither of those files is owned here.

---

## 1. Which Render product we need

**Web Services, Free plan. Two of them** — the FastAPI API (`/chat`, `/health`) and
the Streamlit UI. Nothing else on the Render menu applies.

| Render type | Free tier? | Us? |
|---|---|---|
| **Web Services** | yes | **Both our services** |
| Static Sites | yes | No — Streamlit is a running process, not static files |
| Postgres | yes | No — Chroma is file-based; the brief says a paid DB is not required |
| Key Value | yes | No |
| Cron Jobs | **no** | — |
| Private Services | **no** | — |
| Background Workers | **no** | — |
| Workflows | **no** | — |

The MCP server does **not** get its own service. It runs in-process with the API over
stdio, which is the architecture the brief itself recommends for free-tier hosting.
`MCP_TRANSPORT` switches it to `streamable-http` if we ever split it out.

## 2. Free-tier limits that shape our design

- **512 MB RAM, 0.1 CPU** per free instance. This is the binding constraint.
- **Spin-down after 15 minutes** of no inbound traffic; **~1 minute cold start**.
- **750 free instance hours per month per workspace** — a single pool **shared by both
  services**, not 750 each.

Three consequences:

**Two services beats one.** Each free instance gets its own 512 MB. Collapsing
Streamlit + FastAPI + Chroma + an embedding model into one instance makes the memory
problem worse, not better.

**Cold start is a graded item.** The brief requires `deployed.md` and the demo to
explain it, and the evaluation must report **cold-start and warm-start latency
separately**. Practically: hit both URLs a few minutes before recording on 10/1 so the
demo runs warm.

**The hour pool is shared.** Heavy testing in demo week can drain it, and when it is
gone *both* services stop.

## 3. The dependency problem — measured

`sentence-transformers` pulls `torch`, which on Linux pulls the whole CUDA stack.
Render free tier has no GPU, so **none of it is ever used**.

Resolved for Linux (what Render installs), deduplicated to one wheel per package:

| | Linux download | torch / nvidia / triton |
|---|---|---|
| **`sentence-transformers`** (current) | **3.46 GB** | 3.30 GB across 17 packages |
| **`fastembed`** (the planned swap) | **113 MB** | none |

**31× smaller.** Of the current 3.46 GB, only **159 MB is real dependency weight** —
the other 95% is GPU machinery. Installed on disk that is roughly 7–8 GB, against a
512 MB instance.

Reproduce it:

```bash
python3 - <<'PY'
import re, pathlib
best = {}
for b in pathlib.Path("uv.lock").read_text().split("[[package]]"):
    m = re.search(r'^name = "([^"]+)"', b, re.M)
    if not m: continue
    for wm in re.finditer(r'url = "([^"]+)".*?size = (\d+)', b, re.S):
        url, size = wm.group(1), int(wm.group(2))
        if ("manylinux" in url or "linux_x86_64" in url) and ("cp312" in url or "py3" in url or "abi3" in url):
            best[m.group(1)] = max(best.get(m.group(1), 0), size)
heavy = {k: v for k, v in best.items() if k.startswith("nvidia") or k in ("torch", "triton")}
print(f"linux download: {sum(best.values())/1e9:.2f} GB   torch/cuda: {sum(heavy.values())/1e9:.2f} GB")
PY
```

**This makes `sentence-transformers` → `fastembed` a deployment blocker, not a
cleanup item.** `Group_Assignment.md` currently files it under "Fixes" beside a README
typo. It belongs on the critical path — Rob owns the swap and the re-lock.

Note it is a real swap, not a drop-in: `fastembed` has its own API, and
`BAAI/bge-small-en-v1.5` (already pinned as `EMBED_MODEL` in Contract D) is available
in both, so the model choice does not change.

## 4. CI/CD — the gate does not exist yet

The brief requires **"Deployment must only occur if tests pass."** Today
`.github/workflows/ci.yml`:

- runs on **windows-latest and macos-latest only** — **no `ubuntu-latest`**, despite
  Render being Linux. The 3.46 GB Linux tree is therefore never exercised by CI.
- runs **no tests**. It installs and import-checks `chromadb, langchain, langgraph,
  sentence_transformers` — never our own packages. A completely broken `mcp_server`
  passed this check until 2026-09-20.
- has **no deploy job**, so nothing is gated on anything.
- only triggers on `main` and PRs into `main` — a feature-branch push runs no CI at all.

There are now **50 tests** in `tests/`. Wiring `pytest` into CI is the cheapest rubric
points available.

**Deploy trigger:** with auto-deploy OFF, CI needs a **deploy hook** — a secret
per-service URL from the service's Settings tab, `curl`ed from a job with
`needs: test`. **No Render API key is required.** The hook URL goes in **GitHub repo
secrets**, not `.env`.

Keep the two straight:
- `.env` → runtime config for the running app (`GROQ_API_KEY`, `GROQ_MODEL`, …)
- GitHub Actions secrets → things CI needs (the deploy hooks)

## 5. Which repo deploys

The graded submission is the **group repo**, and the brief wants the deployed URL in
its README. The live services should therefore deploy from
`Rob-Ottogalli/quantic-hr-ai-rag-group-project`, not from anyone's personal mirror.
A mirror-backed service is fine as personal staging, but its URL must not end up in
`deployed.md`.

## 6. What a deployable service needs

As of this writing the repo has no web app, so there is nothing for Render to serve:
no `render.yaml`, no `app/api.py`, no `ui/streamlit_app.py`. `main.py` prints a line
and exits.

A Render web service needs a process that **binds `$PORT` and keeps listening** — a
script that prints and exits fails the health check even when the build succeeds. For
the API that means something like
`uvicorn app.api:app --host 0.0.0.0 --port $PORT`.
