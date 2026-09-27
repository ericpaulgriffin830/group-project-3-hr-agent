# Deployed application

> **FILL IN BEFORE SUBMISSION.** Everything except the three URLs is written and
> checked. Replace the `<…>` placeholders once the group-repo services exist, and
> delete this block. — Chris

## URLs

| | |
|---|---|
| **Application (UI)** | `<https://hr-agent-ui-XXXX.onrender.com>` |
| **API** | `<https://hr-agent-api-XXXX.onrender.com>` |
| **Health endpoint** | `<https://hr-agent-api-XXXX.onrender.com/health>` |

`/health` returns:

```json
{"status": "ok", "mcp_connected": true, "tools_discovered": 8, "index_ready": true}
```

All four fields matter. `mcp_connected` and `tools_discovered` confirm the agent
reached the MCP server and discovered its tools at startup; `index_ready` confirms
the Chroma index was built. A service that answers but reports
`index_ready: false` has no corpus behind it.

## Cold-start behaviour

Both services run on Render's **free instance type: 512 MB RAM, 0.1 CPU**.

**Free instances spin down after 15 minutes without inbound traffic, and take
roughly one minute to wake.** The first request after an idle period will hang on
a Render loading page, then complete normally. Nothing is lost — it is a cold
start, not a failure.

Two further limits worth knowing when grading:

- **750 free instance hours per month are shared across the whole workspace**, not
  allocated per service. Two services draw on one pool.
- **Groq's free tier is 200,000 tokens per day per account.** The app rotates
  across three accounts, so a single heavy session will not exhaust it — but a
  throttled call degrades *quietly*: the agent catches the failure, records it in
  the trace as a guardrail step with `status: error`, and answers from whatever
  evidence it already has. **If an answer looks thin, check the trace panel for
  that step before concluding the agent got it wrong.**

**If you are grading this:** load the UI and send one throwaway question, wait for
it to return, then run the demo tasks. That takes the cold start out of the
measurement.

## What runs where

A single API service holds the web API, the agent orchestrator, the MCP server
(in-process over stdio), the Chroma index and the mock HR data — the free-tier
architecture the brief recommends. The UI is a second service that talks to it
over `API_BASE_URL`.

The index is **built at deploy time**, not committed:

```yaml
buildCommand: "pip install uv && uv sync --locked && uv run python -m scripts.build_index"
```

So the deployed service always indexes the corpus that shipped with that commit.

`MCP_TRANSPORT` selects `stdio` (default, in-process) or `streamable-http` if the
MCP server is ever split into its own service. Nothing else in the codebase
branches on transport.

## Deployment pipeline

Push to `main` → GitHub Actions runs the full test suite on **Windows, macOS and
Linux** → the deploy job runs **only if tests pass** (`needs: test`), triggering
each service's Render deploy hook.

```yaml
deploy:
  needs: test
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
```

Auto-deploy is **off** on the Render side so this gate is the only path to
production — deployment cannot happen on a red build.

## Reproducing the demo tasks

The UI sidebar carries two **Run demo task** buttons, which POST to `/chat` with
`demo_task: "A"` or `"B"` and the matching persona.

| | Question | Persona |
|---|---|---|
| **A** | Can I work from Colorado for six weeks? | E1007 — Sofia Reyes, Consultant, Chicago |
| **B** | Can I take three days of PTO in mid-October? | E1008 — Grace Lin, Client Support, Denver |

Task A is the multi-document case: it cites the remote-work policy and the
tax/work-location policy together. Task B answers **no** on two independent
grounds — a 1.5-day balance against a three-day request, and a department blackout
covering 10/05–10/16.

Either can also be typed directly into the chat box.

## Known free-tier limitations

- Cold start after 15 minutes idle, as above.
- The Chroma index is rebuilt on every deploy, so a deploy takes a few minutes
  longer than a code-only change would.
- No persistent disk: mock HR tickets created through `create_mock_hr_ticket` do
  not survive a restart. They are mock writes by design — the brief requires these
  actions be mock or confirmed, and they are both.
