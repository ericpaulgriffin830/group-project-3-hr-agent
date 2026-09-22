"""The operational trace. Contract B's `trace[]`.

The brief asks for "a visible or logged trace of agent reasoning steps at the
architectural level: selected tools, tool arguments, tool outputs, retrieved policy
sources, final answer basis, and any escalation decision" and then draws a hard
line: *"Do not expose hidden chain-of-thought; provide concise operational traces
instead."*

So this module records **what the agent did**, never **why it thought it should**.
That distinction is a rubric item, not a style preference, and it is enforced here
rather than left to whoever writes the next node: `add()` rejects a step carrying a
reasoning-shaped field, and `result_summary` is truncated to a short factual
string.

Enforcing it in one place matters because the leak is easy and invisible. The
gpt-oss models return a `reasoning` field next to `content`; a node that passed a
raw provider response into a trace step would ship literal chain-of-thought to the
UI, and nothing would look wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

StepType = Literal["intent", "retrieval", "tool_call", "synthesis", "guardrail",
                   "escalation"]

VALID_TYPES = frozenset(
    ("intent", "retrieval", "tool_call", "synthesis", "guardrail", "escalation")
)

#: Keys that would carry model reasoning into the trace. Checked by name because
#: the leak arrives as a field someone forwarded without looking, not as a decision.
FORBIDDEN_KEYS = frozenset(
    ("reasoning", "reasoning_content", "chain_of_thought", "thought", "thoughts",
     "scratchpad", "deliberation", "analysis", "thinking")
)

SUMMARY_LIMIT = 120


class ChainOfThoughtLeak(ValueError):
    """Raised when a step would put model reasoning into the trace.

    Loud on purpose. A silently dropped field would let the same mistake recur,
    and this is the one trace rule the brief states outright.
    """


@dataclass
class Trace:
    """An append-only record of one turn.

    Steps are numbered from 1 in the order they happened, which is what the UI
    panel renders and what the presenter narrates on camera.
    """

    steps: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def add(self, type: StepType, *, tool: str | None = None,
            args: dict | None = None, result_summary: str = "",
            status: str = "ok", latency_ms: int | None = None,
            **extra: Any) -> dict:
        """Append one step and return it.

        Extra fields are allowed -- an escalation carries a route, a synthesis
        carries a basis -- but never reasoning-shaped ones.
        """
        if type not in VALID_TYPES:
            raise ValueError(f"{type!r} is not a trace step type: {sorted(VALID_TYPES)}")

        leaked = FORBIDDEN_KEYS & set(extra)
        if leaked:
            raise ChainOfThoughtLeak(
                f"trace step would expose model reasoning via {sorted(leaked)}. "
                "The brief forbids exposing chain-of-thought; record what was done, "
                "not why."
            )

        step: dict[str, Any] = {
            "step": len(self.steps) + 1,
            "type": type,
            "result_summary": _truncate(result_summary),
            "status": status,
        }
        if tool is not None:
            step["tool"] = tool
        if args is not None:
            step["args"] = _scrub_args(args)
        if latency_ms is not None:
            step["latency_ms"] = latency_ms
        step.update(extra)

        self.steps.append(step)
        return step

    def add_tool_call(self, call: Any) -> dict:
        """Append a `ToolCall` from the MCP client."""
        return self.add(
            "tool_call", tool=call.tool, args=call.args,
            result_summary=call.result_summary(),
            status="ok" if call.ok else "error",
            latency_ms=call.latency_ms,
        )

    def as_list(self) -> list[dict]:
        return list(self.steps)

    def timings(self) -> dict:
        """Contract B's `timings`, summed from what actually happened.

        Retrieval and tool time are separated because the evaluation reports them
        separately, and because a slow turn is usually one or the other rather
        than "the agent".
        """
        retrieval = sum(s.get("latency_ms", 0) for s in self.steps
                        if s["type"] == "retrieval")
        tools = sum(s.get("latency_ms", 0) for s in self.steps
                    if s["type"] == "tool_call")
        return {"retrieval_ms": retrieval, "tools_ms": tools,
                "total_ms": sum(s.get("latency_ms", 0) for s in self.steps)}

    def tools_used(self) -> list[str]:
        """Ordered tool names. Feeds the tool-selection-accuracy metric."""
        return [s["tool"] for s in self.steps
                if s["type"] == "tool_call" and "tool" in s]

    def had_error(self) -> bool:
        return any(s.get("status") == "error" for s in self.steps)

    def render(self) -> str:
        """Plain-text rendering for logs and for narrating the demo."""
        lines = []
        for s in self.steps:
            latency = f" [{s['latency_ms']}ms]" if "latency_ms" in s else ""
            target = f" {s['tool']}" if "tool" in s else ""
            lines.append(f"{s['step']}. {s['type']}{target}{latency}: "
                         f"{s['result_summary']}")
        return "\n".join(lines)


def _truncate(text: str, limit: int = SUMMARY_LIMIT) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


#: Argument names whose values must never be written into a trace the UI renders.
SENSITIVE_ARGS = frozenset(("confirm_token", "api_key", "token", "password"))


def _scrub_args(args: dict) -> dict:
    """Redact secrets from recorded arguments.

    `confirm_token` is the one that actually occurs: it authorises a write, and a
    trace panel is rendered in a browser and pasted into bug reports. The trace
    should show that a token was present, not what it was.
    """
    return {k: ("<redacted>" if k in SENSITIVE_ARGS else v) for k, v in args.items()}
