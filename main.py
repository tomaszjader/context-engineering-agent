import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore")
load_dotenv()

# =========================
# 1. Runtime limits (budget-safe)
# =========================

MAX_STEPS = 4
MAX_MEMORY_RESULTS = 2
MAX_MEMORY_FACTS_PER_STEP = 4
MAX_MEMORY_CHARS = 1000
MAX_WEB_SNIPPETS_CHARS = 2200
MAX_STEP_RESULT_CHARS = 1800
MAX_SUMMARY_LINES = 8

FAST_MODEL = "gpt-4o-mini"
FINAL_MODEL = "gpt-4o-mini"

# USD per 1M tokens (approximate list prices; update when provider pricing changes)
MODEL_PRICING_USD_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

planner_llm = ChatOpenAI(model=FAST_MODEL, temperature=0, max_tokens=250)
executor_llm = ChatOpenAI(model=FAST_MODEL, temperature=0, max_tokens=550)
final_llm = ChatOpenAI(model=FINAL_MODEL, temperature=0, max_tokens=500)

# =========================
# 2. Vector memory (only concise facts)
# =========================

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(
    [Document(page_content="Agent memory initialized.")],
    embeddings,
)


@dataclass
class UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RunTelemetry:
    by_stage: Dict[str, UsageTotals] = field(default_factory=dict)
    by_model: Dict[str, UsageTotals] = field(default_factory=dict)

    def add(self, stage: str, model: str, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        if stage not in self.by_stage:
            self.by_stage[stage] = UsageTotals()
        if model not in self.by_model:
            self.by_model[model] = UsageTotals()

        for bucket in (self.by_stage[stage], self.by_model[model]):
            bucket.input_tokens += input_tokens
            bucket.output_tokens += output_tokens
            bucket.total_tokens += total_tokens

    def estimate_cost_usd(self) -> float:
        total = 0.0
        for model, usage in self.by_model.items():
            pricing = MODEL_PRICING_USD_PER_1M.get(model)
            if not pricing:
                continue
            total += (usage.input_tokens / 1_000_000) * pricing["input"]
            total += (usage.output_tokens / 1_000_000) * pricing["output"]
        return total

    def print_report(self) -> None:
        print("\nUsage telemetry (LLM only):")
        if not self.by_stage:
            print("- No LLM usage captured.")
            return

        for stage, usage in self.by_stage.items():
            print(
                f"- {stage}: input={usage.input_tokens}, "
                f"output={usage.output_tokens}, total={usage.total_tokens}"
            )

        total_input = sum(x.input_tokens for x in self.by_stage.values())
        total_output = sum(x.output_tokens for x in self.by_stage.values())
        total_tokens = sum(x.total_tokens for x in self.by_stage.values())
        est_cost = self.estimate_cost_usd()
        print(f"- total: input={total_input}, output={total_output}, total={total_tokens}")
        print(f"- estimated_cost_usd={est_cost:.6f} (excludes embeddings/search, approx pricing)")


def _extract_token_usage(response: Any) -> Tuple[int, int, int]:
    usage_meta = getattr(response, "usage_metadata", None) or {}
    input_tokens = int(usage_meta.get("input_tokens", 0))
    output_tokens = int(usage_meta.get("output_tokens", 0))
    total_tokens = int(usage_meta.get("total_tokens", 0))

    if total_tokens:
        return input_tokens, output_tokens, total_tokens

    response_meta = getattr(response, "response_metadata", None) or {}
    token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}
    input_tokens = int(token_usage.get("prompt_tokens", 0))
    output_tokens = int(token_usage.get("completion_tokens", 0))
    total_tokens = int(token_usage.get("total_tokens", input_tokens + output_tokens))
    return input_tokens, output_tokens, total_tokens


def _invoke_with_telemetry(
    llm: ChatOpenAI,
    messages: List[Any],
    telemetry: RunTelemetry,
    stage: str,
) -> Any:
    response = llm.invoke(messages)
    input_tokens, output_tokens, total_tokens = _extract_token_usage(response)
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    telemetry.add(stage, model_name, input_tokens, output_tokens, total_tokens)
    return response


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def extract_facts(text: str) -> List[str]:
    lines = [line.strip("-* ").strip() for line in text.splitlines() if line.strip()]
    facts = [line for line in lines if len(line) > 20]
    if not facts and text.strip():
        facts = [text.strip()]
    return [clip_text(fact, 220) for fact in facts[:MAX_MEMORY_FACTS_PER_STEP]]


def save_to_memory(text: str) -> None:
    facts = extract_facts(text)
    if not facts:
        return
    vector_store.add_documents([Document(page_content=fact) for fact in facts])


def retrieve_memory(query: str) -> str:
    docs = vector_store.similarity_search(query, k=MAX_MEMORY_RESULTS)
    joined = "\n".join(d.page_content for d in docs if d.page_content.strip())
    return clip_text(joined, MAX_MEMORY_CHARS)


# =========================
# 3. Planner (JSON steps + hard cap)
# =========================

def create_plan(goal: str, telemetry: RunTelemetry) -> List[str]:
    response = _invoke_with_telemetry(
        planner_llm,
        [
            SystemMessage(
                content=(
                    "Create a concise execution plan. Return only valid JSON "
                    'with schema: {"steps": ["..."]}. Use 2-4 steps.'
                )
            ),
            HumanMessage(content=f"Goal: {goal}"),
        ],
        telemetry,
        "planner",
    )

    raw = (response.content or "").strip()
    try:
        data = json.loads(raw)
        steps = data.get("steps", [])
        cleaned = [str(step).strip() for step in steps if str(step).strip()]
        return cleaned[:MAX_STEPS]
    except json.JSONDecodeError:
        fallback = [line.strip("-* 1234567890. ").strip() for line in raw.splitlines()]
        return [line for line in fallback if line][:MAX_STEPS]


# =========================
# 4. Executor (single web search per step)
# =========================

search_tool = DuckDuckGoSearchRun()


def execute_step(step: str, state_summary: str, telemetry: RunTelemetry) -> str:
    memory_context = retrieve_memory(step)

    query = (
        f"{step}\n"
        f"Known context:\n{clip_text(state_summary, 600)}\n"
        f"Relevant memory:\n{memory_context}"
    )
    web_snippets = search_tool.run(query)
    web_snippets = clip_text(web_snippets or "No web results.", MAX_WEB_SNIPPETS_CHARS)

    prompt = (
        "You are an analyst. Based on search snippets, provide concise, factual output.\n"
        "Output format:\n"
        "1) 3-5 bullet facts\n"
        "2) Short conclusion (2 sentences)\n"
        "If evidence is weak, say so clearly.\n\n"
        f"Step: {step}\n\n"
        f"Search snippets:\n{web_snippets}"
    )
    response = _invoke_with_telemetry(
        executor_llm,
        [HumanMessage(content=prompt)],
        telemetry,
        "executor",
    )
    return clip_text(response.content or "", MAX_STEP_RESULT_CHARS)


# =========================
# 5. Rolling summary (no extra API calls)
# =========================

def update_state_summary(summary_lines: List[str], step: str, result: str) -> List[str]:
    first_line = next((line.strip() for line in result.splitlines() if line.strip()), "No result.")
    entry = f"- {step}: {clip_text(first_line, 180)}"
    summary_lines.append(entry)
    return summary_lines[-MAX_SUMMARY_LINES:]


# =========================
# 6. Main loop
# =========================

def run_agent(goal: str) -> None:
    print("Goal:", goal)
    telemetry = RunTelemetry()

    steps = create_plan(goal, telemetry)
    if not steps:
        print("No executable steps were produced by planner.")
        return

    print(f"Planned steps: {len(steps)}")

    summary_lines = ["- Task started."]

    for i, step in enumerate(steps, start=1):
        print(f"\nStep {i}/{len(steps)}: {step}")
        current_summary = "\n".join(summary_lines)

        result = execute_step(step, current_summary, telemetry)
        print(result)

        summary_lines = update_state_summary(summary_lines, step, result)
        save_to_memory(result)

    final_prompt = (
        "Create the final answer from execution summary.\n"
        "Requirements: concrete recommendations, no fluff, max 180 words.\n\n"
        f"Goal: {goal}\n"
        f"Execution summary:\n{chr(10).join(summary_lines)}"
    )
    final_answer = _invoke_with_telemetry(
        final_llm,
        [HumanMessage(content=final_prompt)],
        telemetry,
        "final",
    ).content

    print("\nFinal answer:")
    print(final_answer)
    telemetry.print_report()


if __name__ == "__main__":
    run_agent(
        "Search for popular free AI courses and recommend 3 best focused on building RAG systems with LangChain."
    )
