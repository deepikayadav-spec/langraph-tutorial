"""
Video 15: LangSmith + LangGraph integration.

Graph structure (mirrors the video):
  START
    ├── evaluate_language  ─┐
    ├── evaluate_analysis  ─┼── final_evaluation ── END
    └── evaluate_clarity   ─┘

Three nodes run in PARALLEL; final_evaluation waits for all three.

LangSmith concepts:
  - Project  : the whole application (set via LANGCHAIN_PROJECT)
  - Trace    : one full graph execution (graph.invoke = 1 trace)
  - Run      : one node execution inside a trace
  - @traceable: makes a plain Python function appear as a sub-run inside its node

Without real LangSmith keys tracing is silently skipped; code runs normally.
To enable tracing: copy .env.example → .env, fill keys, re-run.
"""
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langsmith import traceable

# Set project name in code (overrides LANGCHAIN_PROJECT from .env)
# Each different LLM app should use its own project for clean separation
os.environ.setdefault("LANGCHAIN_PROJECT", "langgraph-essay-checker")


# ─── State ────────────────────────────────────────────────────────────────────

class EssayState(TypedDict):
    essay: str
    language_feedback: str
    language_score: int
    analysis_feedback: str
    analysis_score: int
    clarity_feedback: str
    clarity_score: int
    overall_feedback: str
    average_score: float


# ─── Mock evaluator ───────────────────────────────────────────────────────────
# In production: replace with ChatOpenAI(...).with_structured_output(schema)
# which returns {"feedback": str, "score": int} directly from the LLM.

class MockEssayEvaluator:

    def evaluate_language(self, essay: str) -> dict:
        words = essay.split()
        score = min(5, max(1, len(words) // 40 + 2))
        connectives = ["therefore", "however", "furthermore", "consequently", "moreover"]
        if any(w in essay.lower() for w in connectives):
            score = min(5, score + 1)
        level = "strong" if score >= 4 else "adequate" if score >= 3 else "basic"
        return {
            "feedback": (
                f"Language quality is {level}. "
                + (
                    "Vocabulary is varied and connectives are used effectively."
                    if score >= 4
                    else "Vocabulary is functional; vary sentence structure for better flow."
                )
            ),
            "score": score,
        }

    def evaluate_analysis(self, essay: str) -> dict:
        markers = ["because", "due to", "result", "cause", "effect",
                   "evidence", "argue", "suggests", "demonstrates", "shows"]
        count = sum(1 for m in markers if m in essay.lower())
        score = min(5, max(1, count // 2 + 2))
        return {
            "feedback": (
                (
                    "Analysis is thorough with well-supported arguments. "
                    "Reasoning is clearly connected to claims."
                )
                if score >= 4
                else (
                    "Analysis is present but could be deepened. "
                    "Strengthen links between evidence and conclusions."
                )
            ),
            "score": score,
        }

    def evaluate_clarity(self, essay: str) -> dict:
        structure_words = ["firstly", "secondly", "finally",
                           "in conclusion", "to summarize", "in addition"]
        has_structure = any(w in essay.lower() for w in structure_words)
        sentences = [s.strip() for s in essay.split(".") if s.strip()]
        avg_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        score = 4 if (10 <= avg_len <= 25 and has_structure) else 3
        return {
            "feedback": (
                "Clarity is excellent. Ideas flow logically with clear signposting."
                if score >= 4
                else "Clarity is satisfactory. Transitions between ideas could be clearer."
            ),
            "score": score,
        }

    def generate_overall(self, lang_fb: str, analysis_fb: str, clarity_fb: str) -> str:
        return (
            "Overall, the essay presents a coherent and structured argument. "
            + lang_fb.split(".")[0] + ". "
            + analysis_fb.split(".")[0] + ". "
            + clarity_fb.split(".")[0] + ". "
            "To improve: deepen analytical depth and add more specific evidence."
        )


_evaluator = MockEssayEvaluator()


# ─── Node functions ───────────────────────────────────────────────────────────
# @traceable makes each function appear as a sub-run INSIDE its node's run
# in LangSmith (node run → function sub-run → chain/LLM sub-sub-run).
# This is optional — remove decorators if you want simpler trace output.

@traceable(name="evaluate_language")
def evaluate_language_node(state: EssayState) -> dict:
    result = _evaluator.evaluate_language(state["essay"])
    return {
        "language_feedback": result["feedback"],
        "language_score": result["score"],
    }


@traceable(name="evaluate_analysis")
def evaluate_analysis_node(state: EssayState) -> dict:
    result = _evaluator.evaluate_analysis(state["essay"])
    return {
        "analysis_feedback": result["feedback"],
        "analysis_score": result["score"],
    }


@traceable(name="evaluate_clarity")
def evaluate_clarity_node(state: EssayState) -> dict:
    result = _evaluator.evaluate_clarity(state["essay"])
    return {
        "clarity_feedback": result["feedback"],
        "clarity_score": result["score"],
    }


@traceable(name="final_evaluation")
def final_evaluation_node(state: EssayState) -> dict:
    avg = round(
        (state["language_score"] + state["analysis_score"] + state["clarity_score"]) / 3,
        2,
    )
    overall = _evaluator.generate_overall(
        state["language_feedback"],
        state["analysis_feedback"],
        state["clarity_feedback"],
    )
    return {"overall_feedback": overall, "average_score": avg}


# ─── Graph ────────────────────────────────────────────────────────────────────

graph = StateGraph(EssayState)

graph.add_node("evaluate_language", evaluate_language_node)
graph.add_node("evaluate_analysis", evaluate_analysis_node)
graph.add_node("evaluate_clarity", evaluate_clarity_node)
graph.add_node("final_evaluation", final_evaluation_node)

# START fans out to all three evaluators simultaneously
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_clarity")

# All three converge — final_evaluation runs only after ALL THREE complete
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_clarity", "final_evaluation")

graph.add_edge("final_evaluation", END)

essay_evaluator = graph.compile()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE_ESSAY = """
    Climate change represents one of the most pressing challenges of our time.
    The evidence clearly shows that global temperatures have risen significantly
    due to human activities, particularly the burning of fossil fuels.
    Therefore, immediate action is required at both individual and governmental levels.

    Firstly, governments must implement stronger carbon pricing mechanisms.
    Because economic incentives drive behavior, carbon taxes can effectively
    reduce emissions. Furthermore, investment in renewable energy infrastructure
    creates long-term solutions. The impact of solar and wind energy expansion
    has already demonstrated promising results in several countries.

    However, technological solutions alone are insufficient. Individual behavior
    change is equally important. Consequently, education campaigns and community
    initiatives play a crucial role. In conclusion, addressing climate change
    requires coordinated effort across all sectors of society.
    """.strip()

    # config: run_name, tags, metadata appear in LangSmith dashboard per trace
    # Without real keys these are ignored but the code pattern is production-ready
    config = {
        "run_name": "evaluate-climate-essay",
        "tags": ["essay-evaluation", "langgraph", "demo"],
        "metadata": {
            "model": "MockEssayEvaluator",
            "version": "1.0",
            "topic": "climate-change",
        },
    }

    result = essay_evaluator.invoke({"essay": SAMPLE_ESSAY}, config=config)

    print("\n" + "=" * 52)
    print("  ESSAY EVALUATION RESULTS")
    print("=" * 52)
    print(f"  Language Score  : {result['language_score']}/5")
    print(f"  Analysis Score  : {result['analysis_score']}/5")
    print(f"  Clarity  Score  : {result['clarity_score']}/5")
    print(f"  Average  Score  : {result['average_score']}/5")
    print(f"\n  Language  : {result['language_feedback']}")
    print(f"\n  Analysis  : {result['analysis_feedback']}")
    print(f"\n  Clarity   : {result['clarity_feedback']}")
    print(f"\n  Overall   : {result['overall_feedback']}")
    print("=" * 52)
