"""
Automated evaluation of the GraphRAG system against the CDKGQA benchmark.

Reads questions and baseline answers from QA/CDKGQA.csv, runs each question
through GraphRAG, and uses an LLM judge to score the response.

Usage:
    uv run evaluate.py
    uv run evaluate.py --output results.json
    uv run evaluate.py --client GeminiFlash   # the chat on the ingestion client, to compare
"""

import argparse
import csv
import json
import os
import statistics
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ["BAML_LOG"] = "WARN"

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

import config
from rag import GraphRAG

QA_CSV = config.QA_CSV

SCORE_LABELS = {
    1: "no_answer",
    2: "wrong",
    3: "partial",
    4: "acceptable",
    5: "correct",
}


def load_questions(path: Path) -> list[dict]:
    questions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            question = row.get("Question", "").strip()
            baseline = row.get("Baseline answer", "").strip()
            if question:
                questions.append({"id": i, "question": question, "baseline": baseline})
    return questions


_judge_client = None


def get_judge_client() -> genai.Client:
    global _judge_client
    if _judge_client is None:
        _judge_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _judge_client


JUDGE_MODEL = "gemini-3.7-flash"


def judge_response(question: str, baseline: str, response: str,
                   model: str = JUDGE_MODEL) -> tuple[int, str]:
    """Use an LLM to score the RAG response against the baseline. Returns (score 1-5, reasoning)."""
    prompt = f"""You are evaluating a RAG system's answer against a baseline (expected) answer.

Score the system's answer on a scale of 1-5:
1 = no_answer: The system returned nothing useful or said it doesn't know
2 = wrong: The answer is factually incorrect or completely off-topic
3 = partial: The answer addresses the question but is missing key information from the baseline
4 = acceptable: The answer is mostly correct and useful, minor gaps acceptable
5 = correct: The answer is accurate and covers the key points in the baseline

QUESTION: {question}

BASELINE ANSWER: {baseline}

SYSTEM ANSWER: {response}

Respond with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>"}}"""

    # Google's 503 "high demand" spikes pass within a minute; without a retry
    # one of them aborts the whole run after the answers were already paid for.
    for attempt in range(5):
        try:
            result = get_judge_client().models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            break
        except genai_errors.ServerError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt * 2)
    try:
        parsed = json.loads(result.text)
        return int(parsed["score"]), parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        for ch in result.text:
            if ch.isdigit() and 1 <= int(ch) <= 5:
                return int(ch), result.text
        return 1, f"Could not parse judge response: {result.text}"


def run_evaluation(output_path: str | None = None, judge_model: str = JUDGE_MODEL,
                   pause: float = 0) -> list[dict]:
    questions = load_questions(QA_CSV)
    rag = GraphRAG()
    results = []

    print(f"Running evaluation on {len(questions)} questions...\n")

    for index, q in enumerate(questions):
        if pause and index:
            time.sleep(pause)
        qid = q["id"]
        question = q["question"]
        baseline = q["baseline"]

        anchoring = {"grounding": "general", "from_general_knowledge": False,
                     "row_count": 0, "sources": []}
        timings = {}
        started = time.perf_counter()
        try:
            rag_result = rag.run(question)
            timings = rag_result.get("timings") or {}
            response = rag_result.get("response", "")
            cypher = rag_result.get("cypher", "")
            error = None
            # Where the answer came from, compactly: the judge never sees it,
            # but a reviewer asking "is this anchored?" does.
            anchoring = {
                "grounding": rag_result.get("grounding", "general"),
                "from_general_knowledge": rag_result.get("from_general_knowledge", False),
                "row_count": rag_result.get("row_count", 0),
                "sources": [
                    {"talk_id": s["talk_id"], "title": s["title"], "evidence": s["evidence"]}
                    if s.get("kind") == "talk" else {"event": s["title"], "url": s.get("url", "")}
                    for s in rag_result.get("sources") or []
                ],
            }
        except Exception:
            response = ""
            cypher = ""
            error = traceback.format_exc()
        seconds = round(time.perf_counter() - started, 2)

        if error:
            score, reasoning = 1, f"Exception: {error.splitlines()[-1]}"
        elif not response or response == "N/A":
            score, reasoning = 1, "No response returned"
        else:
            score, reasoning = judge_response(question, baseline, response, judge_model)

        label = SCORE_LABELS.get(score, "unknown")
        short_q = question[:55] + "..." if len(question) > 55 else question
        print(f"Q{qid:<3} {score}/5  {label:<12} {short_q}")
        print(f"     Cypher:   {cypher or '(none)'}")
        print(f"     Answer:   {response or '(none)'}")
        print(f"     Baseline: {baseline}")
        print(f"     Reason:   {reasoning}")
        print(f"     Anchored: {len(anchoring['sources'])} sources / {anchoring['grounding']}"
              + (" / general knowledge" if anchoring["from_general_knowledge"] else ""))
        print(f"     Time:     {seconds:.1f}s  "
              + "  ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
        if error:
            print(f"     ERROR:    {error.splitlines()[-1]}")
        print()

        results.append({
            "id": qid,
            "question": question,
            "baseline": baseline,
            "cypher": cypher,
            "response": response,
            "score": score,
            "label": label,
            "reasoning": reasoning,
            "error": error,
            "judge": judge_model,
            "seconds": seconds,
            "timings": timings,
            **anchoring,
        })

    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    label_counts = {label: 0 for label in SCORE_LABELS.values()}
    for r in results:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    print("=" * 80)
    print(f"SUMMARY — {len(questions)} questions | avg score: {avg:.1f}/5")
    print("-" * 40)
    for score_val in sorted(SCORE_LABELS):
        label = SCORE_LABELS[score_val]
        count = label_counts.get(label, 0)
        print(f"  {score_val} {label:<12} {count:>2}  {'█' * count}")
    times = [r["seconds"] for r in results]
    if times:
        print(f"  time per answer: median {statistics.median(times):.1f}s, "
              f"max {max(times):.1f}s, total {sum(times):.0f}s")
    print("=" * 80)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG against CDKGQA benchmark")
    parser.add_argument("--output", "-o", help="Save detailed results to a JSON file", default=None)
    parser.add_argument("--client", default=None,
                        help="Answer with this clients.baml client instead of the one graphrag.baml names")
    parser.add_argument("--judge-model", default=JUDGE_MODEL,
                        help="Gemini model that scores the answers (compare runs only under the same judge)")
    parser.add_argument("--pause", type=float, default=0,
                        help="Seconds to wait between questions, for a per-minute rate limit")
    args = parser.parse_args()
    if args.client:
        os.environ["RAG_CLIENT"] = args.client
    run_evaluation(output_path=args.output, judge_model=args.judge_model, pause=args.pause)
