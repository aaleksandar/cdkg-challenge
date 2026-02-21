"""
Automated evaluation of the GraphRAG system against the CDKGQA benchmark.

Reads questions and baseline answers from QA/CDKGQA.csv, runs each question
through GraphRAG, and uses an LLM judge to score the response.

Usage:
    uv run evaluate.py
    uv run evaluate.py --output results.json
"""

import argparse
import csv
import json
import os
import traceback
from pathlib import Path

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

import config
from rag import GraphRAG

load_dotenv()
os.environ["BAML_LOG"] = "WARN"

QA_CSV = Path(__file__).parent.parent.parent / "QA" / "CDKGQA.csv"

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


def _get_judge_client():
    global _judge_client
    if _judge_client is None:
        _judge_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _judge_client


def judge_response(question: str, baseline: str, response: str) -> tuple[int, str]:
    """
    Use the LLM to score the RAG response against the baseline answer.
    Returns (score 1-5, reasoning).
    """
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

    client = _get_judge_client()
    result = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    try:
        parsed = json.loads(result.text)
        score = int(parsed["score"])
        reasoning = parsed.get("reasoning", "")
        return score, reasoning
    except (json.JSONDecodeError, KeyError, ValueError):
        for ch in result.text:
            if ch.isdigit() and 1 <= int(ch) <= 5:
                return int(ch), result.text
        return 1, f"Could not parse judge response: {result.text}"


def run_evaluation(output_path: str | None = None) -> list[dict]:
    questions = load_questions(QA_CSV)
    graph_rag = GraphRAG()
    results = []

    print(f"Running evaluation on {len(questions)} questions...\n")
    print(f"{'Q':<4} {'Score':<10} {'Label':<12} Question")
    print("-" * 80)

    for q in questions:
        qid = q["id"]
        question = q["question"]
        baseline = q["baseline"]

        try:
            rag_result = graph_rag.run(question)
            response = rag_result.get("response", "")
            cypher = rag_result.get("cypher", "")
            error = None
        except Exception as e:
            response = ""
            cypher = ""
            error = traceback.format_exc()

        if error:
            score, reasoning = 1, f"Exception: {error.splitlines()[-1]}"
        elif not response or response == "N/A":
            score, reasoning = 1, "No response returned"
        else:
            score, reasoning = judge_response(question, baseline, response)

        label = SCORE_LABELS.get(score, "unknown")
        short_q = question[:55] + "..." if len(question) > 55 else question
        print(f"Q{qid:<3} {score}/5       {label:<12} {short_q}")

        results.append(
            {
                "id": qid,
                "question": question,
                "baseline": baseline,
                "cypher": cypher,
                "response": response,
                "score": score,
                "label": label,
                "reasoning": reasoning,
                "error": error,
            }
        )

    # Summary
    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    label_counts = {label: 0 for label in SCORE_LABELS.values()}
    for r in results:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    print("\n" + "=" * 80)
    print(f"SUMMARY — {len(questions)} questions | avg score: {avg:.1f}/5")
    print("-" * 40)
    for score_val in sorted(SCORE_LABELS):
        label = SCORE_LABELS[score_val]
        count = label_counts.get(label, 0)
        bar = "█" * count
        print(f"  {score_val} {label:<12} {count:>2}  {bar}")
    print("=" * 80)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG against CDKGQA benchmark")
    parser.add_argument("--output", "-o", help="Save detailed results to a JSON file", default=None)
    args = parser.parse_args()

    run_evaluation(output_path=args.output)
