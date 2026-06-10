import json, os, sys, time, statistics

# script-relative import so this runs correctly from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

from ground_truth_foucault import QA_PAIRS
import environment

if len(sys.argv) < 2:
    print("Usage: python eval/run_eval.py <condition>")
    print("  e.g. recursive_full | semantic_full | semantic_q4 | recursive_q4")
    sys.exit(1)

CONDITION = sys.argv[1]

JUDGE_PROMPT = """Given the following, reply with JSON only — no extra text, no markdown fences, no thinking.

Question: {query}
Reference answer: {reference}
Retrieved context: {context}
Model answer: {answer}

{{"accurate": true or false, "hallucinated": true or false, "reason": "one sentence"}}"""


def strip_think(text):
    """Remove <think>...</think> blocks that qwen3 sometimes emits."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run_judge(query, reference, context, answer):
    prompt = JUDGE_PROMPT.format(
        query=query, reference=reference, context=context, answer=answer
    )
    response = environment.model.invoke([("human", prompt)])
    raw = strip_think(response.content)
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"accurate": None, "hallucinated": None, "reason": f"parse error: {raw[:120]}"}


results = []
for qa in QA_PAIRS:
    # retrieval — k=5 for Hit@5
    t0 = time.time()
    hits = environment.vector_store.similarity_search_with_score(qa["query"], k=5)
    retrieval_latency = time.time() - t0
    context = "\n\n".join(doc.page_content for doc, _ in hits)

    # generation
    messages = [
        ("system", (
            "Answer using only the context below. "
            "If the context does not contain the answer, say you don't know.\n\n"
            + context
        )),
        ("human", qa["query"]),
    ]
    t1 = time.time()
    response = environment.model.invoke(messages)
    generation_latency = time.time() - t1
    answer_text = strip_think(response.content)

    # token counts and throughput from Ollama metadata
    meta = response.response_metadata
    input_tokens = meta.get("prompt_eval_count", 0)
    output_tokens = meta.get("eval_count", 0)
    eval_duration_s = meta.get("eval_duration", 1) / 1e9
    tokens_per_sec = output_tokens / eval_duration_s if eval_duration_s > 0 else 0

    verdict = run_judge(qa["query"], qa["reference"], context, answer_text)

    results.append({
        "query": qa["query"],
        "answer": answer_text,
        "retrieval_latency_s": round(retrieval_latency, 3),
        "generation_latency_s": round(generation_latency, 3),
        "total_latency_s": round(retrieval_latency + generation_latency, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "avg_similarity_score": round(statistics.mean(s for _, s in hits), 4),
        "sections_retrieved": [doc.metadata.get("section") for doc, _ in hits],
        "accurate": verdict["accurate"],
        "hallucinated": verdict["hallucinated"],
        "reason": verdict["reason"],
    })
    print(f"[{CONDITION}] {qa['query'][:55]} — accurate={verdict['accurate']} hallucinated={verdict['hallucinated']}")

# summary
valid = [r for r in results if r["accurate"] is not None]
n = len(results)
n_valid = len(valid)

hit5 = sum(1 for r in valid if r["accurate"] is True)
hallucinated = sum(1 for r in valid if r["hallucinated"] is True)

print(f"\n=== {CONDITION} summary ({n} queries) ===")
print(f"Hit@5 (accurate):   {hit5}/{n_valid} = {hit5/n_valid*100:.1f}%  (target ≥ 85%)")
print(f"Hallucination rate: {hallucinated}/{n_valid} = {hallucinated/n_valid*100:.1f}%  (target ≤ 15%)")
print(f"Avg gen latency:    {statistics.mean(r['generation_latency_s'] for r in results):.2f}s")
print(f"Avg tokens/sec:     {statistics.mean(r['tokens_per_sec'] for r in results):.1f}")
print(f"Avg input tokens:   {statistics.mean(r['input_tokens'] for r in results):.0f}")
print(f"Avg output tokens:  {statistics.mean(r['output_tokens'] for r in results):.0f}")
print(f"Avg similarity:     {statistics.mean(r['avg_similarity_score'] for r in results):.4f}")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{CONDITION}.json")
with open(out_path, "w") as f:
    json.dump({"condition": CONDITION, "summary": {
        "n_queries": n,
        "hit5_pct": round(hit5/n_valid*100, 1),
        "hallucination_pct": round(hallucinated/n_valid*100, 1),
        "avg_gen_latency_s": round(statistics.mean(r['generation_latency_s'] for r in results), 3),
        "avg_tokens_per_sec": round(statistics.mean(r['tokens_per_sec'] for r in results), 1),
        "avg_input_tokens": round(statistics.mean(r['input_tokens'] for r in results)),
        "avg_output_tokens": round(statistics.mean(r['output_tokens'] for r in results)),
        "avg_similarity": round(statistics.mean(r['avg_similarity_score'] for r in results), 4),
    }, "results": results}, f, indent=2)
print(f"\nSaved to {out_path}")
