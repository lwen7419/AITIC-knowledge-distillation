import json, os, sys, time, statistics

#find absolute file path to current file (run_eval), strips file name to leave just file
#  path up until current file, goes up one level then into src, and add path to list of 
# modules searched
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

import environment
from RAG_chain import ask

#if passed three command line arguments and script itself such that length of command
# line is equal or larger than four
#if smaller, pass error statement that exemplifies how to run the eval file
if len(sys.argv) < 4:
    print("Usage: python eval/run_eval.py <condition> <ground_truth_module> <chunker>")
    print("  chunker: 'recursive' or 'heading'")
    print("  e.g. python eval/run_eval.py recursive_foucault ground_truth_foucault recursive")
    print("       python eval/run_eval.py heading_foucault ground_truth_foucault heading")
    print("       python eval/run_eval.py recursive_biology ground_truth_biology recursive")
    #stops Python program, 1 in argument signals to shell that error occurred
    #whereas 0 in argument means stop program but everything went successful
    # just quit early
    sys.exit(1)

# condition name is used to label the results JSON file e.g. "recursive_foucault"
CONDITION = sys.argv[1]

# dynamically load whichever ground truth module was passed as the second argument
# and extract its QA_PAIRS list of {query, reference} dicts
import importlib
ground_truth_module = importlib.import_module(sys.argv[2])
QA_PAIRS = ground_truth_module.QA_PAIRS

# re-index the PDF using the chosen chunker before running queries
CHUNKER = sys.argv[3]
from Indexing import build_index
build_index(ground_truth_module.PDF_PATH, CHUNKER)

# prompt sent to the LLM judge to evaluate each answer
# judge returns JSON with accurate, hallucinated, and reason fields
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
    # build the judge prompt with the actual query, reference, context, and model answer
    prompt = JUDGE_PROMPT.format(
        query=query, reference=reference, context=context, answer=answer
    )
    response = environment.model.invoke([("human", prompt)])
    # strip any <think> blocks and markdown fences before parsing JSON
    raw = strip_think(response.content)
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # if the model didn't return valid JSON, record the parse failure
        return {"accurate": None, "hallucinated": None, "reason": f"parse error: {raw[:120]}"}


results = []
for qa in QA_PAIRS:
    # time the full ask() call to measure generation latency
    t0 = time.time()
    cited_response, llm_response, hits = ask(qa["query"])
    generation_latency = time.time() - t0
    retrieval_latency = 0  # retrieval happens inside ask; total is generation_latency

    # join retrieved chunks into a single string for the judge to evaluate
    context = "\n\n".join(doc.page_content for doc, _ in hits)
    answer_text = strip_think(llm_response.content)

    # extract token counts and throughput from Ollama's response metadata
    meta = llm_response.response_metadata
    input_tokens = meta.get("prompt_eval_count", 0)
    output_tokens = meta.get("eval_count", 0)
    eval_duration_s = meta.get("eval_duration", 1) / 1e9  # convert nanoseconds to seconds
    tokens_per_sec = output_tokens / eval_duration_s if eval_duration_s > 0 else 0

    # ask the LLM judge whether the answer is accurate and whether it hallucinated
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
        # average ChromaDB similarity score across the top k retrieved chunks
        "avg_similarity_score": round(statistics.mean(s for _, s in hits), 4),
        # which document sections were retrieved for this query
        "sections_retrieved": [doc.metadata.get("section") for doc, _ in hits],
        "accurate": verdict["accurate"],
        "hallucinated": verdict["hallucinated"],
        "reason": verdict["reason"],
    })
    print(f"[{CONDITION}] {qa['query'][:55]} — accurate={verdict['accurate']} hallucinated={verdict['hallucinated']}")

# compute summary metrics across all queries
valid = [r for r in results if r["accurate"] is not None]
n = len(results)
n_valid = len(valid)

# Hit@5: fraction of queries where the model gave an accurate answer
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

# save full results and summary to a JSON file named after the ablation condition
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
