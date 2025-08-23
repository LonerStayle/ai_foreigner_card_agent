import json, os
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness
)
from .retriever import hybrid_search
from .reranker import rerank
from .answer import generate_answer
from Common import Common


def build_eval_dataset():
    
    json_path = os.path.join(Common.DATA_DIR, "eval_dataset_127.json")
    
    with open(json_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)
    questions = eval_dataset["EVAL_QUESTIONS"]
    answers = eval_dataset["GT_MAP"]
    
    
    dataset=[]
    for q in questions:
        candidates = hybrid_search(q, top_k=50)
        top_docs = rerank(q, candidates, top_k=8)
        answer, _ = generate_answer(q)
        gt = json.dumps(answers.get(q, {}), ensure_ascii=False)
        dataset.append({
            "user_input":q,
            "answer":answer.content,
            "contexts":[d["text"] for d in top_docs],
            "ground_truth":gt
        })
    
    
    return Dataset.from_list(dataset)

def run_evel():
    dataset = build_eval_dataset()
    metrics = [context_precision, context_recall, answer_relevancy, faithfulness]
    results = evaluate(dataset, metrics)
    df = results.to_pandas()
    print("=== RAG Evaluation Results ===")
    datetime = datetime.now().strftime("%m-%d_%H-%M-%S")
    df.to_csv(f"eval_result_{datetime}.csv", index=False)
    print("📦 Saved → rag_eval_results.csv")
    return df
