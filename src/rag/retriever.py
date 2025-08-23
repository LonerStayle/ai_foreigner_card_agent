import os, json, heapq
from Common import Common
from src.pipelines.index import get_embeddings
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


PARSED_FILE = os.path.join(Common.DATA_DIR, "parsed.jsonl")

def load_bm25_corpus():
    docs = [json.loads(line) for line in open(PARSED_FILE, "r", encoding="utf-8")]
    corpus = [d["text"].split() for d in docs]
    bm25 = BM25Okapi(corpus)
    return bm25, docs

def normalize(scores):
    min_s, max_s = min(scores), max(scores)
    return [(s - min_s) / (max_s - min_s + 1e-9) for s in scores]

def hybrid_search(query, top_k = 50):
    bm25_model, bm25_docs = load_bm25_corpus()
    bm25_scores = bm25_model.get_scores(query.split())
    bm25_results = [
        (bm25_docs[i], float(score))
        for i,score in enumerate(bm25_scores)
    ]
    bm25_results = heapq.nlargest(top_k, bm25_results, key = lambda x: x[1])
    query = get_embeddings(query)
    client = Common.get_qdrant()
    query_result = client.search(
        collection_name = Common.COLLECTION_NAME,
        query_vector = query,
        limit = top_k
    )

    bm25_norm = normalize([s for _, s in bm25_results])    
    vector_norm = normalize([q.score for q in query_result])
    
    bm25_weight = 0.3
    merged = []
    for (doc, bm25_score), q, b, v in zip(bm25_results, query_result, bm25_norm, vector_norm):
        merged.append({
            "text" : doc["text"],
            "source" : doc["source"],
            "page": doc["page"],
            "bm25_score": bm25_score,
            "vector_score": q.score,
            "combined": bm25_weight * b + (1 - bm25_weight) * v
        })

    return heapq.nlargest(top_k, merged, key = lambda x: x["combined"])
