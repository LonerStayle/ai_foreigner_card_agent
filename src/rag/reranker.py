from FlagEmbedding import FlagReranker
import heapq

reranker = FlagReranker("BAAI/bge-reranker-large", use_fp16=True)

def rerank(query, candidates, top_k=8):
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.compute_score(pairs)
    reranked = [{**c, "rerank_score": s} for c, s in zip(candidates, scores)]
    return heapq.nlargest(top_k, reranked, key=lambda x: x["rerank_score"])
