import os, uuid, json, logging
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from FlagEmbedding import BGEM3FlagModel
from Common import Common

PARSED_FILE = os.path.join(Common.DATA_DIR, "parsed.jsonl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 16비트로 메모리 절약
embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# Dict[Literal["dense_vecs", "lexical_weights", "colbert_vecs"], Union[np.ndarray, List[Dict[str, float]], List[np.ndarray]]
def get_embeddings(texts):
    return embedder.encode(texts)["dense_vecs"]

def run_indexing(batch_size=32):
    if not os.path.exists(PARSED_FILE):
        logger.error(f"Parsed file not found: {PARSED_FILE}")
        return
    
    sample_vector = get_embeddings(["dimension check"])[0]
    vector_size = len(sample_vector)
    
    client = QdrantClient(url=Common.QDRANT_URL)
    client.recreate_collection(
        collection_name=Common.COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size,distance=Distance.COSINE)
    )
    
    logger.info(f"Collection ready: {Common.COLLECTION_NAME}")
    
    docs = [json.loads(line) for line in open(PARSED_FILE,"r", encoding="utf-8")]
    logger.info(f"Loaded {len(docs)} docs")
    
    for i in tqdm(range(0, len(docs), batch_size), desc="Indexing"):
        chunk = docs[i:i+batch_size]
        texts = [d["text"] for d in chunk]
        vectors = get_embeddings(texts)
        
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=v, payload=d)
            for d, v in zip(chunk, vectors)
        ]
        
        try: 
            client.upsert(collection_name=Common.COLLECTION_NAME, points=points)
        except Exception as e:
            logger.error(f"Failed batch {i}: {e}")
            continue
        
    total = client.count(collection_name=Common.COLLECTION_NAME).count
    logger.info(f"Indexing complete. Total vectors: {total}")