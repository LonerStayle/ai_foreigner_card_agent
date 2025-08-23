import torch
import os
from qdrant_client import QdrantClient
class Common:
    device = "mps" if torch.mps.is_available() else "cpu"
    CHROME_DB_PATH = "chroma_db"
    
    QDRANT_URL = "http://localhost:6333"
    COLLECTION_NAME = "foreign_guide"
    
    BASE_DIR = os.path.dirname(os.path.dirname(__file__)) + '/ai_foreigner_card_agent/'
    DATA_DIR = os.path.join(BASE_DIR, "src/data")

    def get_qdrant():
        return QdrantClient(url=Common.QDRANT_URL)
    