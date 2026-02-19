import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    CACHE_DIR = Path('/home/bknd-bobby/projects/rag-search-engine/cache')
    DATA_DIR = Path('/home/bknd-bobby/projects/rag-search-engine/data')
    EMBEDDINGS_FILE = CACHE_DIR / "movie_embeddings.npy"
    MOVIES_FILE = DATA_DIR / "movies.json"


    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.documents_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        movies_list = []
        for document in documents:
            self.documents_map[document['id']] = document
            movies_list.append(f"{document['title']}: {document['description']}")
        
        self.embeddings = self.model.encode(movies_list, show_progress_bar=True)
        np.save(self.EMBEDDINGS_FILE, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.documents_map[document['id']] = document
        
        if self.EMBEDDINGS_FILE.exists():
            self.embeddings = np.load(self.EMBEDDINGS_FILE)
        
        if self.embeddings is not None and len(self.embeddings) == len(self.documents):
            print("Embeddings loaded from file")
            return self.embeddings
        print("Building embeddings")
        return self.build_embeddings(documents)

    def generate_embeddings(self, text: str):
        if not text or not isinstance(text, str) or text.isspace():
            raise ValueError("Text must be a valid string and not just whitespace")
        embedding = self.model.encode([text.strip()])
        return embedding[0]
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        results = []
        query_embeddings = self.generate_embeddings(query)
        for index, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(doc_embedding, query_embeddings)
            results.append((similarity, self.documents[index]))
        results.sort(key=lambda doc: doc[0], reverse=True)
        
        top_results = []
        for result in results[:limit]:
            score = result[0]
            document = result[1]
            top_results.append({
                'score': score,
                'title': document['title'],
                'description': document['description']
            })
        return top_results
        

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embeddings(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    with open(semantic_search.MOVIES_FILE, 'r') as file:
        documents = json.load(file).get("movies", [])
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embeddings(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)