import string
import pickle
import math
from pathlib import Path
from collections import Counter
from constants import STEMMER, BM25_K1, BM25_B

def get_stop_words() -> list[str]:
    with open('/home/bknd-bobby/projects/rag-search-engine/data/stopwords.txt') as file:
        return file.read().splitlines()


def preprocess_text(text: str, stop_words: list[str]) -> list[str]:
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize text
    tokens = text.split()

    # Remove stop words and stem tokens
    stop_words = stop_words
    tokens = [STEMMER.stem(token) for token in tokens if token not in stop_words]
    
    return tokens


class InvertedIndex:
    stopwords = get_stop_words()
    cache_dir = Path('/home/bknd-bobby/projects/rag-search-engine/cache')
    index_file_path = cache_dir / 'index.pkl'
    docmap_file_path = cache_dir / 'docmap.pkl'
    term_frequencies_file_path = cache_dir / 'term_frequencies.pkl'
    doc_lengths_file_path = cache_dir / "doc_lengths.pkl"

    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.doc_lengths = Counter()
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        tokens = preprocess_text(text, self.stopwords)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if self.index.get(token):
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

            if self.term_frequencies.get(doc_id):
                self.term_frequencies[doc_id].update([token])
            else:
                self.term_frequencies[doc_id] = Counter([token])

    def __save_file_operation(self, file_path, object):
        with open(file_path, 'wb') as file:
            pickle.dump(object, file)

    def __load_file_operation(self, file_path):
        with open(file_path, 'rb') as file:
            object = pickle.load(file)
        return object
    
    def __get_avg_doc_length(self):
        doc_count = len(self.doc_lengths)
        if doc_count == 0:
            return 0
        
        return self.doc_lengths.total() / doc_count
    
    def get_documents(self, term):
        term = STEMMER.stem(term.lower())
        doc_ids = self.index.get(term, {})
        return sorted(list(doc_ids))

    def get_tf(self, doc_id, term):
        term = STEMMER.stem(term.lower())
        doc_term_frequencies = self.term_frequencies.get(doc_id, Counter())
        return doc_term_frequencies.get(term, 0)
    
    def get_bm25_idf(self, term):
        N = len(self.docmap)
        df = len(self.get_documents(term))
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        length_norm = 1 - b + b * (doc_length / self.__get_avg_doc_length())
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf
    
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        query_tokens = preprocess_text(query, self.stopwords)
        scores = Counter()
        for doc_id in self.docmap.keys():
            for token in query_tokens:
                scores[doc_id] += self.bm25(doc_id, token)
        return scores.most_common(limit)
    
    def build(self, movies):
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie

    def save(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.__save_file_operation(self.index_file_path, self.index)
        self.__save_file_operation(self.docmap_file_path, self.docmap)
        self.__save_file_operation(self.term_frequencies_file_path, self.term_frequencies)
        self.__save_file_operation(self.doc_lengths_file_path, self.doc_lengths)

    def load(self):
        self.index = self.__load_file_operation(self.index_file_path)
        self.docmap = self.__load_file_operation(self.docmap_file_path)
        self.term_frequencies = self.__load_file_operation(self.term_frequencies_file_path)
        self.doc_lengths = self.__load_file_operation(self.doc_lengths_file_path)