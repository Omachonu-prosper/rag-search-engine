import string
import pickle
from pathlib import Path
from collections import Counter
from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()


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
    index = {}
    docmap = {}
    term_frequencies = {}

    stopwords = get_stop_words()
    cache_dir = Path('/home/bknd-bobby/projects/rag-search-engine/cache')
    index_file_path = cache_dir / 'index.pkl'
    docmap_file_path = cache_dir / 'docmap.pkl'
    term_frequencies_file_path = cache_dir / 'term_frequencies.pkl'

    def __add_document(self, doc_id, text):
        tokens = preprocess_text(text, self.stopwords)
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

    def get_documents(self, term):
        term = term.lower()
        doc_ids = self.index.get(term, {})
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id, term):
        term = term.lower()
        doc_term_frequencies = self.term_frequencies.get(doc_id, Counter())
        return doc_term_frequencies.get(term, 0)

    def build(self, movies):
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie

    def save(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.__save_file_operation(self.index_file_path, self.index)
        self.__save_file_operation(self.docmap_file_path, self.docmap)
        self.__save_file_operation(self.term_frequencies_file_path, self.term_frequencies)

    def load(self):
        self.index = self.__load_file_operation(self.index_file_path)
        self.docmap = self.__load_file_operation(self.docmap_file_path)
        self.term_frequencies = self.__load_file_operation(self.term_frequencies_file_path)