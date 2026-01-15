import string
import pickle
from pathlib import Path
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
    stopwords = get_stop_words()

    def __add_document(self, doc_id, text):
        tokens = preprocess_text(text, self.stopwords)
        for token in tokens:
            if self.index.get(token):
                self.index[token].add(doc_id)
            else:
                self.index[token] = {doc_id}

    def get_documents(self, term):
        term = term.lower()
        doc_ids = self.index[term]
        return sorted(list(doc_ids))

    def build(self, movies):
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie

    def save(self):
        cache_dir = Path('/home/bknd-bobby/projects/rag-search-engine/cache')
        index_file_path = cache_dir / 'index.pkl'
        docmap_file_path = cache_dir / 'docmap.pkl'
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(index_file_path, 'wb') as index_file:
            pickle.dump(self.index, index_file)

        with open(docmap_file_path, 'wb') as docmap_file:
            pickle.dump(self.docmap, docmap_file)

