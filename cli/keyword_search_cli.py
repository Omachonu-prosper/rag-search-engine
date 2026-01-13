#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer


STEMMER = PorterStemmer()


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


def search(query: str) -> None:
    print(f"Searching for: {query}")
    with open('/home/bknd-bobby/projects/rag-search-engine/data/movies.json') as file:
        movies = json.load(file).get("movies", [])

    with open('/home/bknd-bobby/projects/rag-search-engine/data/stopwords.txt') as file:
        stop_words = file.read().splitlines()

    results = []
    query_tokens = set(preprocess_text(query, stop_words))
    for movie in movies:
        title_tokens = set(preprocess_text(movie['title'], stop_words))
        matched = False # Protect against multiple matching tokens for same movie

        for query_token in query_tokens:
            for title_token in title_tokens:
                if query_token in title_token and movie and not matched:
                    results.append(movie)
                    matched = True
    
    results = results[:5]
    for index, movie in enumerate(results, start=1):
        print(f"{index}. {movie['title']}")
    

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            search(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()