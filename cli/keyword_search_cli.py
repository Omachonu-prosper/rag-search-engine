#!/usr/bin/env python3

import argparse
import json
from utils import (
    preprocess_text,
    InvertedIndex,
)


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


def build() -> None:
    with open('/home/bknd-bobby/projects/rag-search-engine/data/movies.json') as file:
        movies = json.load(file).get("movies", [])
    
    index = InvertedIndex()
    index.build(movies)
    index.save()
    merida_ids = index.get_documents('merida')
    print(merida_ids[0])
    print(index.docmap[merida_ids[0]])
        

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build the inverted index and save to disk")

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query)
        case "build":
            build()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()