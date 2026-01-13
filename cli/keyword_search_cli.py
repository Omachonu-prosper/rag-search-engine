#!/usr/bin/env python3

import argparse
import json
import string


def preprocess_text(text: str) -> list[str]:
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize text
    tokens = text.split()
    
    return tokens


def search(query: str) -> None:
    print(f"Searching for: {query}")
    with open('/home/bknd-bobby/projects/rag-search-engine/data/movies.json') as file:
        movies = json.load(file).get("movies", [])

    results = []
    for movie in movies:
        query_tokens = preprocess_text(query)
        title_tokens = preprocess_text(movie['title'])

        if any(token in title_token for title_token  in title_tokens for token in query_tokens):
            results.append(movie)
    
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