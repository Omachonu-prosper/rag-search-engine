#!/usr/bin/env python3

import argparse
import json


def search(query: str) -> None:
    print(f"Searching for: {query}")
    with open('/home/bknd-bobby/projects/rag-search-engine/data/movies.json') as file:
        movies = json.load(file).get("movies", [])

    results = [movie for movie in movies if query.lower() in movie["title"].lower()][:5]
    results.sort(key=lambda x: x['id'])
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