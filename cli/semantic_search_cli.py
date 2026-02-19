#!/usr/bin/env python3

import argparse, json
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    SemanticSearch
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify model parser
    subparsers.add_parser("verify", help="Verify the embedding model.")

    # Embed text parser
    embed_text_parser = subparsers.add_parser("embed_text", help="Embed the provided text input")
    embed_text_parser.add_argument("text", type=str, help="The text input to be embedded")

    # Verify embeddings parser
    subparsers.add_parser("verify_embeddings", help="Verify that the embeddings are loaded properly.")

    # Embed query text parser
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed the provided query text")
    embed_query_parser.add_argument("query", type=str, help="The query text to embed")

    # Search parser
    search_parser = subparsers.add_parser("search", help="Search for a movie")
    search_parser.add_argument("query", type=str, help="The search term")
    search_parser.add_argument("--limit", type=int, default=5, required=False, help="The number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search = SemanticSearch()
            with open(semantic_search.MOVIES_FILE, 'r') as file:
                documents = json.load(file).get("movies", [])
            semantic_search.load_or_create_embeddings(documents)
            results = semantic_search.search(args.query, args.limit)
            for index, result in enumerate(results, start=1):
                print(round(int(result['score'])))
                print(f"{index}. {result['title']} (score: {round(result['score'], 4)}) \n{result['description'][:100]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()