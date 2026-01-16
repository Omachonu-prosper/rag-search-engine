#!/usr/bin/env python3

import argparse
import json
import sys
import math
from utils import (
    preprocess_text,
    InvertedIndex,
    get_stop_words
)


def search(query: str) -> None:
    print(f"Searching for: {query}")

    try:
        index = InvertedIndex()
        index.load()
    except FileNotFoundError:
        print("Required index files not found! Build the index before searching...")
        sys.exit(1)
    
    query_tokens = preprocess_text(query, get_stop_words())
    doc_ids = []
    for token in query_tokens:
        token_doc_ids = index.get_documents(token)
        for id in token_doc_ids:
            if len(doc_ids) < 5:
                doc_ids.append(id)
    
    for id in doc_ids:
        doc = index.docmap[id]
        print(f"{doc['id']} - {doc['title']}")
    

def build() -> None:
    with open('/home/bknd-bobby/projects/rag-search-engine/data/movies.json') as file:
        movies = json.load(file).get("movies", [])
    
    index = InvertedIndex()
    index.build(movies)
    index.save()


def tf(doc_id, term) -> None:
    index = InvertedIndex()
    index.load()
    _tf = index.get_tf(doc_id, term)
    return _tf


def idf(term) -> None:
    index = InvertedIndex()
    index.load()
    docs = index.get_documents(term)
    _idf = math.log((len(index.docmap) + 1) / (len(docs) + 1))
    return _idf


def tfidf(doc_id, term) -> None:
    index = InvertedIndex()
    index.load()
    tf_idf = tf(doc_id, term) * idf(term)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def bm25_idf_command(term):
    index = InvertedIndex()
    index.load()
    bm25_idf = index.get_bm25_idf(term)
    return float(bm25_idf)
    

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index and save to disk")

    tf_parser = subparsers.add_parser("tf", help="Check the frequency of a search term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser("idf", help="Check the inverse document frequency of a search term")
    idf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Check the frequency and uniqueness of a search term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query)
        case "build":
            build()
        case "tf":
            tf(args.doc_id, args.term)
        case "idf":
            idf(args.term)
        case 'tfidf':
            tfidf(args.doc_id, args.term)
        case 'bm25idf':
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()