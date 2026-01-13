#!/usr/bin/env python3

import argparse
import json


# def preprocess_query(query: str) -> str:
#     # Case Insensitive 
#     query = query.lower()

#     # Remove punctuation
#     query = ''.join(char for char in query if char.isalnum() or char.isspace())

#     # Tokenize
#     tokens = query.split()

#     # Remove stop words
#     stop_words = set({
#         "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
#         "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
#         "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
#         "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
#         "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
#         "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
#         "the", "and", "but", "if", "or", "because", "as", "until", "while",
#         "of", "at", "by", "for", "with", "about", "against", "between",
#         "into", "through",  "during","before","after","above","below","to",
#         "from","up","down","in","out","on","off","over","under","again",
#         "further","then","once","here","there","when","where","why","how",
#         "all","any","both","each","few","more","most","other","some",
#         "such","no","nor","not","only","own","same","so","than",
#         "too","very","can","will","just",})
#     tokens = [word for word in tokens if word not in stop_words]

#     # Stemming (simple suffix stripping)
#     suffixes = ("ning", "ly", "ed", "ious", "ies", "ive", "es", "s", "ment", "ing")
#     def stem(word: str) ->  str:
#         for suffix in suffixes:
#             if word.endswith(suffix):
#                 return word[:-len(suffix)]
#         return word

#     tokens = [stem(word) for word in tokens]

#     return tokens


def search(query: str) -> None:
    print(f"Searching for: {query}")
    with open('/home/bknd-bobby/projects/rag-search-engine/data/movies.json') as file:
        movies = json.load(file).get("movies", [])

    # tokens = preprocess_query(query)
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