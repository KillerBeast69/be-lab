import re                          # provides regex for tokenizing (stripping punctuation)

# Small document collection for demonstration (each string = one document)
documents = [
    "the quick brown fox jumps over the lazy dog",
    "the dog sat next to the lazy fox",
    "the quick cat chased the brown mouse"
]

# Build the inverted index: maps each unique word → set of document IDs that contain it
index = {}
for doc_id, text in enumerate(documents):
    # Tokenize: lowercase, extract only word characters (ignores punctuation)
    words = re.findall(r'\b\w+\b', text.lower())
    # Use a set to keep each word only once per document (saves space)
    for word in set(words):
        if word not in index: index[word] = set()  # create the set on first use
        index[word].add(doc_id)                    # add this document ID to the word's list

# Print the full inverted index for inspection
for word in sorted(index.keys()):
    print(f"  {word:10} -> {sorted(index[word])}")

# Search function: finds documents containing ALL query terms (AND logic)
def search(query):
    # Tokenize the query the same way as the documents
    terms = re.findall(r'\b\w+\b', query.lower())
    if not terms: return []
    # Gather the document-set for each query term from the index (empty set if word not found)
    doc_sets = [index.get(w, set()) for w in terms]
    # Intersection (= AND): keep only documents present in EVERY term's set
    result = set.intersection(*doc_sets) if all(doc_sets) else set()
    return sorted(result)

# Run a sample query and print matching documents
query = "lazy fox"
doc_ids = search(query)
print(f"\nQuery: '{query}'")
print(f"  Matching documents: {doc_ids}")
for d in doc_ids:
    print(f"    [{d}] {documents[d]}")