from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.groq import Groq

import sys

query = sys.argv[1]

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

storage_context = StorageContext.from_defaults(persist_dir="storage")

Settings.llm = Groq(model="llama3-70b-8192", temperature=0.8, context_window=8192)

index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

response = query_engine.query(query)

print(response)
