from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, download_loader, RAKEKeywordTableIndex, StorageContext
from llama_index.core.embeddings import resolve_embed_model

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

index.storage_context.persist()
