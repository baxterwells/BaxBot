import chromadb
import ollama
import pathlib
from rich.console import Console
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import Any, List, Optional

memoryManagerTag = "[bold white][ MemoryManager]:[/bold white]"

class OllamaEmbeddingFunction(EmbeddingFunction):
    console = Console()
    """A custom embedding function that follows the ChromaDB protocol."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = ollama.embed(model=self.model_name, input=input)
        return response.embeddings

    @staticmethod
    def name() -> str:
        return "ollama_embedding_function"

    def get_config(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(model_name=config["model_name"])

class PersonalAgentMemory:
    console = Console()
    def __init__(self, db_path: str = "./agent_memory", model_name: str = "nomic-embed-text"):
        self.db_path = db_path
        self.model_name = model_name
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_fn = OllamaEmbeddingFunction(model_name=self.model_name)
        
        self.collection = self.client.get_or_create_collection(
            name="personal_memory",
            embedding_function=self.embedding_fn
        )

    def add_memory(self, category: str, key: str, content: str):
        """
        The standard way for BaxBot to save distilled conversation summaries.
        'key' should be unique (e.g., a timestamp or a specific topic).
        """
        doc_id = f"{category}_{key}"
        self.collection.upsert(
            documents=[content],
            metadatas=[{"type": category}],
            ids=[doc_id]
        )
        # self.console.print(f"{memoryManagerTag} Saved: {doc_id} (Category: {category})")
        self.console.print(f"{memoryManagerTag} Saved: {doc_id}")

    def add_document(self, text: str, category: str, doc_id: str):
        """Low-level: Uses UPSERT for manual file/document ingestion."""
        self.collection.upsert(
            documents=[text],
            metadatas=[{"type": category}],
            ids=[doc_id]
        )
        self.console.print(f"{memoryManagerTag} Embedded Document: {doc_id} (Category: {category})")

    def update_entry(self, category: str, key: str, text: str):
        """Agent-friendly surgical update for specific facts."""
        doc_id = f"{category}_{key}"
        self.collection.upsert(
            documents=[text],
            metadatas=[{"type": category}],
            ids=[doc_id]
        )
        self.console.print(f"{memoryManagerTag} Updated: {doc_id} (Category: {category})")

    def ingest_vault(self, vault_path: str):
        """Scans a directory and all subdirectories recursively to sync with the database."""
        root = pathlib.Path(vault_path)
        if not root.exists():
            return f"Error: Vault path '{vault_path}' not found."

        self.console.print(f"--- Starting Recursive Vault Sync: {root.absolute()} ---")

        # .rglob("*.txt") finds every .txt file in every subfolder automatically
        for file_path in root.rglob("*.txt"):
            try:
                # 1. Get the path relative to the root (e.g., 'faith/holy_spirit_zoom/note1.txt')
                relative_path = file_path.relative_to(root)
                
                # 2. Create the doc_id
                # We take the relative path, remove the suffix (.txt), 
                # then turn the path parts into a single string joined by underscores.
                # This is much safer and avoids the 'str has no attribute stem' error.
                path_no_ext = relative_path.with_suffix('')
                doc_id = "_".join(path_no_ext.parts)

                # s 3. Determine the category (the first folder in the relative path)
                category_name = relative_path.parts[0]

                # 4. Read and ingest the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.add_document(content, category_name, doc_id)
                        # print(f"[Ingested] {doc_id}") # Uncomment for verbose logging

            except Exception as e:
                self.console.print(f"Error processing {file_path}: {e}")

        return "Recursive Vault sync complete."

    def query_memory(self, query_text: str, category: str, n_results: int = 2) -> List[str]:
        """Queries the memory filtered by category. Returns a list of strings."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"type": category}
        )
        
        # NEW: Defensive check. If Chroma returns nothing, return an empty list 
        # to prevent BaxBot from crashing when trying to join None/Empty.
        if results and results['documents'] and len(results['documents']) > 0:
            return results['documents'][0]
        self.console.print(f"\t{memoryManagerTag} Take a look (cmd+f) at this error, Bax. Returned 0 documents.")
        return []

    def list_all_memories(self):
        """Prints every single piece of memory currently in the database."""
        self.console.print("\n--- BEGIN MEMORY INSPECTION ---")
        # .get() retrieves the actual human-readable text and metadata
        results = self.collection.get()
        
        for i in range(len(results['ids'])):
            doc_id = results['ids'][i]
            content = results['documents'][i]
            metadata = results['metadatas'][i]
            self.console.print(f"ID: {doc_id} | Type: {metadata['type']} | Content: {content[:100]}...") # Print first 100 chars
        self.console.print("--- END MEMORY INSPECTION ---\n")

    def delete_memory(self, category: str, key: str):
        """Deletes a specific memory entry by its ID."""
        doc_id = f"{category}_{key}"
        self.collection.delete(ids=[doc_id])
        self.console.print(f"\t{memoryManagerTag} Deleted: {doc_id}")



# --- EXECUTION ---
if __name__ == "__main__":
    memory = PersonalAgentMemory()
    # Test Manual Update
    memory.update_entry("info", "bio", "I am a developer who loves Apple Silicon.")
    # Test the new add_memory method
    memory.add_memory("info", "session_1", "User likes dark mode and Python.")
    # Test Query
    memory.console.print("Test Query Result:", memory.query_memory("What does the user like?", "info"))
    
    # Force ingest
    memory.ingest_vault("./memory_vault")
