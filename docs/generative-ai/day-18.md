--- 
sidebar_position: 19
id: day-18
title: 'Day 18: Retrieval-Augmented Generation (RAG) - Grounding LLMs in Reality'
---

## Day 18: Retrieval-Augmented Generation (RAG) - Grounding LLMs in Reality

### Objective

Understand the RAG architecture, which enhances LLMs by allowing them to "look up" information from an external knowledge base before generating a response, reducing hallucinations and enabling knowledge updates.

### Core Concepts

*   **The Problem with LLMs:**
    *   **Hallucination:** LLMs can "make up" facts, mixing real information with plausible-sounding but incorrect details.
    *   **Stale Knowledge:** An LLM's knowledge is frozen at the time of its training. It knows nothing about events that have happened since.
    *   **Lack of Transparency:** It's often impossible to know *why* an LLM gave a certain answer or where its information came from.
    *   **Inability to Use Private Data:** You can't ask an LLM about your company's internal documents because it was never trained on them.

*   **RAG: The Big Idea:**
    *   Don't rely solely on the LLM's internal (parametric) memory.
    *   When a query comes in, first use it to **retrieve** relevant documents from an external, up-to-date knowledge base.
    *   Then, **augment** the original prompt with the content of these retrieved documents.
    *   Finally, feed this augmented prompt to the LLM to **generate** a response.
    *   **In short:** RAG = Retrieval + Augmented Generation.

*   **The RAG Architecture:**
    1.  **Indexing (Offline Process):**
        *   Take your source documents (e.g., PDFs, web pages, text files).
        *   Split them into manageable chunks (e.g., paragraphs).
        *   Use an **Embedding Model** (like a pre-trained BERT or CLIP) to convert each chunk into a vector embedding.
        *   Store these embeddings in a specialized **Vector Database** (e.g., FAISS, Pinecone, ChromaDB).
    2.  **Retrieval & Generation (Online Process):**
        *   A user submits a query.
        *   The query is converted into a vector embedding using the *same* embedding model.
        *   This query vector is used to search the vector database for the `k` most similar document chunk embeddings (using cosine similarity).
        *   The text of these top-`k` chunks is retrieved.
        *   A new prompt is constructed: "Based on this context: [retrieved text chunks], answer this question: [original query]"
        *   This augmented prompt is sent to the LLM, which generates a final answer grounded in the provided context.

### 🧠 Math & Stats Focus: Vector Search

The core of RAG is efficient vector search.

*   **Dense Vectors:** The embeddings for the document chunks and the query are "dense" vectors (most values are non-zero), typically with hundreds of dimensions.
*   **Nearest Neighbor (NN) Search:** The goal is to find the vectors in the database that are closest to the query vector.
*   **The Challenge of Scale:** Calculating the cosine similarity between the query vector and *every single one* of the millions of document vectors in the database is computationally infeasible for real-time applications. This is called an exhaustive search.
*   **Approximate Nearest Neighbor (ANN):** Vector databases use ANN algorithms to solve this. These algorithms trade a tiny bit of accuracy for a massive speedup. They find vectors that are *probably* the nearest neighbors without having to check every single one.
    *   **Common ANN Techniques:**
        *   **Hashing-based (LSH):** Groups similar vectors into buckets using hash functions.
        *   **Tree-based (Annoy):** Recursively partitions the vector space.
        *   **Graph-based (HNSW):** The current state-of-the-art. It builds a graph where similar vectors are connected, allowing for very fast traversal to find the nearest neighbors.

### 📜 Key Research Paper

*   **Paper:** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
*   **Link:** [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
*   **Contribution:** This paper from Facebook AI Research (now Meta AI) introduced and formalized the RAG framework. It showed that RAG models could achieve state-of-the-art results on knowledge-intensive tasks like question answering, while also providing the benefits of being more factual, verifiable (you can cite the retrieved sources), and easier to update (just update the vector database, no need to retrain the LLM).

### 💻 Project: Build a Mini-RAG System

Build a simplified RAG system that can answer questions about a small text file.

1.  **Choose a Document:** Find a plain text file about a specific topic (e.g., the Wikipedia page for "Giraffe").
2.  **Indexing:**
    *   Load the text and split it into chunks (e.g., paragraphs or sentences).
    *   Load a pre-trained sentence-transformer embedding model from Hugging Face (`pip install sentence-transformers`).
    *   `from sentence_transformers import SentenceTransformer`
    *   `model = SentenceTransformer('all-MiniLM-L6-v2')`
    *   Convert all your text chunks into embeddings using `model.encode(chunks)`.
3.  **Vector "Database":** For this simple project, your "database" can just be a NumPy array of the embeddings.
4.  **Retrieval:**
    *   Write a function that takes a user query.
    *   Encode the query into an embedding with the same model.
    *   Use `scikit-learn`'s `cosine_similarity` to calculate the similarity between the query embedding and all the chunk embeddings.
    *   Find the top 2-3 chunks with the highest similarity.
5.  **Generation:**
    *   Construct an augmented prompt: `f"Context: {retrieved_chunks}\n\nQuestion: {query}\n\nAnswer:"`
    *   Use a pre-trained generative model (like GPT-2 from Day 8) to generate an answer based on this prompt.
    *   Ask it questions that can only be answered with the text, e.g., "How tall is a giraffe?".

### ✅ Progress Tracker

*   [ ] I can explain why LLMs sometimes "hallucinate" and how RAG helps solve this.
*   [ ] I can outline the two main phases of a RAG system: Indexing and Retrieval/Generation.
*   [ ] I understand the role of the embedding model and the vector database.
*   [ ] I have built a simplified RAG system that uses a document to answer questions.
