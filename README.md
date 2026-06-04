# Research Paper Semantic Search

  A semantic search and summarization engine for academic literature built by Ankit Kumar — AI/GenAI Engineer with 3+ years of experience building RAG and LLM systems.

  ## Overview
  Indexes research papers with dense vector embeddings, enables semantic similarity search beyond keyword matching, and generates GPT-4o summaries with key findings extraction.

  ## Features
  - Dense vector indexing via OpenAI text-embedding-3-large
  - Semantic similarity search across 50K+ papers
  - GPT-4o-powered summarization with key findings
  - Cross-lingual search support
  - Citation graph traversal
  - FastAPI endpoints + bulk ingestion pipeline
  - Relevance scoring with reranking

  ## Architecture
  ```
  Papers → Chunker → Embedder → Pinecone → Retriever → Reranker → GPT-4o Summary
  ```

  ## Tech Stack
  Python · OpenAI · Pinecone · LangChain · FastAPI · arXiv API · Docker

  ## Setup
  ```bash
  pip install -r requirements.txt
  cp .env.example .env
  uvicorn main:app --reload
  ```

  ## Metrics
  | Metric | Value |
  |--------|-------|
  | Retrieval NDCG@10 | 0.87 |
  | Summary Quality (human eval) | 4.3/5 |
  | Search Latency | <800ms |
  | Index Size | 50K+ papers |

  ## Contact
  **Ankit Kumar** · ankitthakur104@gmail.com · [GitHub](https://github.com/ankitthakur104)
  