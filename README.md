# RAG
Retrieval augmented generation
PDF Question Answering with Large Language Models (LLM)
This project demonstrates how to leverage a Large Language Model (LLM) to process and understand PDF documents for interactive question-answering.

Key Technologies
Torch: GPU acceleration for model inference

Langchain: Document loading, text splitting, embedding generation

HuggingFace Embeddings: Semantic text representations

Chroma: Vector store for efficient similarity search

LlamaCpp: LLM for generating contextual answers

Conversational Retrieval Chain: Maintains context during Q&A

Features
Device Adaptation: Uses GPU if available, else CPU

PDF Processing: Loads and splits documents into chunks

Embedding & Retrieval: Creates vector store for fast info lookup

Conversational Q&A: Context-aware interaction for accurate responses

How It Works
Load and extract text from the PDF

Split text into manageable chunks

Embed chunks and build vector store

Initialize LLM for answer generation

Run an interactive loop to answer user questions based on document content

Usage
Run the script and ask questions about the PDF. The system uses the LLM to provide relevant, context-aware answers, showcasing advanced document understanding capabilities.
