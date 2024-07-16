# Medical ChatBot

This project is a Medical ChatBot system that allows users to ask questions about medical documents. The system uses vector embeddings and Pinecone for efficient similarity search to retrieve relevant information from a pre-processed database of medical documents.

## Features

- Extracts and processes text data from PDF documents.
- Uses Hugging Face embeddings for text representation.
- Stores and retrieves vectors using Pinecone for efficient search.
- Integrates a language model for generating answers to user queries.
- Utilizes Streamlit for a user-friendly web interface.


### Prerequisites

- Python 3.8 or higher
- Pip (Python package installer)
- Pinecone account and API key
- Hugging Face account and model access

## Workflow

![Workflow](medicalchatbot/images/Workflow.png)

## Training Data

- The GALE ENCYCLOPEDIA of MEEDICINE Second Edition Volume 1 (https://github.com/Rittik2002/medicalchatbot/tree/main/data)

## Database

- Pinecone (https://www.pinecone.io/)

## Model and Embeddings Installation

- Model: - 
- llama-2-7b-chat.ggmlv3.q4_0.bin (https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)

- Embedding Model: - 
- sentence-transformers/all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)


## Clone the Repository

   ```bash
   git clone https://github.com/Rittik2002/medicalchatbot.git
   cd medicalchatbot
