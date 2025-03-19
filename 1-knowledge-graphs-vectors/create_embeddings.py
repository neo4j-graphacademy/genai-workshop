import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
    )

embedding = embedding_provider.embed_query(
    "Text to create embeddings for"
    )

print(embedding)