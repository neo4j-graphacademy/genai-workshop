import os
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

COURSES_PATH = "1-knowledge-graphs-vectors/data/asciidoc"

loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
    add_start_index=True
)

chunks = text_splitter.split_documents(docs)

# Create a function to get the embedding

# Create a function to get the course data

# Create OpenAI object

# Connect to Neo4j

# Create a function to run the Cypher query

# Iterate through the chunks and create the graph

# Close the neo4j driver