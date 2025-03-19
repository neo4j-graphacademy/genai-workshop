import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader

COURSES_PATH = "1-knowledge-graphs-vectors/data/asciidoc"

# Load lesson documents
loader = DirectoryLoader(COURSES_PATH, glob="**/lesson.adoc", loader_cls=TextLoader)
docs = loader.load()

# Create a text splitter
# text_splitter =

# Split documents into chunks
# chunks =

# Create an embedding provider
# embedding_provider =

# Create a Neo4j vector store
# neo4j_db =