import os
from dotenv import load_dotenv
load_dotenv()

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship

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

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
    )

def get_course_data(embedding_provider, chunk):
    filename = chunk.metadata["source"]
    path = filename.split(os.path.sep)

    data = {}
    data['course'] = path[-6]
    data['module'] = path[-4]
    data['lesson'] = path[-2]
    data['url'] = f"https://graphacademy.neo4j.com/courses/{data['course']}/{data['module']}/{data['lesson']}"
    data['id'] = f"{filename}.{chunk.metadata["start_index"]}"
    data['text'] = chunk.page_content
    data['embedding'] = embedding_provider.embed_query(chunk.page_content)
    return data

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

def create_chunk(graph, data):
    graph.query("""
        MERGE (c:Course {name: $course})
        MERGE (c)-[:HAS_MODULE]->(m:Module{name: $module})
        MERGE (m)-[:HAS_LESSON]->(l:Lesson{name: $lesson, url: $url})
        MERGE (l)-[:CONTAINS]->(p:Paragraph{id: $id, text: $text})
        WITH p
        CALL db.create.setNodeVectorProperty(p, "embedding", $embedding)
        """, 
        data
    )

# Create an OpenAI LLM instance
# llm = 

# Create an LLMGraphTransformer instance
# doc_transformer =

for chunk in chunks:
    data = get_course_data(embedding_provider, chunk)
    create_chunk(graph, data)

    # Generate the graph docs
    # graph_docs =
    
    # Map the entities in the graph documents to the paragraph node
    # for graph_doc in graph_docs:
            
    # Add the graph documents to the graph
    # graph.
    
    print("Processed chunk", data['id'])
