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
    data = {}

    filename = chunk.metadata["source"]
    paragraph_id = f"{filename}.{chunk.metadata["start_index"]}"
    
    path = filename.split(os.path.sep)
    data['course'] = path[-6]
    data['module'] = path[-4]
    data['lesson'] = path[-2]
    data['url'] = f"https://graphacademy.neo4j.com/courses/{data['course']}/{data['module']}/{data['lesson']}"
    data['id'] = paragraph_id
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

# tag::llm[]
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model_name="gpt-3.5-turbo"
)
# end::llm[]

# tag::doc_transformer[]
doc_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Technology", "Concept", "Skill", "Event", "Person", "Object"],
    )
# end::doc_transformer[]

# tag::llm_graph_docs[]
for chunk in chunks:
    data = get_course_data(embedding_provider, chunk)
    create_chunk(graph, data)

    graph_docs = doc_transformer.convert_to_graph_documents([chunk])
    # end::llm_graph_docs[]
    
    # tag::map_entities[]
    for graph_doc in graph_docs:
        paragraph_node = Node(
            id=data["id"],
            type="Paragraph",
        )

        for node in graph_doc.nodes:

            graph_doc.relationships.append(
                Relationship(
                    source=paragraph_node,
                    target=node, 
                    type="HAS_ENTITY"
                    )
                )
    # end::map_entities[]

    # tag::llm_add_graph[]
    graph.add_graph_documents(graph_docs)
    # end::llm_add_graph[]

    print("Processed chunk", data['id'])
