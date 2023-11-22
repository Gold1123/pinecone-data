import os
import pypdf
import langchain
import pptx
# import unstructured
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import CSVLoader, PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader
import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import openai
import pinecone
from dotenv import load_dotenv


load_dotenv()

embeddings = OpenAIEmbeddings()
api_key = os.getenv('PINECONE_API_KEY')

pinecone.init(
    api_key=api_key,  # find at app.pinecone.io
    environment=os.getenv('PINECONE_ENV'),  # next to api key in console
)
index_name = os.getenv('PINECONE_INDEX')


tokenizer = tiktoken.get_encoding('cl100k_base')

file_name = ""
path = ""


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def remove_filename_from_path(Path, Filename):
    return Path.rsplit(Filename, 1)[0]


def split_document(doc: Document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])
    return chunks


def get_chunks(documents):
    global path
    final_chunks = []
    answer = []
    chunk_metadata = {}
    path = remove_filename_from_path(path, file_name)
    for document in documents:
        # print(document.metadata)
        if hasattr(document, "metadata"):
            chunk_metadata = document.metadata
        chunk_metadata['path'] = path
        chunk_metadata['source'] = file_name
        document.metadata = chunk_metadata
        chunks = split_document(document)
        final_chunks += chunks
    for final_chunk in final_chunks:
        final_chunk.metadata['text'] = final_chunk.page_content
        # print(final_chunk.metadata)
    return final_chunks


def from_pdf():
    folder_path = '/tmp'
    loader = PyPDFLoader(os.path.join(folder_path, file_name))
    documents = loader.load()
    return get_chunks(documents)


def from_csv():
    folder_path = '/tmp'
    loader = CSVLoader(os.path.join(folder_path, file_name))
    documents = loader.load()
    return get_chunks(documents)


def from_txt():
    folder_path = '/tmp'
    loader = TextLoader(os.path.join(folder_path, file_name))
    documents = loader.load()
    return get_chunks(documents)


def from_pptx():
    folder_path = '/tmp'
    loader = UnstructuredPowerPointLoader(os.path.join(folder_path, file_name))
    documents = loader.load()
    return get_chunks(documents)


def from_ms_word():
    folder_path = '/tmp'
    loader = Docx2txtLoader(os.path.join(folder_path, file_name))
    documents = loader.load()
    return get_chunks(documents)


def convert_to_vectors(chunks):
    texts = [chunk.page_content for chunk in chunks]
    return embeddings.embed_documents(texts)


def handler():
    global file_name, path
    # file_name = 'C7335 Supplemental Response Detail - unitiFM_v4-9t.pdf'
    extension = os.path.splitext(file_name)[1]
    print(extension)

    if extension == ".csv":
        chunks = from_csv()
    elif extension == ".pdf":
        chunks = from_pdf()
    elif extension == ".txt":
        chunks = from_txt()
    elif extension == ".docx":
        chunks = from_ms_word()
    elif extension == ".pptx":
        chunks = from_pptx()

    vectors = convert_to_vectors(chunks)

    vectors_to_upsert = []
    for vector_id, vector in enumerate(vectors):
        item = {
            "id": f"vec{vector_id}",
            "values": vector,
            "metadata": chunks[vector_id].metadata
        }
        vectors_to_upsert.append(item)
    index = pinecone.Index(index_name)
    index.upsert(vectors_to_upsert)
    # return chunks


def main():
    # Specify the folder path
    folder_path = './SYD'
    # Get a list of all files in the folder
    file_names = [f for f in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(file_names)


main()
