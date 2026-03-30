from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from confluence_client import get_all_pages, extract_text, get_page_metadata
from dotenv import load_dotenv
import os

load_dotenv()


def build_index():
    print("Fetching Confluence pages...")
    pages = get_all_pages(os.getenv("CONFLUENCE_SPACE_KEY"))
    print(f"Found {len(pages)} pages.")

    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for page in pages:
        text = extract_text(page)
        meta = get_page_metadata(page)
        chunks = splitter.create_documents([text], metadatas=[meta])
        docs.extend(chunks)

    print(f"Embedding {len(docs)} chunks...")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION")
    )

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="./confluence_index"
    )

    print(f"Index saved to ./confluence_index — {len(docs)} chunks indexed.")


if __name__ == "__main__":
    build_index()