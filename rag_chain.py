from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


def load_chain():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION")
    )

    vectorstore = Chroma(
        persist_directory="./confluence_index",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant for Business Analysts.
Answer the question using ONLY the context provided below.
If the content is unclear, incomplete, or contradictory — say so explicitly.
If you cannot find a confident answer, say exactly what you did find and what is missing.
Never make up information not present in the context.
Always mention which page the information came from.

Context:
{context}

Question: {question}
Answer:"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


def detect_relevant_page(conversation: str, embeddings, vectorstore) -> dict:
    """Use RAG to find the most relevant Confluence page for a conversation."""
    results = vectorstore.similarity_search(conversation[:2000], k=1)
    if results:
        return results[0].metadata
    return None


if __name__ == "__main__":
    chain = load_chain()
    result = chain.invoke("What is the claims process?")
    print(result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata['title']}: {doc.metadata['url']}")