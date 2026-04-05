from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document


def build_retriever(data):
    docs = [
        Document(page_content=item["context"], metadata={"answer": item["answer"]})
        for item in data
    ]

    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore.as_retriever()