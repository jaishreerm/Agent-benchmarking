from openai import OpenAI
client = OpenAI()

def rag_answer(question, retriever):
    docs = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in docs[:3]])

    prompt = f"""
    Answer using ONLY the context.

    Context:
    {context}

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content