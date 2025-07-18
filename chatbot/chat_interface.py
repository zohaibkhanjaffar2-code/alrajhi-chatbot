from openai import OpenAI
from chatbot.utils import get_openai_api_key
from chatbot.rag_engine import search_similar_docs

def ask_chatgpt(query, vectorstore):
    context = search_similar_docs(query, vectorstore)

    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions about Al Rajhi Bank. "
                    "Use only the information provided in the context. Always answer in clear, helpful English."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )

    return response.choices[0].message.content
