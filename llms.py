from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="gemma3:latest",
    temperature=0.1,
    base_url="http://localhost:11434",
)


embeddings = OllamaEmbeddings(
    model="embeddinggemma:latest",
    base_url="http://localhost:11434",
)


# prompt = ChatPromptTemplate.from_template("""here is the query: {input}""")

# chain = prompt | llm | StrOutputParser()


# response = chain.invoke({"input": "The meaning of life is "})

# print(response)


# vector = embeddings.embed_query("The meaning of life")

# print(vector)
