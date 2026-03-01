import os
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()
# =========================
# 1. Model
# =========================

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

# =========================
# 2. Vector Memory (Long-term)
# =========================

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents([], embeddings)

def save_to_memory(text: str):
    vector_store.add_documents([Document(page_content=text)])

def retrieve_memory(query: str):
    docs = vector_store.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# =========================
# 3. Planner
# =========================

def create_plan(goal: str) -> str:
    response = llm([
        SystemMessage(content="Jesteś planerem. Twórz krótki plan kroków."),
        HumanMessage(content=f"Cel: {goal}")
    ])
    return response.content

# =========================
# 4. Executor (izolowany kontekst)
# =========================

def execute_step(step: str, state_summary: str) -> str:
    context = f"""
    Aktualny stan:
    {state_summary}

    Wykonaj krok:
    {step}

    Zwróć tylko kluczowe fakty.
    """

    response = llm([HumanMessage(content=context)])
    return response.content

# =========================
# 5. Rolling summary
# =========================

def compress_state(previous_summary: str, new_info: str) -> str:
    prompt = f"""
    Poprzedni stan:
    {previous_summary}

    Nowe informacje:
    {new_info}

    Zaktualizuj krótki stan zadania.
    """

    response = llm([HumanMessage(content=prompt)])
    return response.content

# =========================
# 6. Główna pętla
# =========================

def run_agent(goal: str):
    print("Cel:", goal)

    plan = create_plan(goal)
    steps = plan.split("\n")

    state_summary = "Zadanie rozpoczęte."

    for step in steps:
        if not step.strip():
            continue

        memory_context = retrieve_memory(step)
        result = execute_step(step, state_summary + "\n" + memory_context)

        state_summary = compress_state(state_summary, result)
        save_to_memory(result)

    print("Finalny stan:")
    print(state_summary)


if __name__ == "__main__":
    run_agent("Zbadaj rynek kursów AI i podsumuj trendy.")