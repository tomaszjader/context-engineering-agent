import os
import warnings
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()
# =========================
# 1. Model
# =========================

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# =========================
# 2. Vector Memory (Long-term)
# =========================

embeddings = OpenAIEmbeddings()
# W nowszych wersjach LangChain FAISS wymaga przynajmniej jednego dokumentu na start do ustalenia wymiarowości wektora.
vector_store = FAISS.from_documents([Document(page_content="Inicjalizacja pamięci agenta.")], embeddings)

def save_to_memory(text: str):
    vector_store.add_documents([Document(page_content=text)])

def retrieve_memory(query: str):
    docs = vector_store.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# =========================
# 3. Planner
# =========================

def create_plan(goal: str) -> str:
    response = llm.invoke([
        SystemMessage(content="Jesteś planerem. Twórz krótki plan kroków."),
        HumanMessage(content=f"Cel: {goal}")
    ])
    return response.content

from langgraph.prebuilt import create_react_agent
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

# =========================
# 4. Executor (izolowany kontekst z narzędziami)
# =========================

search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

def execute_step(step: str, state_summary: str) -> str:
    prompt = f"""
    Jesteś asystentem wykonującym pobieranie i analizowanie danych. Zawsze zwracaj konkretne fakty. Używaj wyszukiwarki internetowej, aby dowiedzieć się potrzebnych i bieżących informacji na dany temat.
    Aktualny stan zadania:
    {state_summary}
    
    Wykonaj ten krok:
    {step}
    """
    
    agent = create_react_agent(llm, tools=tools)
    
    response = agent.invoke({"messages": [("user", prompt)]})
    return response["messages"][-1].content

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

    response = llm.invoke([HumanMessage(content=prompt)])
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
    run_agent("Przeszukaj listę najpopularniejszych darmowych kursów AI i wskaż 3 najlepsze skupiające się na budowaniu RAGów w LangChain")