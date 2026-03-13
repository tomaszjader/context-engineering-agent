# Context Engineering Agent

Ten projekt to prosty agent wykorzystujący architekturę LangChain. Demonstruje on zastosowanie pamięci wektorowej (FAISS) oraz inżynierii kontekstu, dzięki czemu agent może wykonywać złożone zadania bez ryzyka przepełnienia okna kontekstowego modelu językowego.

Agent dzieli pracę na 3 główne obszary:
1. **Planner** – tworzy krótki plan kroków dla zadanego celu.
2. **Executor** – wykonuje poszczególne kroki w odizolowanym kontekście.
3. **Pamięć wektorowa i Rolling Summary** – korzysta z bazy wektorowej (FAISS) do trwałego zapisu konkluzji z kroków i kompresuje rosnący stan zadania, żeby nie przekroczyć limitów tokenów.

## Wymagania

- Python 3.8+
- [Klucz API OpenAI](https://platform.openai.com/api-keys)

## Instalacja

1. Pobierz lub sklonuj kod projektu do wybranego folderu.
2. (Zalecane) Utwórz i aktywuj wirtualne środowisko:
   ```bash
   python -m venv venv
   # Na Windows (PowerShell/CMD):
   venv\Scripts\activate
   # Na Linux / macOS:
   source venv/bin/activate
   ```
3. Zainstaluj wymagane pakiety:
   ```bash
   pip install -r requirements.txt
   ```

## Konfiguracja środowiska

Do działania skrypt wymaga klucza API OpenAI.

1. Skopiuj plik `.env.example` do nowego pliku `.env`.
   ```bash
   cp .env.example .env
   ```
2. Otwórz plik `.env` i wstaw swój klucz API w miejsce wartości domyślnej:
   ```env
   OPENAI_API_KEY=sk-...
   ```
*(Opcjonalnie: skrypt `main.py` bazowo oczekuje istnienia zmiennej środowiskowej `OPENAI_API_KEY` w systemie. Pamiętaj, aby przed uruchomieniem wczytać ją z pliku `.env` albo po prostu dodać do środowiska)*

## Uruchamianie

Aby uruchomić aplikację i zobaczyć jak agent wykonuje zakodowane w `main.py` zadanie, wykonaj komendę:

```bash
python main.py
```
