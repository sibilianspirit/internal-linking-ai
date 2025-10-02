import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from jina_reranker.api import JinaReranker
import time

# --- Konfiguracja strony Streamlit ---
st.set_page_config(page_title="AI do Linkowania Wewnętrznego", layout="centered")
st.title("🤖 AI do Linkowania Wewnętrznego")
st.info(
    """
    **Etapy procesu:**
    1. **Wyszukiwanie -** model embeddingowy znajduje 10 potencjalnych kandydatów.
    2. **Reranking -** model rerankingowy precyzyjnie ocenia tych 10 kandydatów, aby wybrać 5 najlepszych.

    **Wymagania:**
    1. **Plik CSV -** musi zawierać kolumny: `url`, `h1`, `title`.
    2. **Wybór pomiędzy h1 i title -** wskaż, na której kolumnie ma bazować model.
    """
)

# --- Stałe konfiguracyjne ---
EMBEDDING_MODEL = 'text-embedding-3-large'
# ZMIANA: Nowy, lepszy model rerankujący
RERANKER_MODEL = 'jinaai/jina-reranker-v2-base-multilingual' 
NUM_CANDIDATES = 10
NUM_FINAL_RESULTS = 5

# --- Funkcje pomocnicze ---

@st.cache_data
def get_embeddings(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    # ... (ta funkcja pozostaje bez zmian)
    client = OpenAI(api_key=api_key)
    embeddings = []
    batch_size = 100
    progress_bar = st.progress(0, text="Etap 1: Generowanie embeddingów...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [text if isinstance(text, str) and text.strip() != "" else " " for text in batch]
        response = client.embeddings.create(input=batch, model=model)
        embeddings.extend([item.embedding for item in response.data])
        progress_val = min((i + batch_size) / len(texts), 1.0)
        progress_text = f"Etap 1: Generowanie embeddingów... Przetworzono {min(i + batch_size, len(texts))}/{len(texts)} URLi."
        progress_bar.progress(progress_val, text=progress_text)
        time.sleep(0.5)
    progress_bar.empty()
    return embeddings

@st.cache_resource
def load_reranker_model(model_name: str):
    """Ładuje model Jina Reranker z pamięci podręcznej."""
    # ZMIANA: Inicjalizujemy nową klasę modelu
    return JinaReranker(model_name)

# --- Główna logika aplikacji ---
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Błąd: Klucz OPENAI_API_KEY nie został ustawiony w sekretach aplikacji!")
    st.info("Przejdź do ustawień aplikacji w Streamlit Community Cloud i dodaj swój klucz.")
    st.stop()

# Interfejs użytkownika
uploaded_file = st.file_uploader("1. Wgraj plik CSV", type=["csv"], help="Upewnij się, że plik zawiera kolumny: 'url', 'title' oraz 'h1'.")
column_to_embed = st.selectbox("2. Wybierz kolumnę do analizy", ("h1", "title"), disabled=uploaded_file is None)

if st.button("🚀 Uruchom analizę z Jina Reranker", disabled=(uploaded_file is None)):
    try:
        df = pd.read_csv(uploaded_file)
        # ... (walidacja pliku bez zmian)
        required_columns = ['url', 'title', 'h1']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Błąd: Plik CSV musi zawierać kolumny: {', '.join(required_columns)}")
        else:
            # ... (Kroki 1, 2, 3 - generowanie embeddingów, retrieval, ładowanie modelu - pozostają koncepcyjnie takie same)
            st.write("✅ **Etap 1/4:** Generowanie embeddingów...")
            texts_to_embed = df[column_to_embed].fillna(" ").tolist()
            df['embedding'] = get_embeddings(texts_to_embed, EMBEDDING_MODEL, api_key)
            
            st.write("✅ **Etap 2/4:** Wstępne wyszukiwanie kandydatów...")
            embeddings_matrix = np.vstack(df['embedding'].values)
            similarity_matrix = cosine_similarity(embeddings_matrix)
            
            st.write(f"✅ **Etap 3/4:** Ładowanie modelu Jina Reranker...")
            with st.spinner("Model Jina Reranker ładuje się tylko za pierwszym razem, proszę czekać..."):
                reranker = load_reranker_model(RERANKER_MODEL)

            st.write(f"✅ **Etap 4/4:** Precyzyjny Reranking z Jina...")
            progress_bar_rerank = st.progress(0, text="Reranking...")
            
            all_results = []
            for idx in range(len(df)):
                source_text = df[column_to_embed].iloc[idx]
                candidate_indices = similarity_matrix[idx].argsort()[-(NUM_CANDIDATES + 1):-1][::-1]
                candidate_texts = df[column_to_embed].iloc[candidate_indices].tolist()
                
                # ZMIANA: Nowy, prostszy sposób na reranking
                reranked_results = reranker.rerank(
                    query=source_text,
                    documents=candidate_texts,
                    top_n=NUM_FINAL_RESULTS # Od razu prosimy o 5 najlepszych wyników
                )
                
                # ZMIANA: Przetwarzanie nowego formatu wyników
                # Wynik to lista słowników, np. [{'index': 2, 'relevance_score': 0.9, 'document': 'text...'}, ...]
                # Potrzebujemy oryginalnych URLi, więc musimy dopasować indeksy
                original_urls = df['url'].iloc[candidate_indices].tolist()
                top_urls = [original_urls[res['index']] for res in reranked_results]

                result_row = {'original_url': df['url'].iloc[idx]}
                for i, url in enumerate(top_urls):
                    result_row[f'rekomendowany_link_{i+1}'] = url
                all_results.append(result_row)

                progress_bar_rerank.progress((idx + 1) / len(df), text=f"Reranking... {idx + 1}/{len(df)}")

            # ... (reszta kodu bez zmian)
            progress_bar_rerank.empty()
            output_df = pd.DataFrame(all_results)
            st.success("🎉 Analiza zakończona pomyślnie!")
            st.dataframe(output_df.head())
            csv_output = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Pobierz wyniki (output_jina_reranked.csv)",
                data=csv_output,
                file_name='output_jina_reranked.csv',
                mime='text/csv',
            )
    except Exception as e:
        st.error(f"Wystąpił nieoczekiwany błąd podczas przetwarzania: {e}")
        st.warning("Sprawdź swój klucz API, limity konta oraz połączenie z internetem.")
