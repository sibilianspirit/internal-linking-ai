import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# ZMIANA 1: Poprawny import z biblioteki sentence-transformers
from sentence_transformers import CrossEncoder
import time

# --- Konfiguracja strony Streamlit ---
st.set_page_config(page_title="AI do Linkowania Wewnętrznego", layout="centered")
st.title("🤖 AI do Linkowania Wewnętrznego")
analysis_mode = st.radio(
    "Wybierz tryb analizy:",
    ("Linkowanie Wewnętrzne (jeden plik)", "Linkowanie Krzyżowe (dwa pliki)"),
    horizontal=True
)

if analysis_mode == "Linkowanie Wewnętrzne (jeden plik)":
    # --- Tutaj wklej cały istniejący kod logiki dla jednego pliku ---
    # (od st.info(...) aż do samego końca)
    st.info(...) # i tak dalej
    uploaded_file = st.file_uploader(...)
    # ... reszta starego kodu
else: # analysis_mode == "Linkowanie Krzyżowe (dwa pliki)"
    # --- Tutaj zdefiniujemy nową logikę dla dwóch plików ---
    st.info(
        """
        **Proces:** Model znajduje powiązania semantyczne między dwoma różnymi listami URL-i (np. kategorie i blog).
        **Wynik:** Dwa arkusze - jeden z rekomendacjami z Pliku 2 dla Pliku 1, a drugi odwrotnie.
        """
    )
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
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    # ... obsługa błędu klucza API (bez zmian)

# Interfejs dla dwóch plików
col1, col2 = st.columns(2)
with col1:
    uploaded_file_1 = st.file_uploader("1. Wgraj Plik 1 (np. Kategorie)", type=["csv"])
    column_to_embed_1 = st.selectbox("Wybierz kolumnę dla Pliku 1", ("h1", "title"), key="col1_select")

with col2:
    uploaded_file_2 = st.file_uploader("2. Wgraj Plik 2 (np. Blog)", type=["csv"])
    column_to_embed_2 = st.selectbox("Wybierz kolumnę dla Pliku 2", ("h1", "title"), key="col2_select")

if st.button("🚀 Uruchom analizę krzyżową", disabled=(uploaded_file_1 is None or uploaded_file_2 is None)):
    try:
        df1 = pd.read_csv(uploaded_file_1)
        df2 = pd.read_csv(uploaded_file_2)

        # Walidacja kolumn (można rozbudować)
        if 'url' not in df1.columns or 'url' not in df2.columns:
            st.error("Oba pliki muszą zawierać kolumnę 'url'.")
        else:
            # ETAP 1: Generowanie embeddingów dla obu plików
            st.write("✅ **Etap 1/4:** Generowanie embeddingów dla Pliku 1...")
            texts1 = df1[column_to_embed_1].fillna(" ").tolist()
            embeddings1 = get_embeddings(texts1, EMBEDDING_MODEL, api_key)

            st.write("✅ **Etap 2/4:** Generowanie embeddingów dla Pliku 2...")
            texts2 = df2[column_to_embed_2].fillna(" ").tolist()
            embeddings2 = get_embeddings(texts2, EMBEDDING_MODEL, api_key)

            # Przygotowanie macierzy
            matrix1 = np.vstack(embeddings1)
            matrix2 = np.vstack(embeddings2)

            # ETAP 3: Obliczanie podobieństwa krzyżowego
            st.write("✅ **Etap 3/4:** Obliczanie podobieństwa i reranking...")
            similarity_matrix_1_vs_2 = cosine_similarity(matrix1, matrix2)

            # Ładowanie modelu rerankującego
            reranker = load_reranker_model(RERANKER_MODEL)
            
            # ETAP 4: Przetwarzanie wyników w obie strony
            
            # --- Przetwarzanie 1 -> 2 (np. Kategorie -> Blog) ---
            results_1_to_2 = []
            for idx in range(len(df1)):
                source_text = texts1[idx]
                # Znajdź kandydatów z pliku 2
                candidate_indices = similarity_matrix_1_vs_2[idx].argsort()[-NUM_CANDIDATES:][::-1]
                candidate_texts = [texts2[i] for i in candidate_indices]
                
                # Reranking
                reranked_results = reranker.rank(source_text, candidate_texts, top_k=NUM_FINAL_RESULTS)
                
                # Zbieranie wyników
                original_urls_candidates = df2['url'].iloc[candidate_indices].tolist()
                top_urls = [original_urls_candidates[res['corpus_id']] for res in reranked_results]
                
                result_row = {'original_url': df1['url'].iloc[idx]}
                for i, url in enumerate(top_urls):
                    result_row[f'rekomendowany_link_{i+1}'] = url
                results_1_to_2.append(result_row)

            output_df_1_to_2 = pd.DataFrame(results_1_to_2)
            
            # --- Przetwarzanie 2 -> 1 (np. Blog -> Kategorie) ---
            # Używamy transpozycji macierzy podobieństwa, aby uniknąć ponownych obliczeń
            similarity_matrix_2_vs_1 = similarity_matrix_1_vs_2.T
            results_2_to_1 = []
            for idx in range(len(df2)):
                source_text = texts2[idx]
                # Znajdź kandydatów z pliku 1
                candidate_indices = similarity_matrix_2_vs_1[idx].argsort()[-NUM_CANDIDATES:][::-1]
                candidate_texts = [texts1[i] for i in candidate_indices]

                # Reranking
                reranked_results = reranker.rank(source_text, candidate_texts, top_k=NUM_FINAL_RESULTS)
                
                # Zbieranie wyników
                original_urls_candidates = df1['url'].iloc[candidate_indices].tolist()
                top_urls = [original_urls_candidates[res['corpus_id']] for res in reranked_results]
                
                result_row = {'original_url': df2['url'].iloc[idx]}
                for i, url in enumerate(top_urls):
                    result_row[f'rekomendowany_link_{i+1}'] = url
                results_2_to_1.append(result_row)
            
            output_df_2_to_1 = pd.DataFrame(results_2_to_1)

            st.success("🎉 Analiza krzyżowa zakończona pomyślnie!")
            
            # Wyświetlanie wyników
            st.subheader("Wyniki: Plik 1 -> Plik 2")
            st.dataframe(output_df_1_to_2.head())
            
            st.subheader("Wyniki: Plik 2 -> Plik 1")
            st.dataframe(output_df_2_to_1.head())
            
            # Pobieranie wyników jako plik Excel z dwoma arkuszami
            # Do tego potrzebna będzie nowa biblioteka: openpyxl
            from io import BytesIO
            output_excel = BytesIO()
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                output_df_1_to_2.to_excel(writer, sheet_name='Plik1_do_Pliku2', index=False)
                output_df_2_to_1.to_excel(writer, sheet_name='Plik2_do_Pliku1', index=False)
            
            st.download_button(
                label="📥 Pobierz wyniki (plik Excel)",
                data=output_excel.getvalue(),
                file_name='cross_linking_results.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

    except Exception as e:
        st.error(f"Wystąpił nieoczekiwany błąd: {e}")
# --- Stałe konfiguracyjne ---
EMBEDDING_MODEL = 'text-embedding-3-large'
RERANKER_MODEL = 'jinaai/jina-reranker-v2-base-multilingual' 
NUM_CANDIDATES = 10
NUM_FINAL_RESULTS = 5

# --- Funkcje pomocnicze ---

@st.cache_data
def get_embeddings(texts: list[str], model: str, api_key: str) -> list[list[float]]:
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
    """Ładuje model CrossEncoder z pamięci podręcznej."""
    # ZMIANA 2: Używamy klasy CrossEncoder do załadowania modelu
    # max_length jest ważne dla dłuższych tekstów, zgodnie z dokumentacją modelu
    return CrossEncoder(model_name, max_length=1024, trust_remote_code=True)

# --- Główna logika aplikacji ---
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Błąd: Klucz OPENAI_API_KEY nie został ustawiony w sekretach aplikacji!")
    st.info("Przejdź do ustawień aplikacji w Streamlit Community Cloud i dodaj swój klucz.")
    st.stop()

uploaded_file = st.file_uploader("1. Wgraj plik CSV", type=["csv"], help="Upewnij się, że plik zawiera kolumny: 'url', 'title' oraz 'h1'.")
column_to_embed = st.selectbox("2. Wybierz kolumnę do analizy", ("h1", "title"), disabled=uploaded_file is None)

if st.button("🚀 Uruchom analizę z Jina Reranker", disabled=(uploaded_file is None)):
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['url', 'title', 'h1']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Błąd: Plik CSV musi zawierać kolumny: {', '.join(required_columns)}")
        else:
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
                
                # ZMIANA 3: Używamy metody .rank() z odpowiednimi parametrami
                reranked_results = reranker.rank(
                    source_text,
                    candidate_texts,
                    return_documents=False, # Nie potrzebujemy tekstu, tylko indeksy
                    top_k=NUM_FINAL_RESULTS
                )
                
                # ZMIANA 4: Wynik to lista słowników, np. [{'corpus_id': 2, 'score': 0.9}, ...]
                # Dostosowujemy klucz z 'index' na 'corpus_id'
                original_urls = df['url'].iloc[candidate_indices].tolist()
                top_urls = [original_urls[res['corpus_id']] for res in reranked_results]

                result_row = {'original_url': df['url'].iloc[idx]}
                for i, url in enumerate(top_urls):
                    result_row[f'rekomendowany_link_{i+1}'] = url
                all_results.append(result_row)

                progress_bar_rerank.progress((idx + 1) / len(df), text=f"Reranking... {idx + 1}/{len(df)}")

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

