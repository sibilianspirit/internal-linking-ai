import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.cross_encoder import CrossEncoder
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
# ZMIANA: Poprawiona nazwa modelu rerankującego
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2' 
NUM_CANDIDATES = 10
NUM_FINAL_RESULTS = 5

# --- Funkcje pomocnicze ---

@st.cache_data
def get_embeddings(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    """Generuje embeddingi dla listy tekstów przy użyciu API OpenAI."""
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
    """Ładuje model rerankujący z pamięci podręcznej."""
    return CrossEncoder(model_name)

# --- Główna logika aplikacji ---

# ZMIANA: Sprawdzenie, czy klucz API został ustawiony w sekretach Streamlit
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Błąd: Klucz OPENAI_API_KEY nie został ustawiony w sekretach aplikacji!")
    st.info("Przejdź do ustawień aplikacji w Streamlit Community Cloud, kliknij 'Settings' > 'Secrets' i dodaj swój klucz.")
    st.stop() # Zatrzymaj wykonywanie aplikacji, jeśli klucz nie jest dostępny

# Interfejs użytkownika
uploaded_file = st.file_uploader("1. Wgraj plik CSV", type=["csv"], help="Upewnij się, że plik zawiera kolumny: 'url', 'title' oraz 'h1'.")
column_to_embed = st.selectbox("2. Wybierz kolumnę do analizy", ("h1", "title"), disabled=uploaded_file is None)

if st.button("🚀 Uruchom zaawansowaną analizę", disabled=(uploaded_file is None)):
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['url', 'title', 'h1']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Błąd: Plik CSV musi zawierać kolumny: {', '.join(required_columns)}")
        else:
            # KROK 1: Generowanie embeddingów
            st.write("✅ **Etap 1/4:** Generowanie embeddingów...")
            texts_to_embed = df[column_to_embed].fillna(" ").tolist()
            embeddings_list = get_embeddings(texts_to_embed, EMBEDDING_MODEL, api_key)
            df['embedding'] = embeddings_list
            
            # KROK 2: Wyszukiwanie kandydatów (Retrieval)
            st.write("✅ **Etap 2/4:** Wstępne wyszukiwanie kandydatów...")
            embeddings_matrix = np.vstack(df['embedding'].values)
            similarity_matrix = cosine_similarity(embeddings_matrix)
            
            # KROK 3: Ładowanie modelu rerankującego
            st.write(f"✅ **Etap 3/4:** Ładowanie modelu do rerankingu...")
            with st.spinner("Model rerankujący ładuje się tylko za pierwszym razem, proszę czekać..."):
                reranker = load_reranker_model(RERANKER_MODEL)

            # KROK 4: Precyzyjny Reranking
            st.write(f"✅ **Etap 4/4:** Precyzyjny Reranking kandydatów...")
            progress_bar_rerank = st.progress(0, text="Reranking...")
            
            all_results = []
            for idx in range(len(df)):
                source_text = df[column_to_embed].iloc[idx]
                candidate_indices = similarity_matrix[idx].argsort()[-(NUM_CANDIDATES + 1):-1][::-1]
                candidate_texts = df[column_to_embed].iloc[candidate_indices].tolist()
                pairs = [[source_text, candidate_text] for candidate_text in candidate_texts]
                
                scores = reranker.predict(pairs)
                
                reranked_results = list(zip(candidate_indices, scores))
                reranked_results.sort(key=lambda x: x[1], reverse=True)
                
                top_final_indices = [res[0] for res in reranked_results[:NUM_FINAL_RESULTS]]
                top_urls = df['url'].iloc[top_final_indices].tolist()
                
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
                label="📥 Pobierz wyniki (output_reranked.csv)",
                data=csv_output,
                file_name='output_reranked.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"Wystąpił nieoczekiwany błąd podczas przetwarzania: {e}")
        st.warning("Sprawdź swój klucz API w ustawieniach, limity konta oraz połączenie z internetem.")



