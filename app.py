import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import time
import re
from io import BytesIO

### FINALNE POPRAWKI UX ###
st.markdown("""
    <style>
    /* Import czcionki Readex Pro */
    @import url('https://fonts.googleapis.com/css2?family=Readex+Pro:wght@400;500;700&display=swap');
    
    /* Zastosowanie czcionki do całej aplikacji */
    html, body, [class*="st-"], h1, h2, h3 {
        font-family: 'Readex Pro', sans-serif;
    }
    
    /* Definicja niestandardowej ramki info */
    .custom-info-box {
        background-color: #75f86f; /* NOWY KOLOR TŁA */
        border-radius: 10px;
        padding: 20px;
        color: #111111;           /* NOWY KOLOR TEKSTU */
        margin-bottom: 20px;
    }

    /* Styl dla etykiet w ramce */
    .custom-info-box strong {
        color: #111111;           /* Kolor dla kontrastu na zielonym tle */
        font-weight: 700;        /* Pogrubienie dla lepszej czytelności */
    }
    </style>
    """, unsafe_allow_html=True)

# --- Konfiguracja strony Streamlit ---
st.set_page_config(page_title="Embedding-Based Linker", layout="centered")

# --- Stylizowany nagłówek z wymuszoną czcionką ---
st.markdown(
    "<h2 style='text-align: center; color: #5CFF87; font-family: \"Readex Pro\", sans-serif;'>🔗 Embedding-Based Linker</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; color: #FFFFFF; font-family: \"Readex Pro\", sans-serif;'>☢️ by RANKING RENEGADES</h3>",
    unsafe_allow_html=True
)

# --- Stałe konfiguracyjne ---
EMBEDDING_MODEL = 'text-embedding-3-large'
RERANKER_MODEL = 'jinaai/jina-reranker-v2-base-multilingual' 
NUM_CANDIDATES = 10
NUM_FINAL_RESULTS = 5

# Definicje aliasów dla kolumn
COLUMN_ALIASES = {
    'url': ['url', 'address'],
    'h1': ['h1'],
    'title': ['title']
}

# --- Funkcje pomocnicze ---

def find_column_map(df: pd.DataFrame) -> dict:
    df_columns_lower = {col.lower(): col for col in df.columns}
    column_map = {}
    
    for internal_name, aliases in COLUMN_ALIASES.items():
        found = False
        for alias in aliases:
            if alias in df_columns_lower:
                column_map[internal_name] = df_columns_lower[alias]
                found = True
                break
            for col_lower, original_col_name in df_columns_lower.items():
                if re.match(f"^{alias}[-_]?\d*$", col_lower):
                    column_map[internal_name] = original_col_name
                    found = True
                    break
            if found:
                break
    
    missing = [name for name in COLUMN_ALIASES if name not in column_map]
    if missing:
        raise ValueError(f"Nie znaleziono wymaganych kolumn w pliku CSV: {', '.join(missing)}. "
                         f"Upewnij się, że plik zawiera kolumny o nazwach podobnych do 'url'/'address', 'h1', 'title'.")
                         
    return column_map

@st.cache_data
def get_embeddings(texts: list[str], model: str, api_key: str, progress_text_prefix: str) -> list[list[float]]:
    client = OpenAI(api_key=api_key)
    embeddings = []
    batch_size = 100
    progress_bar = st.progress(0, text=f"{progress_text_prefix}: Generowanie embeddingów...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [text if isinstance(text, str) and text.strip() != "" else " " for text in batch]
        response = client.embeddings.create(input=batch, model=model)
        embeddings.extend([item.embedding for item in response.data])
        progress_val = min((i + batch_size) / len(texts), 1.0)
        progress_text = f"{progress_text_prefix}: Przetworzono {min(i + batch_size, len(texts))}/{len(texts)} URLi."
        progress_bar.progress(progress_val, text=progress_text)
        time.sleep(0.5)
    progress_bar.empty()
    return embeddings

@st.cache_resource
def load_reranker_model(model_name: str):
    return CrossEncoder(model_name, max_length=1024, trust_remote_code=True)

# --- Wybór trybu analizy ---
analysis_mode = st.radio(
    "Wybierz tryb analizy:",
    ("Linkowanie wewnętrzne (jeden plik)", "Linkowanie wewnętrzne (dwa pliki)"),
    horizontal=True
)


# ==============================================================================
# TRYB 1: LINKOWANIE WEWNĘTRZNE (JEDEN PLIK)
# ==============================================================================
if analysis_mode == "Linkowanie wewnętrzne (jeden plik)":
    ### POPRAWKA: Zmiana składni z Markdown na HTML ###
    info_text = """
    <strong>Proces:</strong><br>
    Analiza powiązań semantycznych w ramach jednego pliku.
    <br><br>
    <strong>Wymagania:</strong><br>
    Plik CSV musi zawierać kolumny `url`, `h1`, `title` (lub ich warianty, np. `Address`, `h1-1`).
    """
    st.markdown(f'<div class="custom-info-box">{info_text}</div>', unsafe_allow_html=True)

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Błąd: Klucz OPENAI_API_KEY nie został ustawiony w sekretach aplikacji!")
        st.stop()

    uploaded_file = st.file_uploader("1. Wgraj plik CSV", type=["csv"])
    
    column_options = st.empty()

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            column_map = find_column_map(df)
            
            df[column_map['h1']] = df[column_map['h1']].fillna(" ")
            df[column_map['title']] = df[column_map['title']].fillna(" ")

            options_for_select = [column_map['h1'], column_map['title']]
            column_to_embed = column_options.selectbox("2. Wybierz kolumnę do analizy", options_for_select)

            if st.button("🚀 Uruchom analizę wewnętrzną"):
                st.write("✅ **Etap 1/4:** Generowanie embeddingów...")
                texts_to_embed = df[column_to_embed].tolist()
                df['embedding'] = get_embeddings(texts_to_embed, EMBEDDING_MODEL, api_key, "Etap 1")
                
                st.write("✅ **Etap 2/4:** Wstępne wyszukiwanie kandydatów...")
                embeddings_matrix = np.vstack(df['embedding'].values)
                similarity_matrix = cosine_similarity(embeddings_matrix)
                
                st.write(f"✅ **Etap 3/4:** Ładowanie modelu Rerankującego...")
                reranker = load_reranker_model(RERANKER_MODEL)

                st.write(f"✅ **Etap 4/4:** Precyzyjny Reranking...")
                progress_bar_rerank = st.progress(0, text="Reranking...")
                
                all_results = []
                for idx in range(len(df)):
                    source_text = df[column_to_embed].iloc[idx]
                    candidate_indices = similarity_matrix[idx].argsort()[-(NUM_CANDIDATES + 1):-1][::-1]
                    candidate_texts = df[column_to_embed].iloc[candidate_indices].tolist()
                    
                    reranked_results = reranker.rank(source_text, candidate_texts, return_documents=False, top_k=NUM_FINAL_RESULTS)
                    
                    original_urls = df[column_map['url']].iloc[candidate_indices].tolist()
                    top_urls = [original_urls[res['corpus_id']] for res in reranked_results]
                    
                    result_row = {'URL': df[column_map['url']].iloc[idx]}
                    for i, url in enumerate(top_urls):
                        result_row[f'LINK {i+1}'] = url
                    all_results.append(result_row)
                    progress_bar_rerank.progress((idx + 1) / len(df), text=f"Reranking... {idx + 1}/{len(df)}")

                progress_bar_rerank.empty()
                output_df = pd.DataFrame(all_results)
                st.success("🎉 Analiza zakończona pomyślnie!")
                st.dataframe(output_df.head())
                csv_output = output_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Pobierz wyniki (CSV)", csv_output, 'output_internal_links.csv', 'text/csv')

        except Exception as e:
            st.error(f"Wystąpił błąd: {e}")

# ==============================================================================
# TRYB 2: LINKOWANIE WEWNĘTRZNE (DWA PLIKI)
# ==============================================================================
else:
    ### POPRAWKA: Zmiana składni z Markdown na HTML ###
    info_text = """
    <strong>Proces:</strong><br>
    Analiza powiązań między dwoma plikami (np. kategorie vs blog).
    <br><br>
    <strong>Wymagania:</strong><br>
    Oba pliki muszą zawierać kolumny `url`, `h1`, `title` (lub ich warianty).
    """
    st.markdown(f'<div class="custom-info-box">{info_text}</div>', unsafe_allow_html=True)
    
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Błąd: Klucz OPENAI_API_KEY nie został ustawiony w sekretach aplikacji!")
        st.stop()

    col1, col2 = st.columns(2)
    
    column_options_1 = col1.empty()
    column_options_2 = col2.empty()
    
    df1, df2 = None, None
    column_map1, column_map2 = None, None

    with col1:
        uploaded_file_1 = st.file_uploader("1. Wgraj Plik 1 (np. Kategorie)", type=["csv"])
        if uploaded_file_1:
            try:
                df1 = pd.read_csv(uploaded_file_1)
                column_map1 = find_column_map(df1)
                
                df1[column_map1['h1']] = df1[column_map1['h1']].fillna(" ")
                df1[column_map1['title']] = df1[column_map1['title']].fillna(" ")

                options1 = [column_map1['h1'], column_map1['title']]
                column_to_embed_1 = column_options_1.selectbox("Wybierz kolumnę dla Pliku 1", options1, key="col1_select")
            except Exception as e:
                st.error(f"Błąd w Pliku 1: {e}")
                df1 = None

    with col2:
        uploaded_file_2 = st.file_uploader("2. Wgraj Plik 2 (np. Blog)", type=["csv"])
        if uploaded_file_2:
            try:
                df2 = pd.read_csv(uploaded_file_2)
                column_map2 = find_column_map(df2)
                
                df2[column_map2['h1']] = df2[column_map2['h1']].fillna(" ")
                df2[column_map2['title']] = df2[column_map2['title']].fillna(" ")

                options2 = [column_map2['h1'], column_map2['title']]
                column_to_embed_2 = column_options_2.selectbox("Wybierz kolumnę dla Pliku 2", options2, key="col2_select")
            except Exception as e:
                st.error(f"Błąd w Pliku 2: {e}")
                df2 = None

    if st.button("🚀 Uruchom analizę krzyżową", disabled=(df1 is None or df2 is None)):
        try:
            texts1 = df1[column_to_embed_1].tolist()
            embeddings1 = get_embeddings(texts1, EMBEDDING_MODEL, api_key, "Plik 1")

            texts2 = df2[column_to_embed_2].tolist()
            embeddings2 = get_embeddings(texts2, EMBEDDING_MODEL, api_key, "Plik 2")

            matrix1 = np.vstack(embeddings1)
            matrix2 = np.vstack(embeddings2)

            st.write("✅ Obliczanie podobieństwa i reranking...")
            similarity_matrix_1_vs_2 = cosine_similarity(matrix1, matrix2)
            
            reranker = load_reranker_model(RERANKER_MODEL)
            
            progress_bar_rerank = st.progress(0, text="Reranking: Plik 1 -> Plik 2...")

            results_1_to_2 = []
            for idx in range(len(df1)):
                source_text = texts1[idx]
                candidate_indices = similarity_matrix_1_vs_2[idx].argsort()[-NUM_CANDIDATES:][::-1]
                candidate_texts = [texts2[i] for i in candidate_indices]
                
                reranked_results = reranker.rank(source_text, candidate_texts, top_k=NUM_FINAL_RESULTS, return_documents=False)
                
                original_urls_candidates = df2[column_map2['url']].iloc[candidate_indices].tolist()
                top_urls = [original_urls_candidates[res['corpus_id']] for res in reranked_results]
                
                result_row = {'URL': df1[column_map1['url']].iloc[idx]}
                for i, url in enumerate(top_urls):
                    result_row[f'LINK {i+1}'] = url
                results_1_to_2.append(result_row)
                progress_bar_rerank.progress((idx + 1) / len(df1), text=f"Reranking: Plik 1 -> Plik 2... ({idx + 1}/{len(df1)})")

            output_df_1_to_2 = pd.DataFrame(results_1_to_2)
            
            progress_bar_rerank.progress(0, text="Reranking: Plik 2 -> Plik 1...")

            similarity_matrix_2_vs_1 = similarity_matrix_1_vs_2.T
            results_2_to_1 = []
            for idx in range(len(df2)):
                source_text = texts2[idx]
                candidate_indices = similarity_matrix_2_vs_1[idx].argsort()[-NUM_CANDIDATES:][::-1]
                candidate_texts = [texts1[i] for i in candidate_indices]

                reranked_results = reranker.rank(source_text, candidate_texts, top_k=NUM_FINAL_RESULTS, return_documents=False)
                
                original_urls_candidates = df1[column_map1['url']].iloc[candidate_indices].tolist()
                top_urls = [original_urls_candidates[res['corpus_id']] for res in reranked_results]
                
                result_row = {'URL': df2[column_map2['url']].iloc[idx]}
                for i, url in enumerate(top_urls):
                    result_row[f'LINK {i+1}'] = url
                results_2_to_1.append(result_row)
                progress_bar_rerank.progress((idx + 1) / len(df2), text=f"Reranking: Plik 2 -> Plik 1... ({idx + 1}/{len(df2)})")
            
            progress_bar_rerank.empty()
            output_df_2_to_1 = pd.DataFrame(results_2_to_1)

            st.success("🎉 Analiza krzyżowa zakończona pomyślnie!")
            
            st.subheader("Wyniki: Rekomendacje z Pliku 2 dla Pliku 1")
            st.dataframe(output_df_1_to_2.head())
            
            st.subheader("Wyniki: Rekomendacje z Pliku 1 dla Pliku 2")
            st.dataframe(output_df_2_to_1.head())
            
            output_excel = BytesIO()
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                output_df_1_to_2.to_excel(writer, sheet_name='Rekomendacje_dla_Pliku_1', index=False)
                output_df_2_to_1.to_excel(writer, sheet_name='Rekomendacje_dla_Pliku_2', index=False)
            
            st.download_button(
                "📥 Pobierz wyniki (Excel)",
                output_excel.getvalue(),
                'cross_linking_results.xlsx',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            st.error(f"Wystąpił błąd podczas analizy: {e}")

