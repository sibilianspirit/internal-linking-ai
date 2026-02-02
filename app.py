import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import re
from io import BytesIO
import torch

# Monkey-patch: przywróć funkcję usuniętą w transformers>=4.40,
# wymaganą przez starszą wersję kodu modelu Jina w cache HuggingFace
import transformers.models.xlm_roberta.modeling_xlm_roberta as _xlm_module
if not hasattr(_xlm_module, 'create_position_ids_from_input_ids'):
    def _create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx
    _xlm_module.create_position_ids_from_input_ids = _create_position_ids_from_input_ids

from sentence_transformers import CrossEncoder

# --- Konfiguracja strony Streamlit ---
st.set_page_config(page_title="Embedding-Based Linker", layout="centered")

# --- Style CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Readex+Pro:wght@400;500;700&display=swap');
    
    html, body, [class*="st-"], h1, h2, h3 {
        font-family: 'Readex Pro', sans-serif;
    }
    
    .custom-info-box {
        background-color: #1c1e26;
        border: 2px solid #75f86f;
        border-radius: 10px;
        padding: 20px;
        color: #ffffff;
        margin-bottom: 20px;
    }

    .custom-info-box strong {
        color: #75f86f;
        font-weight: 700;
    }
    
    .cost-box {
        background-color: #2d2d3d;
        border: 1px solid #ffa500;
        border-radius: 8px;
        padding: 15px;
        color: #ffffff;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #3d2d2d;
        border: 1px solid #ff6b6b;
        border-radius: 8px;
        padding: 15px;
        color: #ffffff;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Nagłówek ---
st.markdown(
    "<h2 style='text-align: center; color: #5CFF87; font-family: \"Readex Pro\", sans-serif;'>🔗 Embedding-Based Linker</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; color: #FFFFFF; font-family: \"Readex Pro\", sans-serif;'>☢️ RANKING RENEGADES</h3>",
    unsafe_allow_html=True
)

# --- Stałe konfiguracyjne ---
EMBEDDING_MODEL = 'text-embedding-3-large'
RERANKER_MODEL = 'jinaai/jina-reranker-v2-base-multilingual'

# Ceny OpenAI API (per 1M tokenów) - text-embedding-3-large
EMBEDDING_COST_PER_1M_TOKENS = 0.13  # USD

# Rozszerzone aliasy kolumn - case-insensitive, obsługa różnych formatów z crawlerów
COLUMN_ALIASES: dict[str, list[str]] = {
    'url': ['url', 'address', 'link', 'page url', 'page_url', 'pageurl'],
    'h1': ['h1', 'heading 1', 'heading1', 'header 1', 'header1', 'main heading'],
    'title': ['title', 'page title', 'pagetitle', 'page_title', 'meta title', 'metatitle', 'meta_title', 'seo title']
}


# --- Funkcje pomocnicze ---

def normalize_column_name(col: str) -> str:
    """Normalizuje nazwę kolumny do porównania - usuwa cyfry, myślniki, podkreślenia na końcu."""
    col_lower = col.lower().strip()
    # Usuń końcowe wzorce typu: -1, _1, 1, -01, _01 itp.
    col_normalized = re.sub(r'[-_\s]*\d+$', '', col_lower)
    return col_normalized.strip()


def find_column_map(df: pd.DataFrame) -> dict[str, str]:
    """
    Znajduje mapowanie między wewnętrznymi nazwami kolumn a rzeczywistymi nazwami w DataFrame.
    Obsługuje różne formaty nazw kolumn z crawlerów (Screaming Frog, Sitebulb, itp.).
    """
    column_map: dict[str, str] = {}
    
    for internal_name, aliases in COLUMN_ALIASES.items():
        found = False
        
        for original_col in df.columns:
            normalized_col = normalize_column_name(original_col)
            
            # Sprawdź czy znormalizowana nazwa pasuje do któregoś aliasu
            for alias in aliases:
                alias_normalized = normalize_column_name(alias)
                if normalized_col == alias_normalized:
                    column_map[internal_name] = original_col
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            # Dodatkowa próba - sprawdź czy alias jest zawarty w nazwie kolumny
            for original_col in df.columns:
                col_lower = original_col.lower()
                for alias in aliases:
                    if alias in col_lower:
                        column_map[internal_name] = original_col
                        found = True
                        break
                if found:
                    break
    
    missing = [name for name in COLUMN_ALIASES if name not in column_map]
    if missing:
        available_cols = ', '.join(df.columns.tolist()[:10])
        if len(df.columns) > 10:
            available_cols += f'... (+{len(df.columns) - 10} więcej)'
        raise ValueError(
            f"Nie znaleziono wymaganych kolumn: {', '.join(missing)}.\n\n"
            f"Dostępne kolumny w pliku: {available_cols}\n\n"
            f"Akceptowane nazwy:\n"
            f"• URL: {', '.join(COLUMN_ALIASES['url'])}\n"
            f"• H1: {', '.join(COLUMN_ALIASES['h1'])}\n"
            f"• Title: {', '.join(COLUMN_ALIASES['title'])}"
        )
                         
    return column_map


def estimate_cost(texts: list[str], cost_per_1m_tokens: float) -> tuple[int, float, str]:
    """Szacuje liczbę tokenów i koszt API na podstawie tekstów."""
    # Przybliżenie: 1 token ≈ 4 znaki dla języka polskiego/angielskiego
    total_chars = sum(len(str(text)) for text in texts)
    estimated_tokens = total_chars // 4
    estimated_cost = (estimated_tokens / 1_000_000) * cost_per_1m_tokens
    
    # Formatowanie kosztu dla czytelności
    if estimated_cost < 0.01:
        cost_display = "< $0.01 USD"
    else:
        cost_display = f"~${estimated_cost:.2f} USD"
    
    return estimated_tokens, estimated_cost, cost_display


def validate_dataframe(df: pd.DataFrame, file_name: str) -> list[str]:
    """Waliduje DataFrame i zwraca listę ostrzeżeń."""
    warnings = []
    
    if len(df) == 0:
        warnings.append(f"⚠️ Plik {file_name} jest pusty!")
    elif len(df) > 5000:
        warnings.append(f"⚠️ Plik {file_name} zawiera {len(df)} wierszy. Przy dużych zbiorach analiza może trwać długo i generować znaczące koszty API.")
    elif len(df) > 1000:
        warnings.append(f"ℹ️ Plik {file_name} zawiera {len(df)} wierszy. Analiza może potrwać kilka minut.")
    
    return warnings


def check_duplicates(df: pd.DataFrame, url_column: str) -> tuple[int, pd.DataFrame]:
    """Sprawdza i usuwa duplikaty URL, zwraca liczbę usuniętych i oczyszczony DataFrame."""
    duplicates_count = df.duplicated(subset=[url_column], keep='first').sum()
    if duplicates_count > 0:
        df_clean = df.drop_duplicates(subset=[url_column], keep='first').reset_index(drop=True)
        return duplicates_count, df_clean
    return 0, df


@st.cache_data
def get_embeddings(texts: list[str], model: str, api_key: str, progress_text_prefix: str) -> list[list[float]]:
    """Generuje embeddingi dla listy tekstów z progress barem."""
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
        time.sleep(0.1)  # Zmniejszony delay
    
    progress_bar.empty()
    return embeddings


@st.cache_resource
def load_reranker_model(model_name: str):
    """Ładuje model rerankujący z cache."""
    return CrossEncoder(model_name, max_length=1024, trust_remote_code=True)


def batch_rerank(
    reranker: CrossEncoder,
    source_texts: list[str],
    candidate_texts_list: list[list[str]],
    top_k: int,
    progress_bar,
    progress_prefix: str
) -> list[list[dict]]:
    """
    Wykonuje reranking w trybie batch dla lepszej wydajności.
    """
    all_results = []
    total = len(source_texts)
    
    # Przetwarzaj w mini-batchach dla lepszego UX (widoczny progress)
    batch_size = 10
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_results = []
        
        for idx in range(batch_start, batch_end):
            source_text = source_texts[idx]
            candidate_texts = candidate_texts_list[idx]
            
            reranked = reranker.rank(
                source_text, 
                candidate_texts, 
                return_documents=False, 
                top_k=top_k
            )
            batch_results.append(reranked)
        
        all_results.extend(batch_results)
        progress_bar.progress(
            batch_end / total, 
            text=f"{progress_prefix} ({batch_end}/{total})"
        )
    
    return all_results


def get_candidates_excluding_self(
    similarity_matrix: np.ndarray, 
    idx: int, 
    num_candidates: int,
    exclude_indices: set[int] | None = None
) -> np.ndarray:
    """
    Zwraca indeksy kandydatów wykluczając self-linki i opcjonalnie inne indeksy.
    """
    similarities = similarity_matrix[idx].copy()
    
    # Wyklucz self-link
    similarities[idx] = -1
    
    # Wyklucz dodatkowe indeksy jeśli podane
    if exclude_indices:
        for exc_idx in exclude_indices:
            if exc_idx < len(similarities):
                similarities[exc_idx] = -1
    
    # Znajdź top kandydatów
    candidate_indices = similarities.argsort()[-num_candidates:][::-1]
    
    return candidate_indices


# --- Wybór trybu analizy ---
analysis_mode = st.radio(
    "Wybierz tryb analizy:",
    ("Linkowanie wewnętrzne (jeden plik)", "Linkowanie wewnętrzne (dwa pliki)"),
    horizontal=True
)

# --- Konfigurowalne parametry ---
show_settings = st.checkbox("⚙️ Pokaż ustawienia zaawansowane", value=False)

if show_settings:
    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        NUM_CANDIDATES = st.slider(
            "Liczba wstępnych kandydatów (embedding)", 
            min_value=5, 
            max_value=50, 
            value=15,
            help="Więcej kandydatów = dokładniejsze wyniki, ale dłuższy czas rerankingu"
        )
    with col_settings2:
        NUM_FINAL_RESULTS = st.slider(
            "Liczba finalnych linków", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Ile linków zwrócić dla każdego URL"
        )
    
    remove_duplicates = st.checkbox("Automatycznie usuń duplikaty URL", value=True)
else:
    # Domyślne wartości gdy ustawienia są ukryte
    NUM_CANDIDATES = 15
    NUM_FINAL_RESULTS = 5
    remove_duplicates = True


# ==============================================================================
# TRYB 1: LINKOWANIE WEWNĘTRZNE (JEDEN PLIK)
# ==============================================================================
if analysis_mode == "Linkowanie wewnętrzne (jeden plik)":
    info_text = """
    <strong>Proces:</strong><br>
    Analiza powiązań semantycznych w ramach jednego pliku.
    <br><br>
    <strong>Wymagania:</strong><br>
    Plik CSV musi zawierać kolumny typu URL/Address, H1/H1-1, Title/Title 1 (różne warianty są akceptowane).
    """
    st.markdown(f'<div class="custom-info-box">{info_text}</div>', unsafe_allow_html=True)

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Błąd: Klucz OPENAI_API_KEY nie został ustawiony w sekretach aplikacji!")
        st.stop()

    uploaded_file = st.file_uploader("1. Wgraj plik CSV", type=["csv"])
    
    column_options = st.empty()
    cost_container = st.empty()
    warnings_container = st.empty()

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            column_map = find_column_map(df)
            
            # Informacja o znalezionych kolumnach
            st.success(f"✅ Znaleziono kolumny: URL → `{column_map['url']}`, H1 → `{column_map['h1']}`, Title → `{column_map['title']}`")
            
            # Walidacja i ostrzeżenia
            warnings = validate_dataframe(df, uploaded_file.name)
            
            # Sprawdź duplikaty
            if remove_duplicates:
                duplicates_count, df = check_duplicates(df, column_map['url'])
                if duplicates_count > 0:
                    warnings.append(f"ℹ️ Usunięto {duplicates_count} zduplikowanych URLi.")
            
            if warnings:
                warnings_html = "<br>".join(warnings)
                warnings_container.markdown(f'<div class="warning-box">{warnings_html}</div>', unsafe_allow_html=True)
            
            df[column_map['h1']] = df[column_map['h1']].fillna(" ")
            df[column_map['title']] = df[column_map['title']].fillna(" ")

            options_for_select = [column_map['h1'], column_map['title']]
            column_to_embed = column_options.selectbox("2. Wybierz kolumnę do analizy", options_for_select)

            # Szacowanie kosztów
            texts_preview = df[column_to_embed].tolist()
            est_tokens, est_cost, cost_display = estimate_cost(texts_preview, EMBEDDING_COST_PER_1M_TOKENS)
            cost_container.markdown(
                f'<div class="cost-box">'
                f'<strong>📊 Szacowany koszt:</strong><br>'
                f'• Liczba URLi: {len(df)}<br>'
                f'• Szacowane tokeny: ~{est_tokens:,}<br>'
                f'• Szacowany koszt API: {cost_display}'
                f'</div>',
                unsafe_allow_html=True
            )

            if st.button("🚀 Uruchom analizę wewnętrzną"):
                st.write("✅ **Etap 1/4:** Generowanie embeddingów...")
                texts_to_embed = df[column_to_embed].tolist()
                df['embedding'] = get_embeddings(texts_to_embed, EMBEDDING_MODEL, api_key, "Etap 1")
                
                st.write("✅ **Etap 2/4:** Wstępne wyszukiwanie kandydatów...")
                embeddings_matrix = np.vstack(df['embedding'].values)
                
                # Progress dla obliczania similarity
                with st.spinner("Obliczanie macierzy podobieństwa..."):
                    similarity_matrix = cosine_similarity(embeddings_matrix)
                
                st.write(f"✅ **Etap 3/4:** Ładowanie modelu Rerankującego...")
                reranker = load_reranker_model(RERANKER_MODEL)

                st.write(f"✅ **Etap 4/4:** Precyzyjny Reranking...")
                progress_bar_rerank = st.progress(0, text="Reranking...")
                
                # Przygotuj dane do batch rerankingu
                source_texts = []
                candidate_texts_list = []
                candidate_indices_list = []
                
                for idx in range(len(df)):
                    source_texts.append(df[column_to_embed].iloc[idx])
                    # POPRAWKA: Wykluczenie self-linków
                    candidate_indices = get_candidates_excluding_self(
                        similarity_matrix, idx, NUM_CANDIDATES
                    )
                    candidate_indices_list.append(candidate_indices)
                    candidate_texts_list.append(df[column_to_embed].iloc[candidate_indices].tolist())
                
                # Batch reranking
                all_reranked = batch_rerank(
                    reranker, 
                    source_texts, 
                    candidate_texts_list, 
                    NUM_FINAL_RESULTS,
                    progress_bar_rerank,
                    "Reranking..."
                )
                
                # Budowanie wyników
                all_results = []
                for idx, reranked_results in enumerate(all_reranked):
                    candidate_indices = candidate_indices_list[idx]
                    original_urls = df[column_map['url']].iloc[candidate_indices].tolist()
                    top_urls = [original_urls[res['corpus_id']] for res in reranked_results]
                    
                    result_row = {'URL': df[column_map['url']].iloc[idx]}
                    for i, url in enumerate(top_urls):
                        result_row[f'LINK {i+1}'] = url
                    all_results.append(result_row)

                progress_bar_rerank.empty()
                output_df = pd.DataFrame(all_results)
                st.success("🎉 Analiza zakończona pomyślnie!")
                st.dataframe(output_df.head(10))
                
                # Statystyki
                st.info(f"📈 Wygenerowano rekomendacje dla {len(output_df)} URLi")
                
                csv_output = output_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Pobierz wyniki (CSV)", csv_output, 'output_internal_links.csv', 'text/csv')

        except Exception as e:
            st.error(f"Wystąpił błąd: {e}")


# ==============================================================================
# TRYB 2: LINKOWANIE WEWNĘTRZNE (DWA PLIKI)
# ==============================================================================
else:
    info_text = """
    <strong>Proces:</strong><br>
    Analiza powiązań między dwoma plikami (np. kategorie vs blog).
    <br><br>
    <strong>Wymagania:</strong><br>
    Oba pliki muszą zawierać kolumny typu URL/Address, H1/H1-1, Title/Title 1 (różne warianty są akceptowane).
    """
    st.markdown(f'<div class="custom-info-box">{info_text}</div>', unsafe_allow_html=True)
    
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("Błąd: Klucz OPENAI_API_KEY nie został ustawiony w sekretach aplikacji!")
        st.stop()

    col1, col2 = st.columns(2)
    
    df1, df2 = None, None
    column_map1, column_map2 = None, None
    column_to_embed_1, column_to_embed_2 = None, None

    with col1:
        uploaded_file_1 = st.file_uploader("1. Wgraj Plik 1 (np. Kategorie)", type=["csv"])
        if uploaded_file_1:
            try:
                df1 = pd.read_csv(uploaded_file_1)
                column_map1 = find_column_map(df1)
                
                st.success(f"✅ Kolumny: URL → `{column_map1['url']}`, H1 → `{column_map1['h1']}`, Title → `{column_map1['title']}`")
                
                # Walidacja
                warnings1 = validate_dataframe(df1, uploaded_file_1.name)
                if remove_duplicates:
                    dup_count, df1 = check_duplicates(df1, column_map1['url'])
                    if dup_count > 0:
                        warnings1.append(f"ℹ️ Usunięto {dup_count} duplikatów.")
                
                if warnings1:
                    for w in warnings1:
                        st.warning(w)
                
                df1[column_map1['h1']] = df1[column_map1['h1']].fillna(" ")
                df1[column_map1['title']] = df1[column_map1['title']].fillna(" ")

                options1 = [column_map1['h1'], column_map1['title']]
                column_to_embed_1 = st.selectbox("Wybierz kolumnę dla Pliku 1", options1, key="col1_select")
            except Exception as e:
                st.error(f"Błąd w Pliku 1: {e}")
                df1 = None

    with col2:
        uploaded_file_2 = st.file_uploader("2. Wgraj Plik 2 (np. Blog)", type=["csv"])
        if uploaded_file_2:
            try:
                df2 = pd.read_csv(uploaded_file_2)
                column_map2 = find_column_map(df2)
                
                st.success(f"✅ Kolumny: URL → `{column_map2['url']}`, H1 → `{column_map2['h1']}`, Title → `{column_map2['title']}`")
                
                # Walidacja
                warnings2 = validate_dataframe(df2, uploaded_file_2.name)
                if remove_duplicates:
                    dup_count, df2 = check_duplicates(df2, column_map2['url'])
                    if dup_count > 0:
                        warnings2.append(f"ℹ️ Usunięto {dup_count} duplikatów.")
                
                if warnings2:
                    for w in warnings2:
                        st.warning(w)
                
                df2[column_map2['h1']] = df2[column_map2['h1']].fillna(" ")
                df2[column_map2['title']] = df2[column_map2['title']].fillna(" ")

                options2 = [column_map2['h1'], column_map2['title']]
                column_to_embed_2 = st.selectbox("Wybierz kolumnę dla Pliku 2", options2, key="col2_select")
            except Exception as e:
                st.error(f"Błąd w Pliku 2: {e}")
                df2 = None

    # Szacowanie kosztów dla obu plików
    if df1 is not None and df2 is not None and column_to_embed_1 and column_to_embed_2:
        texts1 = df1[column_to_embed_1].tolist()
        texts2 = df2[column_to_embed_2].tolist()
        all_texts = texts1 + texts2
        est_tokens, est_cost, cost_display = estimate_cost(all_texts, EMBEDDING_COST_PER_1M_TOKENS)
        
        st.markdown(
            f'<div class="cost-box">'
            f'<strong>📊 Szacowany koszt łączny:</strong><br>'
            f'• Plik 1: {len(df1)} URLi | Plik 2: {len(df2)} URLi<br>'
            f'• Szacowane tokeny: ~{est_tokens:,}<br>'
            f'• Szacowany koszt API: {cost_display}'
            f'</div>',
            unsafe_allow_html=True
        )

    if st.button("🚀 Uruchom analizę krzyżową", disabled=(df1 is None or df2 is None)):
        try:
            texts1 = df1[column_to_embed_1].tolist()
            embeddings1 = get_embeddings(texts1, EMBEDDING_MODEL, api_key, "Plik 1")

            texts2 = df2[column_to_embed_2].tolist()
            embeddings2 = get_embeddings(texts2, EMBEDDING_MODEL, api_key, "Plik 2")

            matrix1 = np.vstack(embeddings1)
            matrix2 = np.vstack(embeddings2)

            st.write("✅ Obliczanie podobieństwa...")
            with st.spinner("Obliczanie macierzy podobieństwa..."):
                similarity_matrix_1_vs_2 = cosine_similarity(matrix1, matrix2)
            
            st.write("✅ Ładowanie modelu rerankującego...")
            reranker = load_reranker_model(RERANKER_MODEL)
            
            # --- Reranking Plik 1 -> Plik 2 ---
            st.write("✅ Reranking: Plik 1 → Plik 2...")
            progress_bar_rerank = st.progress(0, text="Reranking: Plik 1 -> Plik 2...")

            source_texts_1 = []
            candidate_texts_list_1 = []
            candidate_indices_list_1 = []
            
            for idx in range(len(df1)):
                source_texts_1.append(texts1[idx])
                candidate_indices = similarity_matrix_1_vs_2[idx].argsort()[-NUM_CANDIDATES:][::-1]
                candidate_indices_list_1.append(candidate_indices)
                candidate_texts_list_1.append([texts2[i] for i in candidate_indices])
            
            all_reranked_1 = batch_rerank(
                reranker,
                source_texts_1,
                candidate_texts_list_1,
                NUM_FINAL_RESULTS,
                progress_bar_rerank,
                "Reranking: Plik 1 -> Plik 2"
            )
            
            results_1_to_2 = []
            for idx, reranked_results in enumerate(all_reranked_1):
                candidate_indices = candidate_indices_list_1[idx]
                original_urls_candidates = df2[column_map2['url']].iloc[candidate_indices].tolist()
                top_urls = [original_urls_candidates[res['corpus_id']] for res in reranked_results]
                
                result_row = {'URL': df1[column_map1['url']].iloc[idx]}
                for i, url in enumerate(top_urls):
                    result_row[f'LINK {i+1}'] = url
                results_1_to_2.append(result_row)

            output_df_1_to_2 = pd.DataFrame(results_1_to_2)
            
            # --- Reranking Plik 2 -> Plik 1 ---
            st.write("✅ Reranking: Plik 2 → Plik 1...")
            progress_bar_rerank.progress(0, text="Reranking: Plik 2 -> Plik 1...")

            similarity_matrix_2_vs_1 = similarity_matrix_1_vs_2.T
            
            source_texts_2 = []
            candidate_texts_list_2 = []
            candidate_indices_list_2 = []
            
            for idx in range(len(df2)):
                source_texts_2.append(texts2[idx])
                candidate_indices = similarity_matrix_2_vs_1[idx].argsort()[-NUM_CANDIDATES:][::-1]
                candidate_indices_list_2.append(candidate_indices)
                candidate_texts_list_2.append([texts1[i] for i in candidate_indices])
            
            all_reranked_2 = batch_rerank(
                reranker,
                source_texts_2,
                candidate_texts_list_2,
                NUM_FINAL_RESULTS,
                progress_bar_rerank,
                "Reranking: Plik 2 -> Plik 1"
            )
            
            results_2_to_1 = []
            for idx, reranked_results in enumerate(all_reranked_2):
                candidate_indices = candidate_indices_list_2[idx]
                original_urls_candidates = df1[column_map1['url']].iloc[candidate_indices].tolist()
                top_urls = [original_urls_candidates[res['corpus_id']] for res in reranked_results]
                
                result_row = {'URL': df2[column_map2['url']].iloc[idx]}
                for i, url in enumerate(top_urls):
                    result_row[f'LINK {i+1}'] = url
                results_2_to_1.append(result_row)
            
            progress_bar_rerank.empty()
            output_df_2_to_1 = pd.DataFrame(results_2_to_1)

            st.success("🎉 Analiza krzyżowa zakończona pomyślnie!")
            
            st.subheader("Wyniki: Rekomendacje z Pliku 2 dla Pliku 1")
            st.dataframe(output_df_1_to_2.head(10))
            
            st.subheader("Wyniki: Rekomendacje z Pliku 1 dla Pliku 2")
            st.dataframe(output_df_2_to_1.head(10))
            
            # Statystyki
            st.info(f"📈 Wygenerowano {len(output_df_1_to_2)} + {len(output_df_2_to_1)} rekomendacji")
            
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
