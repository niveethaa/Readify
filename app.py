"""
Readify - Book & Music Recommender
A Retrieval-Augmented Generation (RAG) system that recommends books and songs
based on a user's mood, theme, or preference.

Built by: Niveetha

Tech stack: Python, Pandas, sentence-transformers, ChromaDB, Groq API, Gradio
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import gradio as gr

# ---------------------------------------------------------------------------
# Part 0: Download datasets from Hugging Face Datasets if not present locally
# ---------------------------------------------------------------------------

HF_DATASET_BASE = "https://huggingface.co/datasets/niveetha/ReadifyDataset/resolve/main"
BOOKS_FILE = "goodreads_top100_from1980to2023_final.csv"
SONGS_FILE = "spotify_songs.csv"


def ensure_dataset(filename):
    """Download a dataset file from Hugging Face if it doesn't already exist locally."""
    if not os.path.exists(filename):
        print(f"Downloading {filename} from Hugging Face Datasets...")
        url = f"{HF_DATASET_BASE}/{filename}"
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")


ensure_dataset(BOOKS_FILE)
ensure_dataset(SONGS_FILE)

# ---------------------------------------------------------------------------
# Part 1: Load and clean datasets
# ---------------------------------------------------------------------------

df_books = pd.read_csv(BOOKS_FILE)
df_songs = pd.read_csv(SONGS_FILE)

# Sample down the datasets to reduce memory usage during embedding and retrieval.
# This keeps the app lightweight enough to run on free-tier hosting (512MB RAM)
# while still providing a diverse, representative set of books and songs.
BOOKS_SAMPLE_SIZE = 2000
SONGS_SAMPLE_SIZE = 3000

if len(df_books) > BOOKS_SAMPLE_SIZE:
    df_books = df_books.sample(n=BOOKS_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

if len(df_songs) > SONGS_SAMPLE_SIZE:
    df_songs = df_songs.sample(n=SONGS_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# Keep only the most useful columns for retrieval
df_books = df_books[
    ["title", "authors", "language", "description", "genres", "rating_score", "num_ratings"]
].copy()

df_songs = df_songs[
    [
        "track_name",
        "track_artist",
        "lyrics",
        "playlist_genre",
        "playlist_subgenre",
        "language",
        "track_popularity",
    ]
].copy()

# Clean books
df_books = df_books.dropna(subset=["description"])
df_books["title"] = df_books["title"].astype(str).str.strip()
df_books["authors"] = df_books["authors"].astype(str).str.strip()
df_books["description"] = df_books["description"].astype(str).str.strip()
df_books["genres"] = df_books["genres"].astype(str).str.strip()
df_books["language"] = df_books["language"].fillna("Unknown").astype(str).str.strip()
df_books = df_books.drop_duplicates(subset=["title", "authors"]).reset_index(drop=True)

# Clean songs
df_songs = df_songs.dropna(subset=["lyrics"])
df_songs["track_name"] = df_songs["track_name"].astype(str).str.strip()
df_songs["track_artist"] = df_songs["track_artist"].astype(str).str.strip()
df_songs["lyrics"] = df_songs["lyrics"].astype(str).str.strip()
df_songs["playlist_genre"] = df_songs["playlist_genre"].astype(str).str.strip()
df_songs["playlist_subgenre"] = df_songs["playlist_subgenre"].astype(str).str.strip()
df_songs["language"] = df_songs["language"].fillna("Unknown").astype(str).str.strip()
df_songs = df_songs.drop_duplicates(subset=["track_name", "track_artist"]).reset_index(drop=True)

# Build combined text fields for embeddings
df_books["combined_text"] = (
    "Book Title: " + df_books["title"].fillna("").astype(str) + ". "
    "Author: " + df_books["authors"].fillna("").astype(str) + ". "
    "Language: " + df_books["language"].fillna("").astype(str) + ". "
    "Genres: " + df_books["genres"].fillna("").astype(str) + ". "
    "Description: " + df_books["description"].fillna("").astype(str) + ". "
    "Rating Score: " + df_books["rating_score"].fillna(0).astype(str) + ". "
    "Number of Ratings: " + df_books["num_ratings"].fillna(0).astype(str) + "."
)

df_songs["combined_text"] = (
    "Song Title: " + df_songs["track_name"].fillna("").astype(str) + ". "
    "Artist: " + df_songs["track_artist"].fillna("").astype(str) + ". "
    "Language: " + df_songs["language"].fillna("").astype(str) + ". "
    "Genre: " + df_songs["playlist_genre"].fillna("").astype(str) + ". "
    "Subgenre: " + df_songs["playlist_subgenre"].fillna("").astype(str) + ". "
    "Lyrics: " + df_songs["lyrics"].fillna("").astype(str) + ". "
    "Popularity: " + df_songs["track_popularity"].fillna(0).astype(str) + "."
)

# Short text used for display only (avoids overwhelming the LLM with full lyrics)
df_songs["short_text"] = (
    "Song Title: " + df_songs["track_name"].fillna("").astype(str) + ". "
    "Artist: " + df_songs["track_artist"].fillna("").astype(str) + ". "
    "Genre: " + df_songs["playlist_genre"].fillna("").astype(str) + ". "
    "Subgenre: " + df_songs["playlist_subgenre"].fillna("").astype(str) + "."
)

# ---------------------------------------------------------------------------
# Part 2: Embeddings + ChromaDB retrieval
# ---------------------------------------------------------------------------

embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

book_embeddings = embedding_model.encode(df_books["combined_text"].tolist(), show_progress_bar=True)
song_embeddings = embedding_model.encode(df_songs["combined_text"].tolist(), show_progress_bar=True)

client = chromadb.Client()

try:
    client.delete_collection(name="books_collection")
    client.delete_collection(name="songs_collection")
except Exception:
    pass

books_collection = client.create_collection(name="books_collection")
songs_collection = client.create_collection(name="songs_collection")

books_collection.add(
    ids=df_books.index.astype(str).tolist(),
    documents=df_books["combined_text"].tolist(),
    metadatas=df_books[["rating_score", "num_ratings"]].to_dict(orient="records"),
    embeddings=book_embeddings.tolist(),
)

# Songs dataset can exceed ChromaDB's max batch size, so it's added in chunks
song_ids = df_songs.index.astype(str).tolist()
song_documents = df_songs["combined_text"].tolist()
song_metadatas = df_songs[["track_popularity"]].to_dict(orient="records")
song_emb_list = song_embeddings.tolist()

for start in range(0, len(song_ids), 5000):
    end = start + 5000
    songs_collection.add(
        ids=song_ids[start:end],
        documents=song_documents[start:end],
        metadatas=song_metadatas[start:end],
        embeddings=song_emb_list[start:end],
    )


def retrieve_relevant_records(query, k=5, book_filter=None, song_filter=None):
    """Retrieve the top-k most semantically similar books and songs for a query."""
    query_embedding = embedding_model.encode([query])

    book_results = books_collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
        where=book_filter if book_filter else None,
    )
    song_results = songs_collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k,
        where=song_filter if song_filter else None,
    )

    book_ids = [int(i) for i in book_results["ids"][0]]
    song_ids_matched = [int(i) for i in song_results["ids"][0]]

    book_rows = df_books.loc[book_ids]
    song_rows = df_songs.loc[song_ids_matched]

    book_context = "\n\n".join(book_results["documents"][0])
    song_context = "\n\n".join(df_songs.loc[song_ids_matched]["short_text"].tolist())

    return book_context, song_context, book_rows, song_rows


# ---------------------------------------------------------------------------
# Part 3: LLM generation via Groq (swapped in from local Ollama for deployment)
# ---------------------------------------------------------------------------

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = "qwen/qwen3.6-27b"


def generate_answer(query, book_context, song_context):
    """Given a user query and retrieved context, ask the LLM to generate a recommendation."""
    prompt = f"""
    You are Readify, a book and music recommendation assistant to help find great books and songs.

    Based ONLY on the following retrieved books and songs, answer the user's request.
    As long as the request refers to books or music more generally, and RETRIEVED BOOKS and RETRIEVED SONGS are not "empty", at least mention one book from RETRIEVED BOOKS and one song from RETRIEVED SONGS as a recommendation.
    Always start your response with a warm and exciting opening line like "Great choice!" or "I have the perfect recommendations for you!" before giving the recommendations.
    Write in a friendly and engaging way and add at least two sentences, maximum three sentences, explaining why each recommended book and song is relevant to the request.
    Always separate your response into two sections Books: and then Songs:
    You are allowed to say "No match found" but only as a last resort if the retrieved books and songs do not match the request.
    Do NOT make up any book titles, author names, song titles, or artist names that are not in the retrieved lists below.
    Do NOT copy or repeat any of the retrieved descriptions, lyrics, or metadata in your response.

    RETRIEVED BOOKS:
    {book_context}

    RETRIEVED SONGS:
    {song_context}

    USER REQUEST: {query}
    """

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return completion.choices[0].message.content


def rag_query(query, k=5, book_filter=None, song_filter=None):
    """Full RAG pipeline: Retrieve -> Augment -> Generate."""
    if len(query.strip()) == 0:
        return "Please enter a query to get recommendations!", pd.DataFrame(), pd.DataFrame()

    book_context_raw, song_context_raw, book_rows, song_rows = retrieve_relevant_records(
        query, k, book_filter, song_filter
    )

    book_context = book_context_raw if len(book_context_raw.strip()) > 0 else "empty"
    song_context = song_context_raw if len(song_context_raw.strip()) > 0 else "empty"

    answer = generate_answer(query, book_context, song_context)
    return answer, book_rows, song_rows


# ---------------------------------------------------------------------------
# Part 4: Gradio interface
# ---------------------------------------------------------------------------

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap');

.gradio-container {
    background: linear-gradient(135deg, #fe6a93, #343344, #01b66b) !important;
    min-height: 100vh !important;
    font-family: 'Lato', sans-serif !important;
}
.block, .gr-box, .gr-panel, .gr-form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
textarea {
    background: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}
button {
    background: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
table {
    background: rgba(0, 0, 0, 0.4) !important;
    border-radius: 8px !important;
}
thead tr th {
    background: rgba(0, 0, 0, 0.6) !important;
    color: white !important;
}
tbody tr td {
    background: rgba(0, 0, 0, 0.3) !important;
    color: white !important;
}
h1 {
    font-family: 'Playfair Display', serif !important;
    color: white !important;
}
h2, h3, h4, p, label, span {
    color: white !important;
}
footer { display: none !important; }
"""


def gradio_rag(query, min_rating, min_popularity, k):
    if len(query.strip()) == 0:
        return "Please enter a query to get recommendations!", None, None

    book_filter = {"rating_score": {"$gte": float(min_rating)}}
    song_filter = {"track_popularity": {"$gte": int(min_popularity)}}

    answer, book_rows, song_rows = rag_query(
        query=query, k=int(k), book_filter=book_filter, song_filter=song_filter
    )

    book_display = book_rows.reset_index(drop=True)[["title", "authors", "rating_score", "genres"]]
    song_display = song_rows.reset_index(drop=True)[
        ["track_name", "track_artist", "track_popularity", "playlist_genre"]
    ]

    return answer.strip(), book_display, song_display


with gr.Blocks(title="Readify - Book & Music Recommender", css=custom_css) as demo:
    gr.Markdown(
        """
        # 📚🎵 Readify
        ### Your personal book and music recommender
        Tell us what you are in the mood for and we will find the perfect books and songs for you.
        """
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="What are you in the mood for?",
            placeholder="e.g. I want a book and a song that is romantic and emotional?",
            lines=2,
        )

    with gr.Row():
        min_rating_slider = gr.Slider(
            minimum=float(df_books["rating_score"].min()),
            maximum=float(df_books["rating_score"].max()),
            value=3.5,
            step=0.1,
            label="Minimum Book Rating (out of 5)",
        )
        min_popularity_slider = gr.Slider(
            minimum=int(df_songs["track_popularity"].min()),
            maximum=int(df_songs["track_popularity"].max()),
            value=50,
            step=1,
            label="Minimum Song Popularity (out of 100)",
        )

    with gr.Row():
        k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-k results to retrieve")

    run_button = gr.Button("🔍 Find my books and songs!")

    gr.Markdown("### ✨ Recommendation")
    answer_box = gr.Markdown(label="Recommendation")

    gr.Markdown("### 📚 Matched Books")
    books_table = gr.Dataframe(
        headers=["Title", "Author", "Rating", "Genres"],
        datatype=["str", "str", "number", "str"],
        label="Matched Books",
        interactive=False,
    )

    gr.Markdown("### 🎵 Matched Songs")
    songs_table = gr.Dataframe(
        headers=["Song", "Artist", "Popularity", "Genre"],
        datatype=["str", "str", "number", "str"],
        label="Matched Songs",
        interactive=False,
    )

    run_button.click(
        fn=gradio_rag,
        inputs=[query_box, min_rating_slider, min_popularity_slider, k_slider],
        outputs=[answer_box, books_table, songs_table],
    )

    gr.Markdown(
        """
        **Tip:** Adjust the sliders to broaden or narrow your results.
        The more specific your query, the better the recommendations!
        """
    )

if __name__ == "__main__":
    # Render provides the PORT environment variable; default to 7860 for local runs
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
