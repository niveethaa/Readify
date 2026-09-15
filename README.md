# Readify

Readify is a Retrieval-Augmented Generation (RAG) system that recommends books and songs based on a user's mood, theme, or preference. It prompts you to tell you what you're feeling and it retrieves the most relevant books and songs from two real-world datasets, then uses an LLM to generate a natural, grounded recommendation with guardrails to prevent hallucinated titles or artists.

Built by: Niveetha

🔗 Try it live: readify.onrender.com 
(Note: hosted on a free tier that spins down after inactivity - the first load after idle time may take up to a minute.)

# How it works
Retrieval — User queries are vectorized using TF-IDF and matched against books (Goodreads) and songs (Spotify) datasets using cosine similarity, retrieving the top-k most textually relevant results.

Filtering — Optional filters for minimum book rating and song popularity narrow results to higher-quality matches.

Augmentation — The top-k retrieved books and songs are injected into a structured prompt as grounding context.

Generation — An LLM (via the Groq API) generates a natural-language recommendation using only the retrieved context, with explicit guardrails against inventing titles, authors, or artists not present in the data.

# Tech
Python · Pandas · scikit-learn (TF-IDF) · Groq API (LLM inference) · Gradio · Render (hosting) · Hugging Face Datasets (data hosting)

# Running locally
```bash
git clone https://github.com/yourusername/readify.git
cd readify
pip install -r requirements.txt
export GROQ_API_KEY="your-api-key-here"
python app.py
```
Datasets are downloaded automatically on first run from Hugging Face Datasets.

# Dataset sources
Goodreads Books Dataset (Kaggle)
Spotify Songs Dataset (Kaggle)

# What worked well
- Mood: and feeling-based queries (e.g. "something romantic and emotional") produced strong, relevant matches.
- Rating and popularity filters meaningfully improved recommendation quality by filtering out obscure or low-signal results.
- The system correctly handled empty queries with a clear prompt instead of returning random results.

# Known limitations
- TF-IDF retrieval is keyword-based rather than fully semantic — it can miss conceptually related matches that don't share vocabulary with the query.
- Small LLMs can still occasionally struggle with complex, multi-constraint queries (e.g. specifying exact counts of books and songs at once).
- Exact-count instructions (e.g. "3 books and 2 songs") aren't always followed precisely, since managing two separate counts across two retrieval sets adds complexity for smaller models.

# License
This project is licensed under the MIT License.
