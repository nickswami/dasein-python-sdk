import os, sys

DATA_URL = "https://storage.googleapis.com/dasein-473321-artifacts/examples/movies.zip"

if "streamlit" not in sys.modules:
    import time, zipfile, pathlib, urllib.request, pandas as pd
    from dasein import Client

    client = Client(api_key=os.environ["DASEIN_API_KEY"])

    if not any(i.get("name") == "movies" and i.get("status") == "active" for i in client.list_indexes()):
        data_dir = pathlib.Path(__file__).resolve().parent / "data"
        zip_path = data_dir / "movies.zip"
        if not zip_path.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading movie data from {DATA_URL} ...")
            urllib.request.urlretrieve(DATA_URL, zip_path)
            print("Done.")
        z = zipfile.ZipFile(zip_path)
        df = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith(".csv"))),
                         usecols=["id", "title", "tagline", "overview", "keywords", "release_date",
                                  "vote_count", "vote_average", "poster_path", "genres",
                                  "original_language", "status"])
        df = df[(df.vote_count >= 100) & df.overview.notna() & df.poster_path.notna() & (df.status == "Released")]
        texts = (df.title.astype(str) + ". " + df.tagline.fillna("") + ". "
                 + df.overview.astype(str) + ". " + df.genres.fillna("") + ". " + df.keywords.fillna(""))
        texts = texts.str.replace(". . ", ". ", regex=False).str.strip()
        yrs = pd.to_numeric(df.release_date.astype(str).str[:4], errors="coerce").fillna(0).astype(int)
        docs = [{"id": str(r), "text": t, "metadata": {"title": str(ti), "year": int(y), "rating": float(ra),
                 "genre": g, "poster": str(p), "language": str(la)}} for r, t, ti, y, ra, g, p, la in
                zip(df.id, texts, df.title, yrs, df.vote_average.fillna(0),
                    df.genres.fillna("").str.split(",").str[0].str.strip(), df.poster_path, df.original_language)]

        idx = client.create_index("movies", index_type="hybrid", model="bge-large-en-v1.5")
        idx.upsert(docs)
        while idx.status().status != "active":
            time.sleep(5)

    os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", __file__,
              "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"])

import streamlit as st
from dasein import Client


@st.cache_resource
def _idx():
    c = Client(api_key=os.environ["DASEIN_API_KEY"])
    return c.get_index(next(i["index_id"] for i in c.list_indexes()
                            if i.get("name") == "movies" and i.get("status") == "active"))


st.set_page_config(page_title="Movie Vibe Search", layout="wide")
st.title("Find movies by vibe")
q = st.text_input("Describe a movie...", placeholder="dystopian future where machines control humans")

with st.sidebar:
    alpha = st.slider("Semantic vs BM25", 0.0, 1.0, 0.5, 0.05, help="0 = pure semantic, 1 = pure keyword")
    yr = st.slider("Year", 1900, 2026, (1900, 2026))
    mr = st.slider("Min rating", 0.0, 10.0, 0.0, 0.5)
    gf = st.multiselect("Genre", ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
         "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance",
         "Science Fiction", "Thriller", "War", "Western"])

if q:
    ix = _idx()
    f = {}
    if yr != (1900, 2026): f["year"] = {"$gte": yr[0], "$lte": yr[1]}
    if mr > 0: f["rating"] = {"$gte": mr}
    if gf: f["genre"] = {"$in": gf}

    res = ix.query(text=q, top_k=10, mode="hybrid", alpha=alpha, filter=f or None,
                   include_metadata=True, include_text=True)

    embed_ms = res.embed_us / 1000
    search_ms = res.search_us / 1000
    server_ms = res.server_total_us / 1000
    network_ms = max(res.round_trip_ms - server_ms, 0)
    c = st.columns(4)
    c[0].metric("Total", f"{res.round_trip_ms:.0f} ms")
    c[1].metric("Embed (GPU)", f"{embed_ms:.0f} ms")
    c[2].metric("Search (Dasein)", f"{search_ms:.1f} ms")
    c[3].metric("Network", f"{network_ms:.0f} ms")
    st.divider()

    top = res[0].score if res else 1
    for r in res:
        m = r.metadata or {}
        pct = r.score / top * 100
        l, rt = st.columns([1, 4])
        with l:
            if m.get("poster"):
                st.image(f"https://image.tmdb.org/t/p/w200{m['poster']}", width=120)
        with rt:
            st.markdown(f"**{m.get('title', r.id)}** ({m.get('year', '?')})  \n"
                        f"{m.get('rating', '—')} · {m.get('genre', '')} · relevance {pct:.0f}%")
            txt = getattr(r, "text", "") or ""
            _, _, blurb = txt.partition(". ")
            if blurb:
                st.caption(blurb[:250])

    info = ix.status()
    with st.sidebar:
        st.divider()
        st.header("Index stats")
        if info.ram_bytes:
            st.metric("RAM", f"{info.ram_bytes / 1024**2:.0f} MB")
        st.metric("Vectors", f"{info.vector_count:,}")
        st.metric("Dims", info.dim)
        st.caption(f"{info.index_type} · {info.model_id}")
