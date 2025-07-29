import streamlit as st
import arxiv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1. MMR Reranking ==========
def call_mmr_rerank(query_embedding, embeddings, use_mmr=True, top_n=10):
    if use_mmr:
        lambda_slider = st.slider("MMR Diversity vs Relevance (λ)", 0.0, 1.0, 0.5)
        base_idxs = mmr_rerank(query_embedding, embeddings, lambda_score=lambda_slider, top_n=top_n)
    else:
        index = create_faiss_index(np.array(embeddings))
        D, I = index.search(np.array([query_embedding]), k=10)
        base_idxs = I[0].tolist()
    return base_idxs

def mmr_rerank(query_emb, doc_embs, lambda_score=0.5, top_n=5,):
    selected = []
    indices = list(range(len(doc_embs)))
    doc_embs = np.array(doc_embs)
    query_emb = np.array(query_emb).reshape(1, -1)

    sim_to_query = cosine_similarity(doc_embs, query_emb).flatten()
    sim_between_docs = cosine_similarity(doc_embs)

    while len(selected) < top_n and indices:
        if not selected:
            idx = np.argmax(sim_to_query)
        else:
            mmr_scores = []
            for i in indices:
                max_sim = max([sim_between_docs[i][j] for j in selected])
                mmr = lambda_score * sim_to_query[i] - (1 - lambda_score) * max_sim
                mmr_scores.append((i, mmr))
            idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected.append(idx)
        indices.remove(idx)
    return selected

# ========== 2. Smart Hybrid Selection ==========
def call_hybrid_selection(query_embedding, embeddings, base_idxs, num_papers, use_hybrid=True):
    if use_hybrid:
        lambda_slider = st.slider("Balance (Relevance vs Diversity)", 0.0, 1.0, 0.5)
        selected_idxs = smart_hybrid_selection(query_embedding, embeddings, base_idxs, lambda_score=lambda_slider, final_n=num_papers)
    else:
        selected_idxs = base_idxs[:num_papers]
    return selected_idxs

def smart_hybrid_selection(query_emb, doc_embs, base_indices, lambda_score=0.5, final_n=5):
    query_emb = np.array(query_emb).reshape(1, -1)
    doc_embs = np.array(doc_embs)

    # Get base candidate embeddings
    candidates = [doc_embs[i] for i in base_indices]
    sim_to_query = cosine_similarity(candidates, query_emb).flatten()
    sim_between = cosine_similarity(candidates)

    selected = []
    indices = list(range(len(base_indices)))

    while len(selected) < final_n and indices:
        if not selected:
            idx = int(np.argmax(sim_to_query))
        else:
            hybrid_scores = []
            for i in indices:
                max_div = max([sim_between[i][j] for j in selected]) if selected else 0
                score = lambda_score * sim_to_query[i] - (1 - lambda_score) * max_div
                hybrid_scores.append((i, score))
            idx, _ = max(hybrid_scores, key=lambda x: x[1])
        selected.append(idx)
        indices.remove(idx)

    return [base_indices[i] for i in selected]


# ========== 3. Plot Similarity Matrix ==========
def plot_similarity_matrix(embs, labels=None,title="Similarity Matrix"):
    sim = cosine_similarity(embs)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(sim, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f",
                cmap="coolwarm", cbar=True, square=True, ax=ax)
    ax.set_title(title, fontsize=14)
    st.pyplot(fig)

def plot_diversity_comparison(faiss_idxs, mmr_idxs, hybrid_idxs, embeddings):
    with st.expander("📊 Diversity Comparison: FAISS vs MMR vs Hybrid", expanded=False):
        st.markdown("## 📊 Diversity Comparison: FAISS vs MMR vs Hybrid")

        cols = st.columns(3)

        strategy_data = {
            "FAISS": faiss_idxs,
            "MMR": mmr_idxs,
            "Hybrid": hybrid_idxs
        }

        for col, (method, idxs) in zip(cols, strategy_data.items()):
            with col:
                st.markdown(f"**{method}**")
                selected_embs = [embeddings[i] for i in idxs]
                labels = [f"P{i+1}" for i in range(len(idxs))]
                plot_similarity_matrix(selected_embs, labels=labels, title=f"{method} Similarity")

def plot_diversity_of_selected_papers(selected_idxs, embeddings, rerank_strategy):
    with st.expander("📊 Diversity of Selected Papers", expanded=False):
        st.markdown("## 🔍 Diversity of Selected Papers")
        if rerank_strategy == "FAISS":
            st.markdown("**Selection Method: FAISS Top-K**")
        elif rerank_strategy == "MMR":
            st.markdown("**Selection Method: MMR Reranking**")
        elif rerank_strategy == "Hybrid":
            st.markdown("**Selection Method: Smart Hybrid Selection**")
        selected_embs = [embeddings[i] for i in selected_idxs]
        labels = [f"P{i+1}" for i in range(len(selected_idxs))]
        plot_similarity_matrix(selected_embs, labels=labels, title="Semantic Similarity Among Selected Papers")
            
# ========== 4. Sort by date ==========(something changed)
def sort_by_date(sort_by, paper_metas, paper_texts):
    # Sort by date if needed
    if sort_by == "Latest":
        paper_metas, paper_texts = zip(*sorted(zip(paper_metas, paper_texts), key=lambda x: x[0]["published"], reverse=True))
    elif sort_by == "Oldest":
        paper_metas, paper_texts = zip(*sorted(zip(paper_metas, paper_texts), key=lambda x: x[0]["published"]))
    else:
        paper_metas, paper_texts = list(paper_metas), list(paper_texts)

# ========== 5.Optional comparison display: FAISS vs MMR ==========
def compare_faiss_mmr(query_embedding, embeddings, papers, paper_metas, num_papers, create_faiss_index):
    with st.expander("🔍 Compare: MMR vs FAISS Selection", expanded=False):
        # Run both independently
        faiss_index = create_faiss_index(np.array(embeddings))
        D, I = faiss_index.search(np.array([query_embedding]), k=min(num_papers, len(papers)))
        faiss_selected_idxs = I[0].tolist()
        mmr_selected_idxs = mmr_rerank(query_embedding, embeddings, lambda_score=0.5, top_n=num_papers)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔁 FAISS (Top-K Similarity)")
            for idx in faiss_selected_idxs:
                meta = paper_metas[idx]
                sim_score = cosine_similarity([query_embedding], [embeddings[idx]])[0][0]
                st.markdown(f"- [{meta['title']}]({meta['url']})  \n  **Sim Score**: `{sim_score:.4f}`")
        with col2:
            st.markdown("### 🔀 MMR (Relevance + Diversity)")
            for idx in mmr_selected_idxs:
                meta = paper_metas[idx]
                sim_score = cosine_similarity([query_embedding], [embeddings[idx]])[0][0]
                st.markdown(f"- [{meta['title']}]({meta['url']})  \n  **Sim Score**: `{sim_score:.4f}`")






