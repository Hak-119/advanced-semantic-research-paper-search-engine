import streamlit as st
import arxiv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from transformers import pipeline
from app_advancedfeaturesOK import(
                                call_mmr_rerank, call_hybrid_selection, mmr_rerank, smart_hybrid_selection,
                                plot_diversity_comparison,plot_diversity_of_selected_papers,plot_similarity_matrix,
                                sort_by_date,compare_faiss_mmr
)

# ========== 1. Local Ollama Query Function ==========
def query_ollama(prompt, model="mistral", max_tokens=1000):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        }
    )
    data = response.json()
    return data.get("response", "No response from Ollama.")

# ========== 2. HuggingFace (Flan-T5) Query Function ==========
@st.cache_resource
def get_huggingface_pipeline(model):
    if model == "facebook/bart-large-cnn":
        return pipeline("summarization", model=model)
    else:
        return pipeline("text2text-generation", model=model)

def query_huggingface(prompt, model="google/flan-t5-large", max_tokens=1000):
    generator = get_huggingface_pipeline(model)
    
    if model == "facebook/bart-large-cnn":
        result = generator(prompt,
                           max_length=max_tokens,
                           min_length=50,
                           do_sample=False)
        return result[0]['summary_text']
    
    else:
        result = generator(prompt,
                           max_length=max_tokens,
                           min_length=50,
                           num_return_sequences=1,
                           do_sample=False,
                           num_beams=4,
                           early_stopping=True,
                           temperature=0.7,
                           top_p=0.9)
        return result[0]['generated_text']


# ========== 3. Load Models ==========
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-mpnet-base-v2')
    return embed_model  # Only returning embedding model now

# ========== 4. ArXiv Search ==========
@st.cache_data(show_spinner=False)
def search_arxiv(query, max_results=10):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = list(client.results(search))
    return results

# ========== 5. Embedding ==========
@st.cache_data(show_spinner=False)
def get_paper_embeddings(paper_texts, _embed_model):
    return _embed_model.encode(paper_texts, convert_to_tensor=False)

# ========== 6. FAISS Index ==========
@st.cache_data(show_spinner=False)
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# ========== 7. Main App ==========
def main():
    st.title("Semantic Search Engine for Academic Papers")
    st.markdown("This tool finds relevant academic papers from ArXiv and synthesizes intelligent answers.")

    # Load models
    embed_model = load_models()

    # User input
    query = st.text_input("Enter your research query:",
                          placeholder="e.g., 'Recent advances in transformer architectures'")

    # Model selection
    model_choice = st.selectbox("Choose LLM:", [
    "Mistral (Ollama)",  # Lightweight, generalist model
    "Gemma3 (Ollama)",    # Google's efficient LLM
    "Flan-T5-Large (HuggingFace)",  # Instruction-tuned LLM
    "BART-Large-CNN (HuggingFace)",    # Great for summarization tasks
])
    # Advanced settings
    with st.sidebar:
        st.header("Advanced Settings")
        with st.expander("Filters", expanded=False):
            num_papers = st.slider("Number of Papers to Display:", min_value=1, max_value=20, value=5)
            sort_by = st.selectbox("Sort by:", ["Relevance", "Latest", "Oldest"])
        with st.expander("Reranking Methods", expanded=False):
            rerank_strategy = st.selectbox("Select Reranking Strategy:", ["FAISS", "MMR", "Hybrid"])
            use_mmr = st.checkbox("Use MMR Reranking", value=False)
            use_hybrid = st.checkbox("Use Smart Hybrid Selection", value=False)
            compare_fmmr=st.checkbox("Compare FAISS with MMR", value=False)
        with st.expander("Retrieved Diversity Visualization", expanded=False):
            visualize_diversity = st.checkbox("Visualize Diversity of Ranking Methods & Selected Papers", value=False)
            VDSP=st.checkbox("Visualize Diversity of Among Selected Papers", value=False)
            VDRM=st.checkbox("Visualize Diversity of Ranking Methods", value=False)
            
    if query:
        with st.spinner("Searching for relevant papers..."):
            papers = search_arxiv(query)

            if not papers:
                st.warning("No papers found for this query.")
                return

            # Prepare paper data
            paper_texts = []
            paper_metas = []
            for paper in papers:
                text = f"Title: {paper.title}\nAuthors: {', '.join(a.name for a in paper.authors)}\n"
                text += f"Published: {paper.published.strftime('%Y-%m-%d')}\nAbstract: {paper.summary}"
                paper_texts.append(text)
                paper_metas.append({
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "published": paper.published,
                    "url": paper.entry_id,
                    "abstract": paper.summary
                })

            # Sort by date if needed
            sort_by_date(sort_by, paper_metas, paper_texts)
            
            # Embeddings and FAISS index
            embeddings = get_paper_embeddings(paper_texts, _embed_model=embed_model)
            index = create_faiss_index(np.array(embeddings))

            # Search top matches
            query_embedding = embed_model.encode(query, convert_to_tensor=False)
            D, I = index.search(np.array([query_embedding]), k=min(num_papers, len(papers))) #5
            faiss_idxs = I[0].tolist() #

            if use_mmr:
                mmr_idxs = call_mmr_rerank(query_embedding, embeddings, use_mmr=True, top_n=num_papers)
            else:
                mmr_idxs = faiss_idxs

            if use_hybrid:
                hybrid_idxs = call_hybrid_selection(query_embedding, embeddings, mmr_idxs, num_papers, use_hybrid=True)
            else:
                hybrid_idxs = mmr_idxs

            selected_idxs = {
                "FAISS": faiss_idxs,
                "MMR": mmr_idxs,
                "Hybrid": hybrid_idxs
            }[rerank_strategy]


            st.subheader("Most Relevant Papers")
            for idx in selected_idxs:
                paper = papers[idx]
                meta = paper_metas[idx]

                with st.expander(f"{meta['title']} - {', '.join(meta['authors'][:2])}..."):
                    st.markdown(f"""
                    **Published**: {meta['published'].strftime('%Y-%m-%d')}  
                    **Abstract**: {meta['abstract']}  
                    [Read Paper]({meta['url']})
                    """)

                    #cleaned abstract for flan_prompt:
                    cleaned_abstract = meta['abstract'][:1024]  # Keep it under model's max input

                    # Generate summary via the selected model
                    with st.spinner(f"Generating key findings using {model_choice}..."):
                        prompt = f"""
You are en expert academic researcher.
Summarize the key findings of the following paper in relation to the query: '{query}'.

Title: {meta['title']}
Abstract: {meta['abstract']}

Provide just 2-3 consise bullet points of the most relevant findings that encapsulates the paper's importance(with accordance to the query).
"""
                        if model_choice == "Mistral (Ollama)":
                            summary = query_ollama(prompt,model="mistral", max_tokens=300)
                        elif model_choice == "Gemma3 (Ollama)":
                            summary = query_ollama(prompt, model="gemma3:4b", max_tokens=300)
                        elif model_choice == "BART-Large-CNN (HuggingFace)":
                            summary = query_huggingface(prompt, model="facebook/bart-large-cnn", max_tokens=300)
                        elif model_choice == "Flan-T5-Large (HuggingFace)":  # HuggingFace (Flan-T5)
                            summary = query_huggingface(prompt, max_tokens=300)
                            
                        st.markdown("**Key Findings:**")
                        st.write(summary)

            # Optional comparison display: FAISS vs MMR
            if use_mmr and compare_fmmr:
                    compare_faiss_mmr(query_embedding, embeddings, papers, paper_metas, num_papers, create_faiss_index)
            # Visualize diversity of selected papers
            if visualize_diversity:
                if VDRM:
                    plot_diversity_comparison(faiss_idxs, mmr_idxs, hybrid_idxs, embeddings)
                if VDSP:
                    plot_diversity_of_selected_papers(selected_idxs, embeddings, rerank_strategy)
            
            # Synthesized answer from top 3 papers
            st.subheader("Synthesized Answer from Relevant Papers")
            with st.spinner("Analyzing papers to generate comprehensive answer..."):
                context = "\n\n".join([paper_texts[idx] for idx in selected_idxs[:3]])
                qa_prompt = f"""
You are an expert academic researcher and a professional answer synthesizer to any given context of query.
Based on the following research papers, provide a comprehensive answer to the query: '{query}'.

Papers:
{context}

Answer should:
- Be clear and relevant to the query and the paper context.
- Cite specific papers when mentioning findings
- Highlight areas of consensus and disagreement
- Mention any limitations or open questions
- Should leave the user satisfied that he got the answer he was looking for.
- 2-3 small paras.
Answer:
"""
                if model_choice == "Mistral (Ollama)":
                    answer = query_ollama(qa_prompt,model="mistral", max_tokens=1000)
                elif model_choice == "Gemma3 (Ollama)":
                    answer = query_ollama(qa_prompt, model="gemma3:4b", max_tokens=1000)
                elif model_choice == "BART-Large-CNN (HuggingFace)":
                    answer = query_huggingface(qa_prompt, model="facebook/bart-large-cnn", max_tokens=1000)
                elif model_choice == "Flan-T5-Large (HuggingFace)":  # HuggingFace (Flan-T5)
                    answer = query_huggingface(qa_prompt, max_tokens=1000)
                    
                st.write(answer)

                # References
                st.markdown("---")
                st.markdown("**References**")
                for idx in I[0][:3]:
                    meta = paper_metas[idx]
                    st.markdown(f"- [{meta['title']}]({meta['url']}) by {', '.join(meta['authors'][:2])} et al.")

if __name__ == "__main__":
    main()
