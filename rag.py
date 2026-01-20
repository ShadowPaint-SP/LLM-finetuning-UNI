"""
RAG Data Augmentation Module
Retrieves relevant context from Wikivoyage corpus and augments training data.
Updated with Advanced RAG techniques: Sentence Chunking, Hybrid Search with Keyword Extraction, 
Cross-Encoder Re-ranking, and MMR Diversification.
"""

import os
import re
import torch # type: ignore
import faiss # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from tqdm import tqdm # type: ignore
from typing import List, Dict, Tuple, Optional
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, CrossEncoder # type: ignore
from rank_bm25 import BM25Okapi # type: ignore
import nltk # type: ignore
from nltk.tokenize import sent_tokenize, word_tokenize # type: ignore
from nltk.corpus import stopwords # type: ignore

class WikivoyageRAG:
    """Advanced RAG system for retrieving relevant context from Wikivoyage corpus"""
    
    def __init__(self, 
                 wikivoyage_xml_path: str = "datasets/wikivoyage.xml",
                 cache_dir: str = "rag_cache",
                 chunk_size: int = 800,       # Smaller chunks for precision
                 chunk_overlap: int = 200,    # Good overlap to preserve context across boundaries
                 use_dense: bool = True,
                 use_sparse: bool = True,
                 use_reranker: bool = True,
                 device: str = None):
        """
        Initialize RAG system
        
        Args:
            wikivoyage_xml_path: Path to Wikivoyage XML file
            cache_dir: Directory to cache embeddings and indices
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap size for sliding window in characters
            use_dense: Enable dense retrieval (E5)
            use_sparse: Enable sparse retrieval (BM25)
            use_reranker: Enable Cross-Encoder re-ranking
            device: Device for E5/Reranker ('cuda', 'cpu', or None for auto)
        """
        
        self.wikivoyage_path = wikivoyage_xml_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_dense = use_dense
        self.use_sparse = use_sparse
        self.use_reranker = use_reranker
        
        # Auto-detect device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device:
            self.device = device
        
        print(f"RAG will use device: {self.device}")
        
        # Storage for corpus
        self.wikivoyage = []
        self.documents = []
        
        # Models and indices
        self.dense_model = None
        self.reranker_model = None
        self.faiss_index = None
        self.bm25 = None
        self.tokenized_docs = None
        
        # Initialize NLTK data
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
    
    def _download_nltk_data(self):
        """Download necessary NLTK data if missing"""
        resources = [
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/stopwords', 'stopwords'),
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
            ('taggers/universal_tagset', 'universal_tagset')
        ]
        
        for resource_path, download_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                print(f"Downloading NLTK resource: {download_name}")
                nltk.download(download_name, quiet=True)

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into overlapping passages respecting sentence boundaries.
        Avoids cutting cultural context in the middle of a sentence.
        """
        text = text.strip()
        if not text:
            return []
        
        sentences = sent_tokenize(text)
        passages = []
        current_chunk = []
        current_len = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sent_len = len(sentence)
            
            # Case 1: Single sentence is larger than chunk size (rare but possible)
            if sent_len > self.chunk_size:
                # Flush current chunk if it exists
                if current_chunk:
                    passages.append(" ".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                # Add the huge sentence as its own chunk
                passages.append(sentence)
                i += 1
                continue

            # Case 2: Add sentence to current chunk
            current_chunk.append(sentence)
            current_len += sent_len
            
            # Case 3: Chunk is full
            if current_len >= self.chunk_size:
                passages.append(" ".join(current_chunk))
                
                # Backtrack to create overlap for the next chunk
                # We keep the last N sentences that fit within chunk_overlap
                overlap_len = 0
                backtrack_idx = i
                new_start_chunk = []
                
                while backtrack_idx >= 0:
                    prev_sent = sentences[backtrack_idx]
                    if overlap_len + len(prev_sent) > self.chunk_overlap:
                        break
                    new_start_chunk.insert(0, prev_sent) # Prepend
                    overlap_len += len(prev_sent)
                    backtrack_idx -= 1
                    
                current_chunk = new_start_chunk
                current_len = overlap_len
            
            i += 1
            
        # Add any remaining text
        if current_chunk:
            passages.append(" ".join(current_chunk))
            
        return passages
    
    def extract_search_keywords(self, query: str) -> str:
        """
        Extracts keywords from a natural language question for BM25.
        Keeps Nouns, Adjectives, Numbers, and Proper Nouns.
        """
        # Tokenize and POS tag
        tokens = word_tokenize(query)
        pos_tags = nltk.pos_tag(tokens, tagset='universal')
        
        keywords = []
        for word, tag in pos_tags:
            word_lower = word.lower()
            # Filter punctuation and stop words
            if word.isalnum() and word_lower not in self.stop_words:
                # Keep Nouns (NOUN), Adjectives (ADJ), Numbers (NUM), or capitalized words (Proper Nouns)
                if tag in ['NOUN', 'ADJ', 'NUM'] or word[0].isupper():
                    keywords.append(word)
        
        # Fallback if aggressive filtering removed everything
        if not keywords:
            return query 
            
        return " ".join(keywords)

    def load_wikivoyage(self, force_reload: bool = False):
        """Load and preprocess Wikivoyage corpus with caching"""
        # We use v3 suffix to distinguish from previous simple-chunked caches
        corpus_cache = self.cache_dir / "wikivoyage_corpus_v3.pkl"
        
        if corpus_cache.exists() and not force_reload:
            print("Loading cached Wikivoyage corpus (Sentence Chunking)...")
            import pickle
            with open(corpus_cache, 'rb') as f:
                data = pickle.load(f)
                self.wikivoyage = data['wikivoyage']
                self.documents = data['documents']
            print(f"✓ Loaded {len(self.wikivoyage)} passages from cache")
            return
        
        print("Loading Wikivoyage from XML...")
        if not os.path.exists(self.wikivoyage_path):
            raise FileNotFoundError(
                f"Wikivoyage XML not found at {self.wikivoyage_path}\n"
                f"Download from: https://datashare.tu-dresden.de/s/RAWB2wDdMnwkBg3"
            )
        
        ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
        passage_id = 0
        
        context = ET.iterparse(self.wikivoyage_path, events=("start", "end"))
        _, root = next(context)
        
        for event, elem in tqdm(context, desc="Parsing XML"):
            if event == "end" and elem.tag.endswith("page"):
                title = elem.find("mw:title", ns)
                revision = elem.find("mw:revision", ns)
                text = revision.find("mw:text", ns) if revision is not None else None
                
                title = title.text if title is not None else None
                text = text.text if text is not None else ""
                
                # Skip empty pages and redirects
                if not text or text.strip().upper().startswith("#REDIRECT"):
                    elem.clear()
                    root.clear()
                    continue
                
                # Clean Wiki markup
                text_clean = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
                text_clean = re.sub(r"\[\[[A-Za-z]+:[^\]]+\]\]", "", text_clean)
                text_clean = re.sub(r"\[\[[^\|\]]*\|([^\]]+)\]\]", r"\1", text_clean)
                text_clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text_clean)
                text_clean = re.sub(r"'{2,}", "", text_clean)
                
                # Split into passages using NEW Sentence Splitter
                passages = self._split_into_sentences(text_clean)
                
                for p in passages:
                    self.wikivoyage.append({
                        "id": passage_id,
                        "title": title,
                        "passage": p.strip()
                    })
                    passage_id += 1
                
                elem.clear()
                root.clear()
        
        self.documents = [item["passage"] for item in self.wikivoyage]
        
        # Cache the corpus
        import pickle
        with open(corpus_cache, 'wb') as f:
            pickle.dump({
                'wikivoyage': self.wikivoyage,
                'documents': self.documents
            }, f)
        
        print(f"✓ Loaded {len(self.wikivoyage)} passages from Wikivoyage")
    
    def build_sparse_index(self, force_rebuild: bool = False):
        """Build BM25 index for sparse retrieval"""
        if not self.use_sparse:
            return
        
        bm25_cache = self.cache_dir / "bm25_tokenized_v3.pkl"
        
        if bm25_cache.exists() and not force_rebuild:
            print("Loading cached BM25 index...")
            import pickle
            with open(bm25_cache, 'rb') as f:
                self.tokenized_docs = pickle.load(f)
            self.bm25 = BM25Okapi(self.tokenized_docs)
            print("✓ BM25 index loaded")
            return
        
        print("Building BM25 index...")
        self.tokenized_docs = [
            word_tokenize(doc.lower()) 
            for doc in tqdm(self.documents, desc="Tokenizing")
        ]
        
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        import pickle
        with open(bm25_cache, 'wb') as f:
            pickle.dump(self.tokenized_docs, f)
        
        print("✓ BM25 index built")
    
    def build_dense_index(self, force_rebuild: bool = False):
        """Build FAISS index for dense retrieval"""
        if not self.use_dense:
            return
        
        faiss_path = self.cache_dir / "faiss_e5_v3.index"
        
        # Load model if needed
        if self.dense_model is None:
            print(f"Loading E5 model on {self.device}...")
            self.dense_model = SentenceTransformer("intfloat/e5-base-v2", device=self.device)

        if faiss_path.exists() and not force_rebuild:
            print("Loading cached FAISS index...")
            self.faiss_index = faiss.read_index(str(faiss_path))
            print(f"✓ FAISS index loaded ({self.faiss_index.ntotal} vectors)")
            return
        
        print("Building FAISS index...")
        
        # Encode documents in batches
        batch_size = 64 if self.device == 'cuda' else 16
        embeddings = []
        
        for i in tqdm(range(0, len(self.documents), batch_size), desc="Encoding"):
            batch = self.documents[i:i + batch_size]
            # E5 requires "passage: " prefix for documents
            batch_formatted = ["passage: " + d for d in batch]
            
            emb = self.dense_model.encode(
                batch_formatted, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                show_progress_bar=False,
                device=self.device
            )
            embeddings.append(emb)
        
        embeddings = np.vstack(embeddings).astype("float32")
        
        # Build FAISS index (Inner Product for Cosine Similarity on normalized vectors)
        d = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(embeddings)
        
        # Save index
        faiss.write_index(self.faiss_index, str(faiss_path))
        
        print(f"✓ FAISS index built ({self.faiss_index.ntotal} vectors)")

    def load_reranker(self):
        """Load the Cross-Encoder for Re-ranking"""
        if self.use_reranker and self.reranker_model is None:
            print(f"Loading Cross-Encoder Re-ranker on {self.device}...")
            # High quality, fast reranker
            self.reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)
    
    def initialize(self, force_rebuild: bool = False):
        """Initialize all components"""
        self.load_wikivoyage(force_reload=force_rebuild)
        
        if not self.documents:
            print("(!) No documents loaded. Please check XML path.")
            return

        if self.use_sparse:
            self.build_sparse_index(force_rebuild=force_rebuild)
        
        if self.use_dense:
            self.build_dense_index(force_rebuild=force_rebuild)
            
        if self.use_reranker:
            self.load_reranker()
        
        print("\n✓ RAG system ready (Advanced Mode)!")
    
    def search_bm25(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """Sparse retrieval using Keyword Extraction + BM25"""
        if not self.use_sparse or self.bm25 is None:
            return []
        
        # KEY IMPROVEMENT: Extract keywords instead of using full question
        search_query = self.extract_search_keywords(query)
        tokenized_query = word_tokenize(search_query.lower())
        
        scores = self.bm25.get_scores(tokenized_query)
        
        if k == -1:
            k = len(scores)
        
        top_k = scores.argsort()[-k:][::-1]
        return [(idx, float(scores[idx])) for idx in top_k]
    
    def search_e5(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        """Dense retrieval using E5 (Full Question)"""
        if not self.use_dense or self.faiss_index is None:
            return []
        
        if k == -1:
            k = self.faiss_index.ntotal
        
        # E5 requires "query: " prefix for questions
        query_formatted = "query: " + query
        q_vec = self.dense_model.encode(
            [query_formatted], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype("float32")
        
        distances, indices = self.faiss_index.search(q_vec, k)
        return list(zip(indices[0], distances[0]))
    
    def mmr_selection(self, query_str: str, retrieved_indices: List[int], k: int = 5, lambda_param: float = 0.5) -> List[int]:
        """
        Maximal Marginal Relevance (MMR) to diversify results.
        Selects documents that are relevant to query but different from already selected docs.
        """
        if not self.use_dense or len(retrieved_indices) == 0:
            return retrieved_indices[:k]

        # 1. Encode Query
        query_fmt = "query: " + query_str
        query_emb = self.dense_model.encode([query_fmt], convert_to_numpy=True, normalize_embeddings=True)

        # 2. Encode Retrieved Documents
        # Reconstruct vectors from FAISS (fastest) or re-encode (fallback)
        try:
            doc_embs = np.array([self.faiss_index.reconstruct(int(idx)) for idx in retrieved_indices])
        except Exception:
            # Fallback for indices that don't support reconstruction
            docs = ["passage: " + self.documents[idx] for idx in retrieved_indices]
            doc_embs = self.dense_model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)

        # 3. Calculate MMR
        selected_indices = []
        candidate_indices = list(range(len(retrieved_indices)))

        for _ in range(min(k, len(retrieved_indices))):
            best_score = -float('inf')
            best_idx_in_candidates = -1

            for i in candidate_indices:
                # Sim(Query, Doc)
                relevance = np.dot(query_emb, doc_embs[i].T).item()
                
                # Max Sim(Doc, Selected)
                if not selected_indices:
                    diversity = 0
                else:
                    selected_embs = doc_embs[selected_indices]
                    diversity = np.max(np.dot(doc_embs[i], selected_embs.T))
                
                # MMR Score = Lambda * Relevance - (1 - Lambda) * Diversity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx_in_candidates = i

            if best_idx_in_candidates != -1:
                selected_indices.append(best_idx_in_candidates)
                candidate_indices.remove(best_idx_in_candidates)
        
        return [retrieved_indices[i] for i in selected_indices]

    def retrieve(self, query: str, k: int = 3, use_mmr: bool = True, mmr_diversity: float = 0.3) -> List[Dict[str, str]]:
        """
        Retrieve relevant passages using advanced hybrid pipeline:
        1. Fetch candidates via Sparse (Keywords) and Dense (Question)
        2. Merge and Re-rank using Cross-Encoder
        3. Diversify using MMR
        """
        # 1. Fetch Candidates (fetch 5x k to allow for reranking/filtering)
        candidate_k = k * 5 
        
        # Sparse Search (Auto-extracts keywords)
        sparse_res = self.search_bm25(query, candidate_k) if self.use_sparse else []
        
        # Dense Search (Uses full question)
        dense_res = self.search_e5(query, candidate_k) if self.use_dense else []
        
        # Merge results (Union by index)
        # Using a dictionary to handle duplicates across methods
        candidates = {} 
        for idx, _ in dense_res:
            candidates[idx] = 0.0 # Score placeholder
        for idx, _ in sparse_res:
            candidates[idx] = 0.0
            
        candidate_indices = list(candidates.keys())
        
        if not candidate_indices:
            return []
        
        # 2. Re-ranking (Crucial step)
        if self.use_reranker and self.reranker_model:
            # Prepare pairs: [Query, Document Text]
            doc_texts = [self.documents[idx] for idx in candidate_indices]
            pairs = [[query, doc] for doc in doc_texts]
            
            # Predict scores
            scores = self.reranker_model.predict(pairs)
            
            # Sort by new Cross-Encoder scores
            scored_candidates = sorted(
                list(zip(candidate_indices, scores)), 
                key=lambda x: x[1], 
                reverse=True
            )
            candidate_indices = [idx for idx, _ in scored_candidates]
        
        # 3. Diversification (MMR)
        if use_mmr and self.use_dense:
            # Only apply MMR on the top sorted results from reranker (e.g. top 20)
            # lambda_param: 1.0 = Pure Relevance, 0.0 = Pure Diversity. 
            # We invert mmr_diversity so 0.3 diversity => lambda 0.7
            final_indices = self.mmr_selection(
                query, 
                candidate_indices[:20], 
                k=k, 
                lambda_param=1.0 - mmr_diversity
            )
        else:
            final_indices = candidate_indices[:k]
            
        # 4. Format Results
        retrieved = []
        for idx in final_indices:
            item = self.wikivoyage[idx]
            retrieved.append({
                'title': item['title'],
                'passage': item['passage'],
                'score': 0.0 # Score is less meaningful after MMR/Rerank mix
            })
            
        return retrieved


def augment_training_data_with_rag(
    training_examples: List[Dict],
    rag: WikivoyageRAG,
    task_type: str,
    k: int = 3,
    add_context_to_user: bool = True,
    debug: bool = False,
    batch_retrieval: bool = True,
    max_context_length: int = 400
) -> List[Dict]:
    """
    Augment training data with RAG-retrieved context using caching.
    """
    augmented = []
    
    # Cache to avoid retrieving same question multiple times
    retrieval_cache = {}
    
    for idx, example in enumerate(tqdm(training_examples, desc="RAG Augmentation", disable=not batch_retrieval)):
        messages = example['messages']
        
        # Extract the question from the first user message
        user_content = messages[0]['content']
        
        # Clean question extraction
        if task_type == 'mcq':
            # Try to grab just the question sentence if possible
            if 'Analyze the following question' in user_content:
                # Heuristic for the 'analysis' prompt type
                try:
                    start = user_content.find("'") + 1
                    end = user_content.find("'", start)
                    if start > 0 and end > start:
                        question = user_content[start:end]
                    else:
                        question = user_content
                except:
                    question = user_content
            else:
                question = user_content.split('?')[0] + '?'
        else:
            # SAQ cleaning
            question = user_content.split('Provide ONLY')[0].strip()
            question = question.split('Target Score:')[0].strip()
        
        # Check cache first
        cache_key = question.strip().lower()
        if cache_key in retrieval_cache:
            retrieved = retrieval_cache[cache_key]
        else:
            # Retrieve relevant context with Advanced RAG
            # We enable MMR to ensure we get diverse cultural facts
            retrieved = rag.retrieve(
                question, 
                k=k, 
                use_mmr=True, 
                mmr_diversity=0.3
            )
            retrieval_cache[cache_key] = retrieved
        
        if not retrieved:
            augmented.append(example)
            continue
        
        # Build context string
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            passage = doc['passage'][:max_context_length]
            if len(doc['passage']) > max_context_length:
                passage += "..."
            context_parts.append(f"[Doc {i}] {doc['title']}: {passage}")
        context_str = "\n\n".join(context_parts)
        
        # Create augmented example
        new_example = {key: val for key, val in example.items()}
        new_messages = []
        
        if add_context_to_user:
            # Add context to user message
            enhanced_content = (
                f"Context Information:\n{context_str}\n\n"
                f"Question: {user_content}"
            )
            new_messages.append({
                "role": "user",
                "content": enhanced_content
            })
        else:
            # Add context as system message
            new_messages.append({
                "role": "system",
                "content": f"Use the following context to answer the user's question:\n{context_str}"
            })
            new_messages.append(messages[0])
        
        # Keep all other messages (assistant response, etc.)
        new_messages.extend(messages[1:])
        
        new_example['messages'] = new_messages
        augmented.append(new_example)
        
        if debug and idx < 3:
            print(f"\n{'='*60}")
            print(f"Example {idx + 1} ({task_type}):")
            print(f"Question: {question[:100]}...")
            print(f"Context: {context_str[:200]}...")
            print(f"{'='*60}\n")
    
    return augmented


def setup_rag_system(wikivoyage_path: str = "datasets/wikivoyage.xml",
                     cache_dir: str = "rag_cache",
                     force_rebuild: bool = False,
                     rag_method: str = "hybrid") -> WikivoyageRAG:
    """
    Setup and initialize RAG system
    """
    rag = WikivoyageRAG(
            wikivoyage_xml_path=wikivoyage_path,
            cache_dir=cache_dir,
            use_dense=(rag_method in ["dense", "hybrid"]),
            use_sparse=(rag_method in ["sparse", "hybrid"]),
            device=None
        )
        
    rag.initialize(force_rebuild=force_rebuild)
    return rag