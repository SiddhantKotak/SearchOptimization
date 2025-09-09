import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
from tqdm import tqdm
import numpy as np
import warnings
import time
import pickle
import os
warnings.filterwarnings("ignore")


# ----------------------
# CONFIG
# ----------------------
ES_HOST = "http://localhost:9200"
INDEX = "clothing_products_enhanced"
MODEL_NAME = "all-mpnet-base-v2"   # 768-dim for semantic vectors
DIMS = 768
TFIDF_DIMS = 1000  # TF-IDF vector dimensions

# ----------------------
# HELPERS
# ----------------------
def parse_attributes(val):
    """Parse p_attributes which may be dict or JSON-like string."""
    if isinstance(val, dict):
        return val
    if not isinstance(val, str) or len(val.strip()) == 0:
        return {}
    try:
        return ast.literal_eval(val)
    except Exception:
        return {}

def get_complementary_keywords(category: str):
    """
    Enhanced mapping from product category to complementary categories.
    """
    c = (category or "").lower()
    
    # Tops/Shirts mappings
    if any(k in c for k in ["shirt", "kurta", "top", "t-shirt", "blouse", "kurta set", "tunic"]):
        return ["pants", "jeans", "palazzos", "skirt", "trousers", "bottoms", "shoes", "belt", "jacket"]
    
    # Dresses mappings
    if any(k in c for k in ["dress", "gown", "frock"]):
        return ["shoes", "clutch", "belt", "jacket", "cardigan", "blazer"]
    
    # Bottoms mappings
    if any(k in c for k in ["jean", "trouser", "pant", "palazzo", "bottom", "legging", "shorts"]):
        return ["top", "shirt", "blouse", "kurta", "jacket", "blazer"]
    
    # Footwear mappings
    if any(k in c for k in ["shoe", "sandal", "heel", "boot", "slipper"]):
        return ["top", "bottom", "dress", "accessories"]
    
    # Accessories mappings
    if any(k in c for k in ["belt", "bag", "clutch", "watch", "jewelry"]):
        return ["top", "bottom", "dress", "shoes"]
    
    # default fallback
    return ["tops", "bottoms", "shoes", "accessories"]

def normalize_color_terms(color_text):
    """Normalize color terms to include variations"""
    color_text = (color_text or "").lower()
    
    # Color mappings for better matching
    color_mappings = {
        'blue': ['blue', 'navy', 'cobalt', 'royal blue', 'sky blue'],
        'red': ['red', 'crimson', 'maroon', 'burgundy', 'cherry'],
        'green': ['green', 'olive', 'emerald', 'forest green', 'mint'],
        'black': ['black', 'charcoal', 'jet black'],
        'white': ['white', 'ivory', 'cream', 'off-white'],
        'yellow': ['yellow', 'golden', 'mustard', 'lemon'],
        'pink': ['pink', 'rose', 'magenta', 'fuchsia'],
        'grey': ['grey', 'gray', 'silver', 'ash'],
        'orange': ['orange', 'coral', 'peach', 'tangerine'],
        'brown': ['brown', 'tan', 'beige', 'khaki', 'coffee'],
        'purple': ['purple', 'violet', 'lavender', 'plum']
    }
    
    # Find which color family this belongs to
    for base_color, variations in color_mappings.items():
        if any(var in color_text for var in variations):
            return base_color, variations
    
    return color_text, [color_text]

def build_similar_text(row):
    """Text used for similar_vector (semantic product representation)."""
    attrs = parse_attributes(row.get("p_attributes", {}))
    attrs_text = " ".join(str(v) for v in attrs.values() if v and v != "NA")
    pieces = [
        str(row.get("name", "")),
        str(row.get("colour", "")),
        attrs_text,
        str(row.get("description", "")),
        str(row.get("brand", ""))
    ]
    return " ".join([p for p in pieces if p]).strip()

def build_exact_text(row):
    """
    Text used for exact_vector (TF-IDF based for exact matching).
    Focus on key attributes for exact matching.
    """
    attrs = parse_attributes(row.get("p_attributes", {}))
    color = row.get("colour", "")
    category = row.get("category", "")
    brand = row.get("brand", "")
    
    # Normalize color for better matching
    base_color, color_variations = normalize_color_terms(color)
    
    # Create exact matching text with emphasis on key attributes
    pieces = [
        category.lower(),  # Primary category
        base_color,        # Normalized color
        brand.lower(),     # Brand
        str(row.get("name", "")).lower()  # Product name
    ]
    
    # Add color variations for better matching
    pieces.extend(color_variations)
    
    # Add specific attributes that are important for exact matching
    fabric = attrs.get("Top Fabric", "") or attrs.get("Bottom Fabric", "")
    if fabric:
        pieces.append(fabric.lower())
    
    pattern = attrs.get("Top Pattern", "") or attrs.get("Bottom Pattern", "")
    if pattern:
        pieces.append(pattern.lower())
    
    return " ".join([p for p in pieces if p]).strip()

def build_comp_text(row):
    """
    Text used for comp_vector (what pairs with this product).
    Enhanced for better complementary matching.
    """
    attrs = parse_attributes(row.get("p_attributes", {}))
    occ = attrs.get("Occasion", "") or attrs.get("Occasion/Usage", "")
    trend = attrs.get("Main Trend", "")
    fabric = attrs.get("Top Fabric", "") or attrs.get("Bottom Fabric", "")
    pattern = attrs.get("Top Pattern", "") or attrs.get("Bottom Pattern", "") or attrs.get("Print or Pattern Type", "")
    color = row.get("colour", "")
    category = row.get("category", "")
    targets = get_complementary_keywords(category)

    # Enhanced compatibility context
    pieces = [
        f"occasion: {occ}",
        f"style: {trend}",
        f"fabric: {fabric}",
        f"pattern: {pattern}",
        f"color_family: {normalize_color_terms(color)[0]}",
        f"season: {attrs.get('Season', '')}",
        "complements: " + " ".join(targets)
    ]
    return " ".join([p for p in pieces if p]).strip()

# ----------------------
# Enhanced Elasticsearch setup with timeout handling
# ----------------------
def create_elasticsearch_client(max_retries=3):
    """Create ES client with proper timeout settings."""
    for attempt in range(max_retries):
        try:
            es = Elasticsearch(
                ["https://localhost:9200"],
                basic_auth=("elastic", "tTj8lpPblyAsfrM-64RJ"),
                verify_certs=False,
                timeout=60,  # Increase timeout to 60 seconds
                max_timeout=120,
                retry_on_timeout=True,
                http_compress=True,
                request_timeout=60
            )
            # Test connection
            es.info()
            print(f"✅ Connected to Elasticsearch on attempt {attempt + 1}")
            return es
        except Exception as e:
            print(f"❌ Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise Exception(f"Failed to connect to Elasticsearch after {max_retries} attempts")

es = create_elasticsearch_client()

# ----------------------
# Index management with error handling
# ----------------------
def setup_index():
    """Setup ES index with proper error handling."""
    try:
        if es.indices.exists(index=INDEX):
            print(f"Deleting existing index: {INDEX}")
            es.indices.delete(index=INDEX)
            time.sleep(2)  # Wait for deletion to complete

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,  # Reduce replicas for faster indexing
                "refresh_interval": "30s",  # Less frequent refresh
                "max_result_window": 50000
            },
            "mappings": {
                "properties": {
                    "p_id": {"type": "keyword"},
                    "name": {"type": "text"},
                    "brand": {"type": "keyword"},
                    "colour": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "price": {"type": "float"},
                    "avg_rating": {"type": "float"},
                    "description": {"type": "text"},
                    "p_attributes": {"type": "text"},
                    # Semantic similarity vector (cosine similarity)
                    "similar_vector": {"type": "dense_vector", "dims": DIMS, "similarity": "cosine"},
                    # Exact matching vector (dot product similarity)
                    "exact_vector": {"type": "dense_vector", "dims": TFIDF_DIMS, "similarity": "dot_product"},
                    # Complementary matching vector (dot product similarity)
                    "comp_vector": {"type": "dense_vector", "dims": DIMS, "similarity": "dot_product"}
                }
            }
        }

        es.indices.create(index=INDEX, body=mapping)
        print(f"✅ Created index: {INDEX}")
        
    except Exception as e:
        print(f"❌ Error setting up index: {str(e)}")
        raise

setup_index()

# ----------------------
# Load and cache models
# ----------------------
def load_or_cache_models():
    """Load models and cache them for faster subsequent runs."""
    model_cache_path = "model_cache.pkl"
    tfidf_cache_path = "tfidf_cache.pkl"
    
    # Load semantic model
    print("Loading semantic model...")
    semantic_model = SentenceTransformer(MODEL_NAME)
    
    return semantic_model, None  # TF-IDF will be created fresh each time

semantic_model, tfidf_vectorizer = load_or_cache_models()

# ----------------------
# Load data
# ----------------------
print("Loading data...")
df = pd.read_csv("products.csv")
print(f"Loaded {len(df)} products")

for col in ["p_id", "name", "colour", "category", "price", "avg_rating", "description", "p_attributes", "brand"]:
    if col not in df.columns:
        df[col] = ""

# ----------------------
# Generate text representations
# ----------------------
print("Building text representations...")
similar_texts = []
exact_texts = []
comp_texts = []
ids = []

for _, row in tqdm(df.iterrows(), desc="Building texts", total=len(df)):
    similar_texts.append(build_similar_text(row))
    exact_texts.append(build_exact_text(row))
    comp_texts.append(build_comp_text(row))
    ids.append(row.get("p_id") or f"idx_{_}")

# ----------------------
# Generate vectors with progress tracking
# ----------------------
print("Generating semantic vectors...")
similar_embeddings = semantic_model.encode(similar_texts, batch_size=32, show_progress_bar=True)
comp_embeddings = semantic_model.encode(comp_texts, batch_size=32, show_progress_bar=True)

print("Generating TF-IDF vectors for exact matching...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=TFIDF_DIMS,
    ngram_range=(1, 2),
    stop_words='english',
    lowercase=True
)

exact_embeddings = tfidf_vectorizer.fit_transform(exact_texts).toarray()

# ----------------------
# Enhanced bulk indexing with retry logic
# ----------------------
def robust_bulk_index(df, ids, similar_embeddings, exact_embeddings, comp_embeddings, chunk_size=100):
    """Robust bulk indexing with error handling and retries."""
    
    def gen_actions():
        for i, (_, row) in enumerate(df.iterrows()):
            yield {
                "_index": INDEX,
                "_id": ids[i],
                "_source": {
                    "p_id": ids[i],
                    "name": row.get("name", ""),
                    "brand": row.get("brand", ""),
                    "colour": row.get("colour", ""),
                    "category": row.get("category", ""),
                    "price": float(row.get("price") or 0),
                    "avg_rating": float(row.get("avg_rating") or 0),
                    "description": row.get("description", ""),
                    "p_attributes": str(row.get("p_attributes", "")),
                    "similar_vector": similar_embeddings[i].tolist(),
                    "exact_vector": exact_embeddings[i].tolist(),
                    "comp_vector": comp_embeddings[i].tolist()
                }
            }

    print(f"Indexing {len(df)} documents with chunk size {chunk_size}...")
    success_count = 0
    error_count = 0
    
    try:
        for success, info in tqdm(
            streaming_bulk(
                es, 
                gen_actions(),
                chunk_size=chunk_size,
                max_retries=3,
                initial_backoff=2,
                max_backoff=600,
                request_timeout=120
            ),
            desc="Indexing",
            total=len(df)
        ):
            if success:
                success_count += 1
            else:
                error_count += 1
                print(f"❌ Error indexing document: {info}")
                
    except Exception as e:
        print(f"❌ Bulk indexing failed: {str(e)}")
        print("Trying with smaller chunk size...")
        # Retry with smaller chunk size
        return robust_bulk_index(df, ids, similar_embeddings, exact_embeddings, comp_embeddings, chunk_size//2)
    
    print(f"✅ Indexing completed: {success_count} success, {error_count} errors")
    
    # Refresh index
    try:
        es.indices.refresh(index=INDEX)
        print("✅ Index refreshed")
    except Exception as e:
        print(f"⚠️ Warning: Could not refresh index: {str(e)}")

# Start indexing
robust_bulk_index(df, ids, similar_embeddings, exact_embeddings, comp_embeddings)

# ----------------------
# ENHANCED SEARCH UTILITIES
# ----------------------
def _fetch_by_vector(query_vector, vector_field="similar_vector", similarity_type="cosine", fetch_k=100, min_rating=0.0, exclude_p_id=None, category_exclude=None):
    """Enhanced vector search with different similarity measures and error handling."""
    bool_query = {"filter": []}
    if min_rating > 0:
        bool_query["filter"].append({"range": {"avg_rating": {"gte": min_rating}}})

    must_not = []
    if exclude_p_id:
        must_not.append({"term": {"p_id": exclude_p_id}})
    if category_exclude:
        must_not.append({"term": {"category": category_exclude}})
    if must_not:
        bool_query["must_not"] = must_not

    # Different similarity calculations based on type
    if similarity_type == "cosine":
        script_source = f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0"
    elif similarity_type == "dot_product":
        script_source = f"dotProduct(params.query_vector, '{vector_field}')"
    else:
        script_source = f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0"

    body = {
        "size": fetch_k,
        "query": {
            "script_score": {
                "query": {"bool": bool_query},
                "script": {
                    "source": script_source,
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }

    try:
        res = es.search(index=INDEX, body=body)
        return res["hits"]["hits"]
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return []

def search_exact(query_text, top_k=10, min_rating=0.0, fetch_k=100):
    """Exact search using TF-IDF vectors and dot product similarity."""
    try:
        exact_text = build_exact_text({"name": query_text, "colour": "", "category": "", "brand": "", "p_attributes": {}})
        query_tfidf = tfidf_vectorizer.transform([exact_text]).toarray()[0]
        
        hits = _fetch_by_vector(query_tfidf.tolist(), vector_field="exact_vector", 
                               similarity_type="dot_product", fetch_k=fetch_k, min_rating=min_rating)
        
        results = []
        for hit in hits:
            s = hit["_source"]
            results.append({
                "p_id": s["p_id"],
                "name": s["name"],
                "brand": s.get("brand", ""),
                "category": s.get("category", ""),
                "colour": s.get("colour", ""),
                "price": s.get("price", 0.0),
                "avg_rating": s.get("avg_rating", 0.0),
                "score": hit["_score"]
            })
        
        results_sorted = sorted(results, key=lambda x: (x["score"], x["avg_rating"]), reverse=True)
        return results_sorted[:top_k]
        
    except Exception as e:
        print(f"❌ Exact search error: {str(e)}")
        return search_similar(query_text, top_k, min_rating, fetch_k)

def search_similar(query_text, top_k=5, min_rating=0.0, fetch_k=100):
    """Semantic similarity search using sentence transformers and cosine similarity."""
    try:
        qv = semantic_model.encode(query_text).tolist()
        hits = _fetch_by_vector(qv, vector_field="similar_vector", similarity_type="cosine",
                               fetch_k=fetch_k, min_rating=min_rating)
        
        results = []
        for hit in hits:
            s = hit["_source"]
            results.append({
                "p_id": s["p_id"],
                "name": s["name"],
                "brand": s.get("brand", ""),
                "category": s.get("category", ""),
                "colour": s.get("colour", ""),
                "price": s.get("price", 0.0),
                "avg_rating": s.get("avg_rating", 0.0),
                "score": hit["_score"]
            })
        
        results_sorted = sorted(results, key=lambda x: (x["avg_rating"], x["score"]), reverse=True)
        return results_sorted[:top_k]
        
    except Exception as e:
        print(f"❌ Similar search error: {str(e)}")
        return []

def search_complementary(query_text_or_p_id, top_k=5, min_rating=0.0, fetch_k=200):
    """Enhanced complementary search using dot product similarity."""
    try:
        source_doc = None
        
        # Try to fetch by ID first
        try:
            if isinstance(query_text_or_p_id, str):
                resp = es.get(index=INDEX, id=query_text_or_p_id, ignore=[404])
                if resp and resp.get("found"):
                    source_doc = resp["_source"]
        except Exception:
            source_doc = None

        if source_doc is None:
            # Find best matching product using exact search first, then similar
            found = search_exact(query_text_or_p_id, top_k=1, min_rating=min_rating, fetch_k=50)
            if not found:
                found = search_similar(query_text_or_p_id, top_k=1, min_rating=min_rating, fetch_k=50)
            if not found:
                return []
            
            try:
                source_doc = es.get(index=INDEX, id=found[0]["p_id"])["_source"]
            except Exception:
                return []

        source_p_id = source_doc["p_id"]
        source_category = source_doc.get("category", "")
        comp_vector = source_doc.get("comp_vector")
        
        if not comp_vector:
            # Fallback: compute comp_vector on the fly
            comp_text = build_comp_text(source_doc)
            comp_vector = semantic_model.encode(comp_text).tolist()

        # Use dot product for complementary matching
        hits = _fetch_by_vector(comp_vector, vector_field="similar_vector", similarity_type="dot_product",
                               fetch_k=fetch_k, min_rating=min_rating,
                               exclude_p_id=source_p_id, category_exclude=source_category)

        results = []
        for hit in hits:
            s = hit["_source"]
            results.append({
                "p_id": s["p_id"],
                "name": s["name"],
                "brand": s.get("brand", ""),
                "category": s.get("category", ""),
                "colour": s.get("colour", ""),
                "price": s.get("price", 0.0),
                "avg_rating": s.get("avg_rating", 0.0),
                "score": hit["_score"]
            })
        
        results_sorted = sorted(results, key=lambda x: (x["avg_rating"], x["score"]), reverse=True)
        return results_sorted[:top_k]
        
    except Exception as e:
        print(f"❌ Complementary search error: {str(e)}")
        return []

def search_hybrid(query_text, top_k=10, min_rating=0.0, exact_weight=0.7, similar_weight=0.3):
    """Hybrid search combining exact and similar search results."""
    try:
        exact_results = search_exact(query_text, top_k=top_k*2, min_rating=min_rating)
        similar_results = search_similar(query_text, top_k=top_k*2, min_rating=min_rating)
        
        # Combine and weight results
        combined_scores = {}
        
        for i, result in enumerate(exact_results):
            p_id = result["p_id"]
            rank_score = (len(exact_results) - i) / len(exact_results)
            combined_scores[p_id] = {
                "result": result,
                "score": exact_weight * rank_score
            }
        
        for i, result in enumerate(similar_results):
            p_id = result["p_id"]
            rank_score = (len(similar_results) - i) / len(similar_results)
            if p_id in combined_scores:
                combined_scores[p_id]["score"] += similar_weight * rank_score
            else:
                combined_scores[p_id] = {
                    "result": result,
                    "score": similar_weight * rank_score
                }
        
        # Sort by combined score
        final_results = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["result"] for item in final_results[:top_k]]
        
    except Exception as e:
        print(f"❌ Hybrid search error: {str(e)}")
        return []

# ----------------------
# EXAMPLE USAGE
# ----------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔹 TESTING SEARCH FUNCTIONALITY")
    print("="*80)
    
    try:
        print("\n🔹 EXACT SEARCH (TF-IDF + Dot Product) - Blue shirts including navy:")
        exact_results = search_exact("blue shirt", top_k=8, min_rating=3.0)
        for i, r in enumerate(exact_results, 1):
            print(f"{i}. {r['name']} - {r['brand']} [{r['category']}] Color: {r['colour']} (${r['price']:.2f}) ⭐{r['avg_rating']:.1f}")

        print("\n🔹 SIMILAR SEARCH (Semantic + Cosine) - Similar products:")
        similar_results = search_similar("blue shirt", top_k=5, min_rating=3.0)
        for i, r in enumerate(similar_results, 1):
            print(f"{i}. {r['name']} - {r['brand']} [{r['category']}] Color: {r['colour']} (${r['price']:.2f}) ⭐{r['avg_rating']:.1f}")

        print("\n🔹 COMPLEMENTARY SEARCH (Dot Product) - Products that go with blue shirts:")
        comp_results = search_complementary("blue shirt", top_k=5, min_rating=3.0)
        for i, r in enumerate(comp_results, 1):
            print(f"{i}. {r['name']} - {r['brand']} [{r['category']}] Color: {r['colour']} (${r['price']:.2f}) ⭐{r['avg_rating']:.1f}")

        print("\n🔹 HYBRID SEARCH - Combined exact and similar:")
        hybrid_results = search_hybrid("blue shirt", top_k=6, min_rating=3.0)
        for i, r in enumerate(hybrid_results, 1):
            print(f"{i}. {r['name']} - {r['brand']} [{r['category']}] Color: {r['colour']} (${r['price']:.2f}) ⭐{r['avg_rating']:.1f}")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
    
    print("\n" + "="*80)
    print("✅ SETUP COMPLETE - You can now use the search functions!")
    print("="*80)