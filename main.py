"""
FAST SEMANTIC SEARCH STREAMLIT APP
Purpose: Lightning-fast product search using pre-computed semantic mappings
Usage: streamlit run streamlit_search_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

# CONFIGURATION
st.set_page_config(
    page_title="Fast Semantic Search",
    page_icon="⚡",
    layout="wide"
)

class FastSemanticSearch:
    """Lightning-fast semantic search engine"""
    
    def __init__(self, mappings: Dict[str, Dict[str, float]], products: List[Dict]):
        """Initialize search engine with pre-computed mappings"""
        self.mappings = mappings
        self.products = products
        self.product_lookup = {}
        self.word_to_products = defaultdict(list)
        self.build_search_indices()
    
    def build_search_indices(self):
        """Build optimized search indices"""
        print("🔧 Building search indices...")
        
        # Create product lookup table
        for product in self.products:
            product_key = f"{product['name']}_{product['brand']}_{product['category']}"
            self.product_lookup[product_key] = product
        
        # Build inverted index: word -> list of relevant products
        for product_key, word_scores in self.mappings.items():
            for word, score in word_scores.items():
                if score > 0.2:  # Only include meaningful associations
                    self.word_to_products[word.lower()].append({
                        'product_key': product_key,
                        'score': score
                    })
        
        print(f"✅ Built indices for {len(self.word_to_products)} searchable words")
    
    def search(self, query: str, max_results: int = 20) -> Tuple[List[Dict], float]:
        """Perform lightning-fast semantic search"""
        start_time = time.time()
        
        # Extract meaningful words from query
        query_words = self.extract_query_words(query)
        
        if not query_words:
            return [], 0.0
        
        # Aggregate relevance scores for each product
        product_scores = defaultdict(float)
        word_matches = defaultdict(list)
        
        for word in query_words:
            word_lower = word.lower()
            if word_lower in self.word_to_products:
                for product_info in self.word_to_products[word_lower]:
                    product_key = product_info['product_key']
                    score = product_info['score']
                    product_scores[product_key] += score
                    word_matches[product_key].append(word)
        
        # Normalize scores by query length and apply boost for multiple matches
        for product_key in product_scores:
            base_score = product_scores[product_key] / len(query_words)
            match_bonus = len(word_matches[product_key]) * 0.1  # Bonus for matching multiple words
            product_scores[product_key] = min(base_score + match_bonus, 1.0)
        
        # Sort by relevance score
        sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to result objects
        results = []
        for product_key, score in sorted_products[:max_results]:
            if product_key in self.product_lookup and score > 0.15:
                product = self.product_lookup[product_key].copy()
                product['relevance_score'] = score
                product['matched_words'] = word_matches[product_key]
                product['match_count'] = len(word_matches[product_key])
                results.append(product)
        
        search_time = time.time() - start_time
        return results, search_time
    
    def extract_query_words(self, query: str) -> List[str]:
        """Extract meaningful words from search query"""
        # Clean and split query
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Remove common stop words but keep domain-specific terms
        stop_words = {
            'i', 'me', 'my', 'am', 'is', 'are', 'was', 'were', 'a', 'an', 'the', 
            'and', 'or', 'but', 'for', 'with', 'at', 'by', 'from', 'to', 'of', 
            'in', 'on', 'up', 'down', 'out', 'off', 'over', 'under'
        }
        
        # Keep meaningful words (length > 2 and not stop words)
        meaningful_words = [
            word for word in words 
            if len(word) > 2 and word not in stop_words
        ]
        
        return meaningful_words

class ProductDatabase:
    """Product database manager"""
    
    def __init__(self, csv_path: str = "quick_commerce_sample_skus_classified.csv"):
        """Initialize product database"""
        self.csv_path = csv_path
        self.products = []
        self.load_products()
    
    def load_products(self):
        """Load products from CSV"""
        try:
            df = pd.read_csv(self.csv_path)
            df = df.fillna(0)
            
            # Clean price column
            if 'MRP' in df.columns:
                df['MRP'] = pd.to_numeric(df['MRP'], errors='coerce').fillna(0)
            
            # Convert to product list
            for idx, row in df.iterrows():
                product = {
                    "name": str(row.get("Product Name", f"Product {idx}")),
                    "brand": str(row.get("Brand", "Unknown")),
                    "size": str(row.get("Size", "Standard")),
                    "price": float(row.get("MRP", 0)),
                    "category": str(row.get("Category", "General")),
                    "rating": min(4.0 + np.random.random() * 1.0, 5.0)  # Random rating for demo
                }
                self.products.append(product)
            
            print(f"✅ Loaded {len(self.products)} products from CSV")
            
        except Exception as e:
            print(f"❌ Error loading products: {e}")
            self.products = []
    
    def get_categories(self):
        """Get unique product categories"""
        return sorted(list(set(p['category'] for p in self.products)))
    
    def get_brands(self):
        """Get unique brands"""
        return sorted(list(set(p['brand'] for p in self.products)))

@st.cache_data
def load_semantic_mappings(mapping_file: str = "product_semantic_mappings.pkl"):
    """Load pre-computed semantic mappings"""
    try:
        with open(mapping_file, 'rb') as f:
            mappings = pickle.load(f)
        return mappings
    except Exception as e:
        st.error(f"❌ Error loading mappings: {e}")
        return None

def handle_file_upload():
    """Handle CSV file upload"""
    st.header("📁 Upload Product Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            with open("quick_commerce_sample_skus_classified.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ File uploaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error uploading file: {e}")

def display_search_results(results: List[Dict], search_time: float, query: str):
    """Display search results with rich formatting"""
    if not results:
        st.warning("🤔 No results found")
        st.info("💡 Try different keywords or check spelling")
        return
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Results Found", len(results))
    with col2:
        st.metric("Search Time", f"{search_time*1000:.1f}ms")
    with col3:
        avg_relevance = np.mean([r['relevance_score'] for r in results])
        st.metric("Avg Relevance", f"{avg_relevance:.1%}")
    with col4:
        high_relevance = sum(1 for r in results if r['relevance_score'] > 0.6)
        st.metric("High Relevance", high_relevance)
    
    st.success(f"⚡ Found {len(results)} products in {search_time*1000:.1f}ms")
    
    # Display results
    for idx, product in enumerate(results):
        relevance = product['relevance_score']
        matched_words = product.get('matched_words', [])
        match_count = product.get('match_count', 0)
        
        # Relevance indicator
        if relevance > 0.7:
            color = "🟢"
            level = "Excellent"
        elif relevance > 0.5:
            color = "🟡" 
            level = "Good"
        elif relevance > 0.3:
            color = "🟠"
            level = "Fair"
        else:
            color = "🔴"
            level = "Weak"
        
        with st.expander(
            f"{color} #{idx+1} {product['name']} - ₹{product['price']:.2f} ({level} Match)",
            expanded=idx < 3
        ):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Brand:** {product['brand']}")
                st.write(f"**Category:** {product['category']}")
                st.write(f"**Size:** {product['size']}")
                
                if matched_words:
                    st.success(f"**Matched Words:** {', '.join(matched_words)}")
                
                # Show why it matches
                if match_count > 1:
                    st.info(f"🎯 Matches {match_count} aspects of your query")
            
            with col2:
                st.metric("Price", f"₹{product['price']:.2f}")
                st.metric("Rating", f"{product['rating']:.1f}/5")
            
            with col3:
                st.metric("Relevance", f"{relevance:.1%}")
                st.metric("Word Matches", match_count)
                
                if st.button(f"Add to Cart", key=f"cart_{idx}"):
                    st.success("Added to cart! 🛒")

def show_analytics(mappings: Dict, products: List[Dict]):
    """Show semantic mapping analytics"""
    st.header("📊 Semantic Search Analytics")
    
    # Basic stats
    total_products = len(mappings)
    total_associations = sum(len(words) for words in mappings.values())
    avg_associations = total_associations / total_products if total_products > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mapped Products", total_products)
    with col2:
        st.metric("Total Associations", f"{total_associations:,}")
    with col3:
        st.metric("Avg per Product", f"{avg_associations:.1f}")
    
    # Word frequency analysis
    st.subheader("🔥 Most Common Semantic Words")
    
    word_counts = Counter()
    for product_words in mappings.values():
        word_counts.update(product_words.keys())
    
    top_words = word_counts.most_common(20)
    if top_words:
        words_df = pd.DataFrame(top_words, columns=['Word', 'Product Count'])
        st.bar_chart(words_df.set_index('Word'))
    
    # Category analysis
    st.subheader("📂 Products by Category")
    category_counts = Counter(p['category'] for p in products)
    if category_counts:
        st.bar_chart(dict(category_counts))

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("⚡ Lightning-Fast Semantic Search")
    st.markdown("---")
    st.markdown("**Instant product search using pre-computed AI mappings**")
    
    # Check for required files
    mapping_file = "product_semantic_mappings.pkl"
    csv_file = "quick_commerce_sample_skus_classified.csv"
    
    if not os.path.exists(mapping_file):
        st.error("❌ Semantic mappings not found!")
        st.info("Please run the mapping creator script first:")
        st.code("python create_semantic_mappings.py", language="bash")
        
        if not os.path.exists(csv_file):
            handle_file_upload()
        return
    
    if not os.path.exists(csv_file):
        st.warning("⚠️ Product CSV file not found")
        handle_file_upload()
        return
    
    # Initialize system
    with st.spinner("🚀 Loading semantic search engine..."):
        try:
            # Load data
            mappings = load_semantic_mappings(mapping_file)
            if not mappings:
                st.error("❌ Failed to load semantic mappings")
                return
            
            db = ProductDatabase(csv_file)
            if not db.products:
                st.error("❌ Failed to load products from CSV")
                return
            
            # Initialize search engine
            search_engine = FastSemanticSearch(mappings, db.products)
            
            # Store in session state
            if 'search_engine' not in st.session_state:
                st.session_state.search_engine = search_engine
                st.session_state.database = db
                st.session_state.mappings = mappings
            
        except Exception as e:
            st.error(f"❌ Error initializing search system: {e}")
            return
    
    st.success(f"✅ Search engine ready! {len(db.products)} products indexed")
    
    # System stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Products", len(db.products))
    with col2:
        st.metric("Semantic Words", len(st.session_state.search_engine.word_to_products))
    with col3:
        file_size_mb = os.path.getsize(mapping_file) / (1024*1024)
        st.metric("Index Size", f"{file_size_mb:.1f} MB")
    with col4:
        total_associations = sum(len(words) for words in mappings.values())
        st.metric("Total Associations", f"{total_associations:,}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        # SEARCH TAB
        st.header("🔍 Semantic Product Search")
        
        # Sidebar filters
        with st.sidebar:
            st.header("🎯 Filters")
            
            # Price range
            prices = [p['price'] for p in db.products if p['price'] > 0]
            if prices:
                min_price, max_price = min(prices), max(prices)
                price_range = st.slider(
                    "Price Range (₹)",
                    min_value=int(min_price),
                    max_value=int(max_price),
                    value=(int(min_price), int(max_price * 0.8))
                )
            else:
                price_range = (0, 1000)
            
            # Category filter
            categories = db.get_categories()
            selected_categories = st.multiselect(
                "Categories",
                options=categories,
                default=categories
            )
            
            # Brand filter  
            brands = db.get_brands()
            selected_brands = st.multiselect(
                "Brands",
                options=brands[:20],  # Limit for performance
                default=[]
            )
            
            # Max results
            max_results = st.slider("Max Results", 5, 50, 15)
        
        # Example queries
        st.subheader("💡 Try These Examples:")
        example_cols = st.columns(3)
        
        examples = [
            "i need something to eat quick",
            "i am going for a trip", 
            "school supplies for my kid",
            "something for menstrual pain",
            "baby needs diaper",
            "headache medicine",
            "party snacks for friends",
            "cooking dinner fast",
            "comfortable sleep"
        ]
        
        selected_example = None
        for i, example in enumerate(examples):
            col_idx = i % 3
            with example_cols[col_idx]:
                if st.button(f"📝 {example}", key=f"example_{i}"):
                    selected_example = example
        
        # Search input
        search_query = st.text_input(
            "🔍 What are you looking for?",
            value=selected_example if selected_example else "",
            placeholder="Describe what you need in natural language...",
            help="Use natural language - the AI understands context and intent!"
        )
        
        # Search button
        if st.button("⚡ Search", type="primary", use_container_width=True) or selected_example:
            if search_query:
                with st.spinner("🔍 Searching..."):
                    # Perform search
                    results, search_time = st.session_state.search_engine.search(
                        search_query, max_results=max_results
                    )
                    
                    # Apply filters
                    if selected_categories:
                        results = [r for r in results if r['category'] in selected_categories]
                    
                    if selected_brands:
                        results = [r for r in results if r['brand'] in selected_brands]
                    
                    # Price filter
                    results = [r for r in results if price_range[0] <= r['price'] <= price_range[1]]
                
                # Display results
                st.markdown("---")
                display_search_results(results, search_time, search_query)
            else:
                st.warning("⚠️ Please enter a search query")
    
    with tab2:
        # ANALYTICS TAB
        show_analytics(st.session_state.mappings, st.session_state.database.products)
        
        # Sample mappings
        st.subheader("🔍 Sample Product Mappings")
        
        # Select a product to examine
        product_names = [p['name'] for p in st.session_state.database.products[:20]]
        selected_product = st.selectbox("Select a product to examine:", product_names)
        
        if selected_product:
            # Find the product
            product = next(p for p in st.session_state.database.products if p['name'] == selected_product)
            product_key = f"{product['name']}_{product['brand']}_{product['category']}"
            
            if product_key in st.session_state.mappings:
                word_scores = st.session_state.mappings[product_key]
                sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
                
                st.write(f"**{selected_product}** is semantically associated with:")
                
                # Show top associations
                cols = st.columns(3)
                for i, (word, score) in enumerate(sorted_words[:15]):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.metric(word.title(), f"{score:.2f}")
                
                # Word cloud style display
                st.subheader("📊 All Associations")
                word_data = pd.DataFrame(sorted_words, columns=['Word', 'Relevance Score'])
                st.dataframe(word_data, use_container_width=True)
    
    with tab3:
        # SETTINGS TAB
        st.header("⚙️ System Settings")
        
        # File information
        st.subheader("📁 File Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Semantic Mappings File**")
            if os.path.exists(mapping_file):
                file_size = os.path.getsize(mapping_file) / (1024*1024)
                mod_time = datetime.fromtimestamp(os.path.getmtime(mapping_file))
                st.write(f"Size: {file_size:.1f} MB")
                st.write(f"Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.error("❌ File not found")
        
        with col2:
            st.info("**Product CSV File**")
            if os.path.exists(csv_file):
                file_size = os.path.getsize(csv_file) / (1024*1024)
                mod_time = datetime.fromtimestamp(os.path.getmtime(csv_file))
                st.write(f"Size: {file_size:.1f} MB")
                st.write(f"Modified: {mod_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.error("❌ File not found")
        
        # System performance
        st.subheader("⚡ Performance Metrics")
        
        if st.button("🧪 Run Performance Test"):
            test_queries = [
                "quick food",
                "baby diaper", 
                "school pen",
                "trip bag",
                "headache pain"
            ]
            
            total_time = 0
            for query in test_queries:
                start = time.time()
                results, _ = st.session_state.search_engine.search(query, max_results=10)
                query_time = time.time() - start
                total_time += query_time
                st.write(f"'{query}': {query_time*1000:.1f}ms ({len(results)} results)")
            
            avg_time = total_time / len(test_queries)
            st.success(f"✅ Average search time: {avg_time*1000:.1f}ms")
        
        # Rebuild option
        st.subheader("🔄 Maintenance")
        st.warning("⚠️ Only recreate mappings if you have new products or want to update the semantic associations")
        
        if st.button("🗑️ Clear Cache & Restart"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Cache cleared! Please refresh the page.")
        
        st.info("💡 To add new products:\n1. Update your CSV file\n2. Run: `python create_semantic_mappings.py`\n3. Restart this app")

if __name__ == "__main__":
    main()