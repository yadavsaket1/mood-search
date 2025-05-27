"""
SEMANTIC MAPPING WORD EXPANDER
Purpose: Add new words to existing semantic mappings without reprocessing everything
Usage: python expand_semantic_mappings.py
"""

import pickle
import pandas as pd
import json
import os
import time
import threading
import queue
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

class WordExpander:
    """Efficiently add new words to existing semantic mappings"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_workers: int = 4):
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        self.client = openai.OpenAI(api_key=api_key)
        self.progress_queue = queue.Queue()
        
        # Rate limiting
        self.request_delay = 0.15  # Conservative delay for stability
        self.request_times = []
        
        print(f"🔧 Word Expander initialized with {max_workers} workers")
    
    def load_existing_mappings(self, pkl_file: str) -> Dict[str, Dict[str, float]]:
        """Load existing semantic mappings"""
        try:
            with open(pkl_file, 'rb') as f:
                mappings = pickle.load(f)
            print(f"✅ Loaded existing mappings for {len(mappings)} products")
            return mappings
        except FileNotFoundError:
            print(f"❌ File {pkl_file} not found!")
            return {}
        except Exception as e:
            print(f"❌ Error loading mappings: {e}")
            return {}
    
    def get_existing_words(self, mappings: Dict[str, Dict[str, float]]) -> Set[str]:
        """Extract all words that already exist in mappings"""
        existing_words = set()
        for product_mappings in mappings.values():
            existing_words.update(product_mappings.keys())
        return existing_words
    
    def identify_missing_words(self, new_words: List[str], existing_words: Set[str]) -> List[str]:
        """Find words that don't exist in current mappings"""
        missing_words = [word for word in new_words if word not in existing_words]
        print(f"📊 Analysis:")
        print(f"   New words provided: {len(new_words)}")
        print(f"   Already in mappings: {len(new_words) - len(missing_words)}")
        print(f"   Need to process: {len(missing_words)}")
        return missing_words
    
    def rate_limit_wait(self):
        """Simple rate limiting"""
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= 80:  # Conservative limit
            sleep_time = 60 - (current_time - self.request_times[0]) + 1
            if sleep_time > 0:
                print(f"   ⏸️ Rate limit pause: {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.request_times.append(current_time)
        time.sleep(self.request_delay)
    
    def analyze_words_for_product(self, product_key: str, product_info: str, new_words: List[str]) -> Dict[str, float]:
        """Analyze new words for a single product"""
        try:
            self.rate_limit_wait()
            
            words_str = ", ".join(new_words)
            
            prompt = f"""
Product: {product_info}

Rate each word's relevance to this product (0.0-1.0):
{words_str}

Consider user search intent:
- "monday blues" → coffee, energy drinks, comfort food
- "feeling lazy" → instant meals, easy prep
- "date night" → premium items, romantic context
- "fitness" → healthy options, protein
- "rainy day" → warm beverages, comfort items

Rate >0.2 if word could be in a search query for this product.

JSON format only:
{{"word1": 0.8, "word2": 0.0, "word3": 0.5}}
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Rate product-word relevance for search queries. JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Filter meaningful scores
            filtered_result = {word: score for word, score in result.items() if score > 0.15}
            
            # Progress update
            self.progress_queue.put({
                'type': 'product_done',
                'product_key': product_key,
                'new_words': len(filtered_result)
            })
            
            return filtered_result
            
        except Exception as e:
            print(f"   ❌ Error processing {product_key}: {e}")
            return {}
    
    def expand_mappings_parallel(self, mappings: Dict[str, Dict[str, float]], new_words: List[str]) -> Dict[str, Dict[str, float]]:
        """Add new words to existing mappings using parallel processing"""
        
        print(f"\n🚀 Expanding mappings with {len(new_words)} new words")
        print(f"📦 Processing {len(mappings)} products")
        
        # Start progress monitor
        monitor_thread = threading.Thread(target=self.progress_monitor, daemon=True)
        monitor_thread.start()
        
        # Create product info strings for API
        product_infos = {}
        for product_key in mappings.keys():
            # Parse product key (name_brand_category format)
            parts = product_key.split('_')
            if len(parts) >= 3:
                name = parts[0]
                brand = parts[1] 
                category = '_'.join(parts[2:])
                product_infos[product_key] = f"{name} ({brand}) - {category}"
            else:
                product_infos[product_key] = product_key
        
        # Process products in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for product_key, product_info in product_infos.items():
                future = executor.submit(
                    self.analyze_words_for_product, 
                    product_key, 
                    product_info, 
                    new_words
                )
                futures.append((future, product_key))
            
            # Collect results and update mappings
            completed = 0
            for future, product_key in futures:
                try:
                    new_word_scores = future.result()
                    if new_word_scores:
                        # Add new words to existing mappings
                        mappings[product_key].update(new_word_scores)
                    completed += 1
                    
                    if completed % 10 == 0:
                        print(f"   ✅ Completed {completed}/{len(mappings)} products")
                        
                except Exception as e:
                    print(f"   ❌ Failed to process {product_key}: {e}")
        
        print(f"✅ Expansion complete! Updated {len(mappings)} products")
        return mappings
    
    def progress_monitor(self):
        """Monitor progress"""
        total_new_words = 0
        while True:
            try:
                update = self.progress_queue.get(timeout=2.0)
                if update['type'] == 'product_done':
                    total_new_words += update['new_words']
                    if update['new_words'] > 0:
                        print(f"   📝 {update['product_key']}: +{update['new_words']} words")
            except queue.Empty:
                continue
            except:
                break

def get_mood_search_words() -> List[str]:
    """Get comprehensive list of mood/search words commonly used in quick commerce"""
    
    mood_words = [
        # Emotional states
        "monday", "blues", "monday-blues", "tired", "exhausted", "sleepy", "drowsy",
        "friday", "party", "celebration", "weekend", "fun", "excitement", "energy",
        "lazy", "sluggish", "unmotivated", "easy", "simple", "effortless", "quick",
        "sad", "down", "depressed", "comfort", "soothing", "healing", "uplifting",
        "stressed", "anxious", "worried", "calming", "relaxing", "peaceful", "zen",
        "happy", "joyful", "cheerful", "bright", "positive", "optimistic", "vibrant",
        "romantic", "love", "date", "special", "intimate", "passionate", "cozy",
        "nostalgic", "memories", "childhood", "traditional", "classic", "vintage",
        
        # Time-based contexts
        "morning", "breakfast", "wake-up", "start", "begin", "fresh", "new-day",
        "afternoon", "lunch", "midday", "work", "office", "meeting", "productivity",
        "evening", "dinner", "sunset", "unwind", "relax", "family-time",
        "night", "late", "midnight", "sleep", "bedtime", "rest", "recovery",
        "weekend", "saturday", "sunday", "leisure", "free-time", "holiday",
        "rush", "hurry", "urgent", "immediate", "emergency", "last-minute",
        
        # Activity contexts
        "study", "exam", "focus", "concentration", "brain", "mental", "academic",
        "workout", "gym", "fitness", "exercise", "training", "strength", "cardio",
        "cooking", "kitchen", "recipe", "ingredients", "homemade", "fresh", "meal",
        "travel", "trip", "journey", "adventure", "portable", "convenient", "mobile",
        "work", "office", "professional", "business", "formal", "corporate",
        "home", "family", "domestic", "household", "personal", "private", "cozy",
        "social", "friends", "gathering", "sharing", "together", "community",
        "outdoor", "nature", "fresh-air", "garden", "picnic", "camping", "hiking",
        
        # Health & wellness
        "healthy", "wellness", "nutrition", "organic", "natural", "pure", "clean",
        "sick", "illness", "medicine", "treatment", "remedy", "healing", "recovery",
        "pain", "headache", "stomachache", "fever", "cold", "flu", "infection",
        "energy", "boost", "vitality", "strength", "power", "endurance", "stamina",
        "diet", "weight", "slim", "fit", "muscle", "protein", "vitamins", "minerals",
        "beauty", "skincare", "haircare", "grooming", "hygiene", "cleanliness",
        "pregnant", "baby", "infant", "child", "kids", "family", "parenting",
        
        # Weather & seasons
        "rainy", "monsoon", "wet", "umbrella", "waterproof", "indoor", "cozy",
        "sunny", "bright", "summer", "hot", "cool", "refreshing", "hydrating",
        "winter", "cold", "warm", "heating", "comfort", "layers", "protection",
        "humid", "sticky", "uncomfortable", "fresh", "dry", "cool", "ventilation",
        
        # Special occasions
        "birthday", "anniversary", "valentine", "mothers-day", "fathers-day",
        "diwali", "christmas", "new-year", "holi", "eid", "festival", "celebration",
        "wedding", "engagement", "graduation", "achievement", "success", "milestone",
        "gift", "present", "surprise", "special", "memorable", "precious",
        
        # Lifestyle & preferences
        "luxury", "premium", "high-end", "expensive", "exclusive", "sophisticated",
        "budget", "affordable", "cheap", "economical", "value", "savings", "discount",
        "trendy", "fashionable", "stylish", "modern", "contemporary", "chic",
        "traditional", "classic", "timeless", "heritage", "authentic", "genuine",
        "eco-friendly", "sustainable", "green", "environment", "recyclable",
        "convenience", "easy", "simple", "hassle-free", "quick", "instant", "ready",
        
        # Sensory & experience
        "delicious", "tasty", "flavorful", "aromatic", "fragrant", "appealing",
        "soft", "smooth", "creamy", "crunchy", "crispy", "chewy", "tender",
        "sweet", "salty", "spicy", "tangy", "bitter", "mild", "strong", "intense",
        "hot", "cold", "warm", "cool", "refreshing", "satisfying", "fulfilling",
        "comfortable", "cozy", "plush", "luxurious", "gentle", "soothing",
        
        # Problem-solving
        "solution", "fix", "repair", "maintenance", "cleaning", "organizing",
        "storage", "space", "compact", "portable", "lightweight", "durable",
        "multipurpose", "versatile", "practical", "functional", "useful", "handy",
        "emergency", "backup", "spare", "replacement", "alternative", "substitute"
    ]
    
    # Remove duplicates and sort
    return sorted(list(set(mood_words)))

def get_user_custom_words() -> List[str]:
    """Get custom words from user input"""
    print("\n📝 Add your custom words (one per line, press Enter twice to finish):")
    print("Examples: 'craving', 'binge-watch', 'productivity', 'self-care'")
    
    custom_words = []
    while True:
        word = input("➤ ").strip().lower()
        if not word:
            break
        if word not in custom_words:
            custom_words.append(word)
    
    return custom_words

def main():
    """Main expansion function"""
    print("🔧 SEMANTIC MAPPING WORD EXPANDER")
    print("=" * 50)
    
    # Configuration
    PKL_FILE = "product_semantic_mappings.pkl"
    OUTPUT_FILE = "product_semantic_mappings_expanded.pkl"
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found!")
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key.startswith("sk-"):
            print("❌ Invalid API key")
            return
    
    # Load existing mappings
    expander = WordExpander(api_key)
    mappings = expander.load_existing_mappings(PKL_FILE)
    
    if not mappings:
        print("❌ No existing mappings found. Please run the main mapper first.")
        return
    
    # Get existing words
    existing_words = expander.get_existing_words(mappings)
    print(f"📊 Current mappings contain {len(existing_words)} unique words")
    
    # Get new words to add
    print("\n🎯 Choose word source:")
    print("1. Mood/Search words (recommended for quick commerce)")
    print("2. Custom words (manual input)")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    new_words = []
    if choice in ['1', '3']:
        mood_words = get_mood_search_words()
        new_words.extend(mood_words)
        print(f"✅ Added {len(mood_words)} mood/search words")
    
    if choice in ['2', '3']:
        custom_words = get_user_custom_words()
        new_words.extend(custom_words)
        print(f"✅ Added {len(custom_words)} custom words")
    
    if not new_words:
        print("❌ No new words to process")
        return
    
    # Identify missing words
    missing_words = expander.identify_missing_words(new_words, existing_words)
    
    if not missing_words:
        print("✅ All words already exist in mappings!")
        return
    
    # Estimate cost and time
    total_requests = len(mappings)
    estimated_cost = total_requests * 0.03  # Conservative estimate
    estimated_minutes = total_requests / 60  # ~60 requests per minute
    
    print(f"\n💰 EXPANSION SUMMARY:")
    print(f"   📦 Products to update: {len(mappings)}")
    print(f"   📝 New words to add: {len(missing_words)}")
    print(f"   🔄 API requests needed: {total_requests}")
    print(f"   💰 Estimated cost: ${estimated_cost:.2f}")
    print(f"   ⏱️ Estimated time: {estimated_minutes:.1f} minutes")
    
    if input(f"\nProceed with expansion? (y/N): ").strip().lower() != 'y':
        print("❌ Expansion cancelled")
        return
    
    # Expand mappings
    try:
        expanded_mappings = expander.expand_mappings_parallel(mappings, missing_words)
        
        # Save expanded mappings
        print(f"\n💾 Saving expanded mappings...")
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(expanded_mappings, f)
        
        # Analysis
        old_total_words = sum(len(words) for words in mappings.values())
        new_total_words = sum(len(words) for words in expanded_mappings.values())
        words_added = new_total_words - old_total_words
        
        print(f"🎉 SUCCESS! Mappings expanded!")
        print(f"📊 EXPANSION RESULTS:")
        print(f"   📦 Products updated: {len(expanded_mappings)}")
        print(f"   📝 Words added: {words_added:,}")
        print(f"   📈 Total associations: {new_total_words:,}")
        print(f"   💾 Saved to: {OUTPUT_FILE}")
        
        # Replace original file option
        if input(f"\nReplace original file {PKL_FILE}? (y/N): ").strip().lower() == 'y':
            os.rename(OUTPUT_FILE, PKL_FILE)
            print(f"✅ Original file updated: {PKL_FILE}")
        
        print(f"\n🚀 Your semantic mappings now support:")
        print(f"   • 'monday blues' searches")
        print(f"   • 'feeling lazy' queries") 
        print(f"   • 'date night' recommendations")
        print(f"   • And {len(missing_words)} more search patterns!")
        
    except Exception as e:
        print(f"❌ Error during expansion: {e}")

if __name__ == "__main__":
    main()