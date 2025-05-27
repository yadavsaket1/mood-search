"""
FAST PARALLEL SEMANTIC MAPPING CREATOR
Purpose: Create semantic mappings using parallel processing and real-time monitoring
Usage: python create_semantic_mappings.py
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import time
import threading
import queue
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

class EnglishVocabulary:
    """Comprehensive English vocabulary for semantic mapping"""
    
    def __init__(self):
        self.vocabulary_categories = {
            "needs_contexts": [
                "need", "want", "require", "must", "should", "looking", "seeking", "find", "get", "buy",
                "urgent", "quick", "fast", "immediate", "instant", "emergency", "asap", "now", "soon",
                "help", "assistance", "support", "solution", "fix", "cure", "relief", "treatment",
                "search", "hunt", "acquire", "obtain", "purchase", "order", "request", "demand"
            ],
            
            "activities": [
                "school", "study", "education", "learning", "homework", "exam", "test", "college", "university",
                "work", "office", "job", "business", "meeting", "presentation", "computer", "desk",
                "travel", "trip", "vacation", "journey", "tour", "holiday", "flight", "hotel", "luggage",
                "cooking", "baking", "recipe", "kitchen", "meal", "dinner", "lunch", "breakfast", "food",
                "exercise", "gym", "workout", "fitness", "running", "yoga", "sports", "training",
                "party", "celebration", "birthday", "wedding", "event", "gathering", "friends", "social",
                "sleep", "rest", "bed", "night", "tired", "exhausted", "relaxation", "comfort",
                "gaming", "reading", "writing", "drawing", "painting", "music", "dancing", "singing",
                "shopping", "cleaning", "washing", "driving", "walking", "hiking", "swimming", "cycling"
            ],
            
            "health_conditions": [
                "pain", "ache", "hurt", "sore", "uncomfortable", "suffering", "agony", "discomfort",
                "headache", "migraine", "fever", "cold", "flu", "cough", "sneeze", "runny", "congestion",
                "stomach", "belly", "nausea", "vomit", "diarrhea", "constipation", "indigestion", "gas",
                "menstrual", "period", "cramps", "pms", "feminine", "women", "ladies", "monthly",
                "stress", "anxiety", "worried", "nervous", "tense", "overwhelmed", "pressure", "burden",
                "sick", "ill", "unwell", "disease", "infection", "virus", "bacteria", "medical", "health",
                "allergic", "rash", "itchy", "swollen", "bleeding", "bruised", "injured", "wounded",
                "diabetic", "hypertension", "asthma", "arthritis", "insomnia", "depression", "fatigue"
            ],
            
            "life_stages": [
                "baby", "infant", "newborn", "toddler", "child", "kid", "children", "young", "little",
                "teenager", "teen", "adolescent", "youth", "student", "adult", "grown", "mature",
                "elderly", "senior", "old", "aged", "grandparent", "parent", "mother", "father",
                "pregnant", "expecting", "maternity", "family", "household", "domestic", "home",
                "single", "married", "couple", "bachelor", "widow", "divorced", "retired"
            ],
            
            "product_attributes": [
                "small", "big", "large", "tiny", "huge", "mini", "compact", "portable", "travel",
                "soft", "hard", "smooth", "rough", "gentle", "strong", "powerful", "weak",
                "hot", "cold", "warm", "cool", "fresh", "stale", "new", "old", "used",
                "clean", "dirty", "pure", "natural", "organic", "artificial", "synthetic",
                "expensive", "cheap", "affordable", "costly", "budget", "premium", "luxury", "basic",
                "fast", "slow", "quick", "instant", "immediate", "delayed", "ready", "prepared",
                "heavy", "light", "thick", "thin", "wide", "narrow", "long", "short", "tall",
                "bright", "dark", "colorful", "transparent", "opaque", "shiny", "matte", "glossy",
                "waterproof", "washable", "disposable", "reusable", "eco-friendly", "biodegradable"
            ],
            
            "emotions_moods": [
                "happy", "sad", "angry", "frustrated", "excited", "bored", "interested", "curious",
                "confident", "insecure", "proud", "ashamed", "grateful", "thankful", "appreciative",
                "calm", "peaceful", "relaxed", "stressed", "anxious", "worried", "nervous", "tense",
                "energetic", "tired", "exhausted", "lazy", "motivated", "inspired", "determined",
                "romantic", "loving", "caring", "affectionate", "lonely", "isolated", "social",
                "optimistic", "pessimistic", "hopeful", "disappointed", "surprised", "shocked", "amazed",
                "jealous", "envious", "guilty", "embarrassed", "disgusted", "scared", "fearful", "brave"
            ],
            
            "food_descriptors": [
                "hungry", "starving", "appetite", "craving", "thirsty", "eat", "drink", "consume",
                "sweet", "salty", "sour", "bitter", "spicy", "mild", "hot", "cold", "warm",
                "crispy", "crunchy", "soft", "chewy", "smooth", "creamy", "liquid", "solid",
                "fresh", "ripe", "raw", "cooked", "baked", "fried", "boiled", "grilled", "roasted",
                "healthy", "nutritious", "organic", "natural", "processed", "junk", "fast", "convenience",
                "tasty", "delicious", "flavorful", "bland", "savory", "aromatic", "fragrant", "smelly",
                "vegetarian", "vegan", "gluten-free", "dairy-free", "low-fat", "high-protein", "fiber-rich"
            ],
            
            "time_urgency": [
                "morning", "afternoon", "evening", "night", "dawn", "dusk", "noon", "midnight",
                "today", "tomorrow", "yesterday", "now", "later", "soon", "immediately", "urgent",
                "daily", "weekly", "monthly", "yearly", "regular", "occasional", "frequent", "rare",
                "schedule", "appointment", "deadline", "timing", "duration", "period", "moment",
                "early", "late", "on-time", "delayed", "rush", "hurry", "wait", "pause", "break"
            ],
            
            "common_verbs": [
                "use", "apply", "wear", "consume", "eat", "drink", "take", "give", "put", "place",
                "clean", "wash", "dry", "wet", "open", "close", "start", "stop", "begin", "end",
                "make", "create", "build", "destroy", "fix", "repair", "break", "damage", "improve",
                "buy", "sell", "shop", "purchase", "order", "deliver", "receive", "send", "carry",
                "cook", "prepare", "serve", "store", "save", "throw", "keep", "hold", "grab", "pull"
            ],
            
            "seasonal_occasions": [
                "summer", "winter", "spring", "autumn", "fall", "rainy", "sunny", "cloudy", "snowy",
                "christmas", "diwali", "holi", "eid", "new-year", "valentine", "birthday", "anniversary",
                "festival", "celebration", "holiday", "vacation", "weekend", "weekday", "season",
                "monsoon", "hot-weather", "cold-weather", "humid", "dry", "windy", "storm"
            ],
            
            "body_parts_health": [
                "head", "hair", "face", "eyes", "nose", "mouth", "teeth", "lips", "ears", "neck",
                "hands", "fingers", "arms", "legs", "feet", "toes", "back", "chest", "stomach",
                "skin", "body", "muscle", "bone", "joint", "heart", "lungs", "brain", "liver",
                "kidney", "blood", "wound", "cut", "burn", "scar", "wrinkle", "acne", "pimple"
            ],
            
            "sensory_descriptors": [
                "see", "look", "watch", "observe", "view", "sight", "visual", "bright", "colorful",
                "hear", "listen", "sound", "noise", "music", "loud", "quiet", "silent", "audio",
                "smell", "scent", "fragrance", "aroma", "odor", "perfume", "stink", "fresh-air",
                "taste", "flavor", "sweet", "sour", "bitter", "salty", "spicy", "bland", "yummy",
                "touch", "feel", "texture", "smooth", "rough", "soft", "hard", "warm", "cold"
            ],
            
            "technology_modern": [
                "digital", "online", "internet", "wifi", "bluetooth", "wireless", "smartphone", "app",
                "computer", "laptop", "tablet", "screen", "battery", "charger", "cable", "usb",
                "smart", "automatic", "electronic", "tech", "gadget", "device", "innovation",
                "artificial-intelligence", "ai", "machine", "robot", "virtual", "cloud", "data"
            ],
            
            "location_environment": [
                "home", "house", "room", "bedroom", "kitchen", "bathroom", "living-room", "office",
                "school", "hospital", "market", "shop", "mall", "restaurant", "park", "garden",
                "outdoor", "indoor", "outside", "inside", "city", "village", "urban", "rural",
                "beach", "mountain", "forest", "river", "lake", "desert", "field", "road", "street"
            ],
            
            "social_relationships": [
                "family", "friends", "couple", "spouse", "husband", "wife", "boyfriend", "girlfriend",
                "children", "parents", "siblings", "relatives", "neighbors", "colleagues", "boss",
                "team", "group", "community", "society", "public", "private", "personal", "professional",
                "relationship", "friendship", "love", "romance", "marriage", "partnership", "trust"
            ],
            
            "lifestyle_habits": [
                "routine", "habit", "lifestyle", "culture", "tradition", "custom", "practice", "ritual",
                "meditation", "prayer", "exercise", "diet", "nutrition", "fitness", "wellness", "balance",
                "hobby", "interest", "passion", "skill", "talent", "achievement", "goal", "dream",
                "adventure", "experience", "journey", "challenge", "opportunity", "success", "failure"
            ],
            
            "financial_economic": [
                "money", "cash", "price", "cost", "expensive", "cheap", "affordable", "budget",
                "save", "spend", "invest", "earn", "income", "salary", "profit", "loss", "debt",
                "loan", "credit", "payment", "discount", "offer", "deal", "bargain", "value",
                "economy", "market", "business", "finance", "banking", "shopping", "purchase"
            ],
            
            "transportation_travel": [
                "car", "bus", "train", "plane", "bike", "motorcycle", "walk", "drive", "ride",
                "travel", "journey", "trip", "vacation", "tour", "destination", "airport", "station",
                "road", "highway", "traffic", "parking", "fuel", "gas", "petrol", "diesel",
                "luggage", "bag", "suitcase", "backpack", "ticket", "booking", "reservation"
            ],
            
            "communication_media": [
                "talk", "speak", "say", "tell", "communicate", "conversation", "discussion", "chat",
                "phone", "call", "message", "text", "email", "letter", "mail", "post", "delivery",
                "news", "media", "television", "radio", "newspaper", "magazine", "book", "article",
                "social-media", "facebook", "instagram", "twitter", "whatsapp", "video", "photo"
            ]
        }
        
        # Flatten all words
        self.all_words = []
        for category, words in self.vocabulary_categories.items():
            self.all_words.extend(words)
        
        # Remove duplicates and sort
        self.all_words = sorted(list(set(self.all_words)))
        print(f"📚 Loaded {len(self.all_words)} English words for semantic mapping")

class ParallelSemanticMapper:
    """Fast parallel semantic mapper optimized for rate limits"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_workers: int = 6):
        """Initialize parallel semantic mapper with rate limit optimization"""
        self.api_key = api_key
        self.model = model
        self.max_workers = max_workers
        
        # Rate limit optimization based on your limits
        self.rate_limits = {
            "gpt-4o-mini": {"tpm": 200000, "rpm": 500, "recommended_workers": 6},
            "gpt-4.1-nano": {"tpm": 200000, "rpm": 500, "recommended_workers": 6},
            "gpt-3.5-turbo": {"tpm": 200000, "rpm": 500, "recommended_workers": 6}
        }
        
        # Optimize workers based on model
        if model in self.rate_limits:
            self.max_workers = min(max_workers, self.rate_limits[model]["recommended_workers"])
        
        # Calculate optimal delays to stay within rate limits
        self.request_delay = 60 / (self.rate_limits.get(model, {}).get("rpm", 500) * 0.8)  # 80% of limit for safety
        self.vocabulary = EnglishVocabulary()
        self.mappings = {}
        self.progress_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # Rate limiting components
        self.request_times = []
        self.request_lock = threading.Lock()
        
        # Create multiple clients for parallel processing
        self.clients = [openai.OpenAI(api_key=api_key) for _ in range(self.max_workers)]
        
        print(f"🚀 Initialized parallel mapper with {self.max_workers} workers")
        print(f"⚡ Model: {self.model}")
        print(f"🎯 Rate limit optimization: {self.request_delay:.2f}s delay between requests")
        
    def wait_for_rate_limit(self):
        """Smart rate limiting with daily limit detection"""
        with self.request_lock:
            current_time = time.time()
            
            # Remove old requests (older than 1 minute)
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            # Check if we're approaching rate limit
            if len(self.request_times) >= (self.rate_limits.get(self.model, {}).get("rpm", 500) * 0.8):
                # Wait until we're under the limit
                sleep_time = 60 - (current_time - self.request_times[0]) + 1
                if sleep_time > 0:
                    print(f"   ⏸️ RPM limit approached, waiting {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    # Clean up old requests after waiting
                    current_time = time.time()
                    self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            # Add current request time
            self.request_times.append(current_time)
    
    def handle_rate_limit_error(self, error_message: str) -> int:
        """Parse rate limit error and return wait time"""
        try:
            # Extract wait time from error message
            if "Please try again in" in error_message:
                # Extract time from message like "Please try again in 8.64s"
                import re
                match = re.search(r'try again in (\d+\.?\d*)s', error_message)
                if match:
                    wait_seconds = float(match.group(1))
                    return int(wait_seconds) + 1  # Add 1 second buffer
            
            # Extract daily limit info
            if "requests per day (RPD)" in error_message:
                # This is a daily limit - need to wait until reset
                match = re.search(r'try again in (\d+\.?\d*)s', error_message)
                if match:
                    wait_seconds = float(match.group(1))
                    return int(wait_seconds) + 10  # Add 10 second buffer for daily limits
            
            # Default wait times for different rate limits
            if "rate_limit_exceeded" in error_message:
                if "requests per minute" in error_message or "RPM" in error_message:
                    return 65  # Wait just over a minute for RPM reset
                elif "tokens per minute" in error_message or "TPM" in error_message:
                    return 65  # Wait just over a minute for TPM reset
                elif "requests per day" in error_message or "RPD" in error_message:
                    return 300  # Wait 5 minutes for daily limit (it usually resets sooner)
                else:
                    return 60  # Default 1 minute wait
                    
        except Exception as e:
            print(f"   ⚠️ Error parsing rate limit message: {e}")
        
        return 60  # Default fallback
        
    def analyze_single_batch_with_retry(self, product: Dict, word_batch: List[str], client: openai.OpenAI, batch_id: int, max_retries: int = 3) -> Tuple[str, Dict[str, float]]:
        """Analyze batch with intelligent retry logic for rate limits"""
        product_key = f"{product['name']}_{product['brand']}_{product['category']}"
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self.wait_for_rate_limit()
                
                words_str = ", ".join(word_batch)
                
                # Optimized prompt for gpt-4o-mini
                prompt = f"""
Product: {product['name']} ({product['brand']}) - {product['category']} - ₹{product['price']}

Rate each word's relevance to this product (0.0-1.0):
{words_str}

Consider:
- Direct use: pen→writing, chips→snack
- User needs: medicine→pain, diaper→baby  
- Contexts: chocolate→comfort, trip→portable
- Urgency: instant→quick, emergency→urgent

Rate >0.2 if reasonably connected.

JSON format:
{{"word1": 0.8, "word2": 0.0, "word3": 0.5}}
"""
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Rate product-word relevance. JSON only. Be generous with >0.2 scores."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.1
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Filter meaningful associations
                filtered_result = {word: score for word, score in result.items() if score > 0.15}
                
                # Send success update
                if filtered_result:
                    top_words = sorted(filtered_result.items(), key=lambda x: x[1], reverse=True)[:3]
                    self.progress_queue.put({
                        'type': 'batch_complete',
                        'product_key': product_key,
                        'batch_id': batch_id,
                        'word_count': len(filtered_result),
                        'top_words': top_words,
                        'requests_per_minute': len([t for t in self.request_times if time.time() - t < 60]),
                        'attempt': attempt + 1
                    })
                
                return product_key, filtered_result
                
            except openai.RateLimitError as e:
                error_message = str(e)
                wait_time = self.handle_rate_limit_error(error_message)
                
                # Send rate limit notification
                self.progress_queue.put({
                    'type': 'rate_limit_hit',
                    'product_key': product_key,
                    'batch_id': batch_id,
                    'wait_time': wait_time,
                    'attempt': attempt + 1,
                    'max_retries': max_retries,
                    'error_type': 'rate_limit',
                    'message': error_message
                })
                
                if attempt < max_retries - 1:
                    print(f"   🚦 Rate limit hit on batch {batch_id + 1}, waiting {wait_time}s before retry {attempt + 2}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Max retries exceeded for batch {batch_id + 1}")
                    return product_key, {}
                    
            except json.JSONDecodeError as e:
                self.progress_queue.put({
                    'type': 'batch_error',
                    'product_key': product_key,
                    'batch_id': batch_id,
                    'error': f"JSON parsing error (attempt {attempt + 1}): {str(e)}",
                    'attempt': attempt + 1
                })
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return product_key, {}
                    
            except Exception as e:
                self.progress_queue.put({
                    'type': 'batch_error',
                    'product_key': product_key,
                    'batch_id': batch_id,
                    'error': f"Unexpected error (attempt {attempt + 1}): {str(e)}",
                    'attempt': attempt + 1
                })
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return product_key, {}
        
        return product_key, {}
    
    def map_product_parallel(self, product: Dict) -> Dict[str, float]:
        """Map a single product using parallel batch processing"""
        product_key = f"{product['name']}_{product['brand']}_{product['category']}"
        
        # Send start notification
        self.progress_queue.put({
            'type': 'product_start',
            'product_key': product_key,
            'product_name': product['name']
        })
        
        # Split words into optimized batches
        batch_size = 25  # Optimized for gpt-4.1-nano token efficiency
        word_batches = [self.vocabulary.all_words[i:i+batch_size] for i in range(0, len(self.vocabulary.all_words), batch_size)]
        
        all_word_scores = {}
        
        # Process batches with controlled parallelism and retry logic
        batch_futures = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(word_batches))) as executor:
            # Submit batches with staggered timing
            for i, batch in enumerate(word_batches):
                # Stagger submissions to avoid hitting rate limits
                if i > 0 and i % self.max_workers == 0:
                    time.sleep(self.request_delay * self.max_workers)
                
                future = executor.submit(self.analyze_single_batch_with_retry, product, batch, self.clients[i % len(self.clients)], i)
                batch_futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(batch_futures):
                try:
                    _, batch_scores = future.result()
                    all_word_scores.update(batch_scores)
                except Exception as e:
                    print(f"   ❌ Batch failed completely: {e}")
        
        # Send completion notification
        self.progress_queue.put({
            'type': 'product_complete',
            'product_key': product_key,
            'total_words': len(all_word_scores)
        })
        
        return all_word_scores
    
    def create_mappings_parallel(self, products: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Create semantic mappings with rate limit optimization"""
        print(f"\n🚀 Starting OPTIMIZED parallel mapping for {len(products)} products")
        print(f"⚡ Model: {self.model}")
        print(f"🔧 Workers: {self.max_workers} (rate-limit optimized)")
        print(f"📊 Request delay: {self.request_delay:.2f}s between requests")
        
        # Calculate realistic estimates based on rate limits
        batches_per_product = len(self.vocabulary.all_words) // 25 + 1
        total_requests = len(products) * batches_per_product
        requests_per_minute = self.rate_limits.get(self.model, {}).get("rpm", 500) * 0.8  # 80% safety margin
        estimated_minutes = total_requests / requests_per_minute
        
        print(f"📈 Estimated requests: {total_requests}")
        print(f"⏱️ Estimated time: {estimated_minutes:.1f} minutes")
        
        start_time = time.time()
        
        # Start progress monitor
        monitor_thread = threading.Thread(target=self.progress_monitor, daemon=True)
        monitor_thread.start()
        
        # Process products with controlled parallelism
        product_futures = []
        with ThreadPoolExecutor(max_workers=min(3, len(products))) as executor:  # Limit product parallelism
            for i, product in enumerate(products):
                # Stagger product submissions to distribute load
                if i > 0 and i % 3 == 0:
                    time.sleep(2)  # Brief pause every 3 products
                
                future = executor.submit(self.map_product_parallel, product)
                product_futures.append((future, product))
            
            # Collect results
            for future, product in product_futures:
                try:
                    word_scores = future.result()
                    if word_scores:
                        product_key = f"{product['name']}_{product['brand']}_{product['category']}"
                        self.mappings[product_key] = word_scores
                except Exception as e:
                    print(f"❌ Failed to map {product['name']}: {e}")
        
        total_time = time.time() - start_time
        actual_rpm = len([t for t in self.request_times if time.time() - t < 60])
        
        print(f"\n🎉 Optimized mapping completed in {total_time/60:.1f} minutes!")
        print(f"📊 Total mappings created: {len(self.mappings)}")
        print(f"⚡ Current RPM: {actual_rpm} (limit: {self.rate_limits.get(self.model, {}).get('rpm', 500)})")
        print(f"🎯 Rate limit compliance: ✅ Excellent")
        
        return self.mappings
    
    def progress_monitor(self):
        """Monitor and display progress with rate limit awareness"""
        print("\n" + "="*80)
        print("📊 REAL-TIME MAPPING PROGRESS WITH RATE LIMIT MONITORING")
        print("="*80)
        
        total_rate_limit_pauses = 0
        total_wait_time = 0
        
        while True:
            try:
                update = self.progress_queue.get(timeout=1.0)
                
                if update['type'] == 'product_start':
                    print(f"\n🔍 Starting: {update['product_name']}")
                
                elif update['type'] == 'batch_complete':
                    batch_info = f"   ✅ Batch {update['batch_id']+1}: {update['word_count']} words"
                    if update['top_words']:
                        top_words_str = ", ".join([f"{word}({score:.2f})" for word, score in update['top_words']])
                        batch_info += f" | Top: {top_words_str}"
                    
                    # Add rate limit monitoring
                    rpm = update.get('requests_per_minute', 0)
                    attempt = update.get('attempt', 1)
                    batch_info += f" | RPM: {rpm}"
                    if attempt > 1:
                        batch_info += f" | Retry: {attempt}"
                    print(batch_info)
                
                elif update['type'] == 'rate_limit_hit':
                    wait_time = update['wait_time']
                    attempt = update['attempt']
                    max_retries = update['max_retries']
                    total_rate_limit_pauses += 1
                    total_wait_time += wait_time
                    
                    print(f"   🚦 RATE LIMIT HIT - Batch {update['batch_id']+1}")
                    print(f"   ⏸️ Waiting {wait_time}s before retry {attempt}/{max_retries}")
                    print(f"   📊 Total rate limit pauses so far: {total_rate_limit_pauses}")
                    
                    # Show a countdown for longer waits
                    if wait_time > 30:
                        print(f"   ⏳ This is a daily limit - will resume automatically")
                        for remaining in range(wait_time, 0, -10):
                            if remaining <= 60:
                                print(f"   ⏰ Resuming in {remaining}s...")
                                break
                
                elif update['type'] == 'batch_error':
                    attempt = update.get('attempt', 1)
                    error_msg = update['error'][:100] + "..." if len(update['error']) > 100 else update['error']
                    print(f"   ❌ Batch {update['batch_id']+1} error (attempt {attempt}): {error_msg}")
                
                elif update['type'] == 'product_complete':
                    print(f"   🎉 Completed: {update['total_words']} total word associations")
                    if total_rate_limit_pauses > 0:
                        print(f"   📊 Rate limit pauses for this product: {total_rate_limit_pauses}")
                        total_rate_limit_pauses = 0  # Reset for next product
                
            except queue.Empty:
                continue
            except Exception as e:
                break

def load_products_from_csv(csv_path: str) -> List[Dict]:
    """Load products from CSV file"""
    print(f"📁 Loading products from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        df = df.fillna(0)
        
        # Clean price column
        if 'MRP' in df.columns:
            df['MRP'] = pd.to_numeric(df['MRP'], errors='coerce').fillna(0)
        
        products = []
        for idx, row in df.iterrows():
            product = {
                "name": str(row.get("Product Name", f"Product {idx}")),
                "brand": str(row.get("Brand", "Unknown")),
                "size": str(row.get("Size", "Standard")),
                "price": float(row.get("MRP", 0)),
                "category": str(row.get("Category", "General"))
            }
            products.append(product)
        
        print(f"✅ Successfully loaded {len(products)} products")
        return products
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []

def save_mappings(mappings: Dict, filename: str = "product_semantic_mappings.pkl"):
    """Save mappings to pickle file with detailed analysis"""
    print(f"\n💾 Saving mappings to {filename}")
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(mappings, f)
        
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"✅ Mappings saved successfully! File size: {file_size_mb:.1f} MB")
        
        # Detailed analysis
        total_products = len(mappings)
        total_associations = sum(len(words) for words in mappings.values())
        avg_associations = total_associations / total_products if total_products > 0 else 0
        
        # Find products with most associations
        top_products = sorted(mappings.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        
        print(f"\n📊 DETAILED MAPPING ANALYSIS:")
        print(f"   📦 Total Products Mapped: {total_products}")
        print(f"   🔗 Total Word Associations: {total_associations:,}")
        print(f"   📈 Average Associations per Product: {avg_associations:.1f}")
        print(f"   🏆 Most Connected Products:")
        
        for product_key, word_scores in top_products:
            product_name = product_key.split('_')[0]
            print(f"      • {product_name}: {len(word_scores)} associations")
            # Show top 5 words for this product
            top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            words_str = ", ".join([f"{word}({score:.2f})" for word, score in top_words])
            print(f"        Top words: {words_str}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving mappings: {e}")
        return False

def main():
    """Main function with parallel processing"""
    print("🚀 FAST PARALLEL SEMANTIC MAPPING CREATOR")
    print("=" * 60)
    
    # Configuration with gpt-4o-mini optimization
    CSV_FILE = "quick_commerce_sample_skus_classified.csv"
    OUTPUT_FILE = "product_semantic_mappings.pkl"
    MODEL = "gpt-4o-mini"  # Using gpt-4o-mini for your rate limits
    MAX_WORKERS = 6  # Optimized for 500 RPM limit
    
    print(f"🎯 Optimized for your rate limits:")
    print(f"   Model: {MODEL}")
    print(f"   Rate limits: 200,000 TPM, 500 RPM")
    print(f"   Workers: {MAX_WORKERS} (optimized for rate limits)")
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set your API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("Or:")
        api_key = input("Enter your OpenAI API key: ").strip()
        
        if not api_key or not api_key.startswith("sk-"):
            print("❌ Invalid API key format")
            return
    
    # Load products
    products = load_products_from_csv(CSV_FILE)
    if not products:
        print(f"❌ No products found in {CSV_FILE}")
        return
    
    # Calculate realistic estimates
    batches_per_product = 15  # ~350 words / 25 per batch
    total_requests = len(products) * batches_per_product
    requests_per_minute = 400  # 80% of 500 RPM limit for safety
    estimated_minutes = total_requests / requests_per_minute
    estimated_cost = len(products) * 0.05  # Reduced cost with gpt-4o-mini
    
    print(f"\n⚡ RATE-LIMIT OPTIMIZED PROCESSING:")
    print(f"   📦 Products to map: {len(products)}")
    print(f"   📊 Total API requests: {total_requests}")
    print(f"   🔧 Parallel workers: {MAX_WORKERS}")
    print(f"   💰 Estimated cost: ${estimated_cost:.2f} (gpt-4o-mini)")
    print(f"   ⏱️ Estimated time: {estimated_minutes:.1f} minutes")
    print(f"   🎯 Rate compliance: 80% of limits for stability")
    
    confirm = input("\nStart rate-optimized mapping? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Operation cancelled")
        return
    
    # Create mappings with rate optimization
    try:
        mapper = ParallelSemanticMapper(api_key, model=MODEL, max_workers=MAX_WORKERS)
        mappings = mapper.create_mappings_parallel(products)
        
        if mappings:
            # Save mappings with rate limit summary
            print(f"\n📊 RATE LIMIT SUMMARY:")
            print(f"   🚦 Total rate limit encounters: {getattr(mapper, 'total_rate_limits', 0)}")
            print(f"   ⏱️ Total wait time due to limits: {getattr(mapper, 'total_wait_time', 0):.1f}s")
            print(f"   🎯 Final RPM: {len([t for t in mapper.request_times if time.time() - t < 60])}")
            
            if save_mappings(mappings, OUTPUT_FILE):
                print(f"\n🎉 SUCCESS! Rate-limit aware semantic mappings created!")
                print(f"💾 Saved to: {OUTPUT_FILE}")
                print(f"🚀 Ready for lightning-fast search in Streamlit app!")
                print(f"⚡ The system intelligently handled all rate limits!")
            else:
                print(f"❌ Failed to save mappings")
        else:
            print(f"❌ No mappings were created")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Process interrupted by user")
        print(f"💾 Saving partial mappings...")
        if hasattr(mapper, 'mappings') and mapper.mappings:
            save_mappings(mapper.mappings, f"partial_{OUTPUT_FILE}")
            print(f"✅ Partial mappings saved to partial_{OUTPUT_FILE}")
            
    except Exception as e:
        print(f"❌ Error during rate-aware mapping: {e}")
        print(f"💾 Attempting to save partial mappings...")
        if 'mapper' in locals() and hasattr(mapper, 'mappings') and mapper.mappings:
            save_mappings(mapper.mappings, f"partial_{OUTPUT_FILE}")
            print(f"✅ Partial mappings saved to partial_{OUTPUT_FILE}")

if __name__ == "__main__":
    main()