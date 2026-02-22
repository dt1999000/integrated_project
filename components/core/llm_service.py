"""
LLM Service for Object Dimension Estimation

Uses Hugging Face transformers to query an LLM for typical object dimensions
when the class name is not in the predefined templates.
Also uses sentence transformers for semantic similarity matching.
"""

from typing import Tuple, Optional, Dict, List
import re
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers torch")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers library not available. Install with: pip install sentence-transformers")

# Import KITTI templates from constants
try:
    from .constants import KITTI_CUBOID_TEMPLATES
except ImportError:
    # Fallback if constants not available
    KITTI_CUBOID_TEMPLATES = {
        'Car': {'length': 3.64, 'width': 1.86, 'height': 1.58},
        'Pedestrian': {'length': 0.88, 'width': 0.90, 'height': 1.77},
        'Cyclist': {'length': 1.68, 'width': 0.75, 'height': 1.76},
        'Van': {'length': 4.76, 'width': 2.22, 'height': 2.27},
        'Truck': {'length': 9.82, 'width': 2.99, 'height': 3.38},
        'Tram': {'length': 15.59, 'width': 3.66, 'height': 3.73},
        'Misc': {'length': 2.56, 'width': 1.91, 'height': 1.68},
        'Person_sitting': {'length': 0.72, 'width': 0.80, 'height': 1.29},
        'Unknown': {'length': 2.0, 'width': 1.5, 'height': 1.5},
    }


# Global model cache to avoid reloading
_model_cache = None
_tokenizer_cache = None
_sentence_model_cache = None
_template_embeddings_cache = None

# Global LLM temperature setting (default: 0.3)
_llm_temperature = 0.3

# Dimension cache to avoid repeated queries for the same class names
_dimension_cache: Dict[str, Tuple[float, float, float]] = {}


def set_llm_temperature(temperature: float):
    """
    Set the temperature for LLM generation.
    
    Args:
        temperature: Temperature value (0.0 to 2.0). Lower values make output more deterministic.
    """
    global _llm_temperature
    _llm_temperature = max(0.0, min(2.0, temperature))  # Clamp between 0 and 2
    print(f"[LLM Service] Temperature set to {_llm_temperature}")


def get_llm_temperature() -> float:
    """Get the current LLM temperature setting."""
    return _llm_temperature


def _get_llm_model():
    """Get or initialize the LLM model (cached)"""
    global _model_cache, _tokenizer_cache
    
    if not HF_AVAILABLE:
        return None, None
    
    if _model_cache is None:
        try:
            # Use a small, free model that can generate text
            # Using GPT-2 as it's lightweight and free
            model_name = "gpt2"
            print(f"[LLM Service] Loading LLM model: {model_name}...")
            _tokenizer_cache = AutoTokenizer.from_pretrained(model_name)
            _model_cache = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set pad token if not present
            if _tokenizer_cache.pad_token is None:
                _tokenizer_cache.pad_token = _tokenizer_cache.eos_token
            
            print(f"[LLM Service] LLM model loaded successfully")
            print(f"[LLM Service] Model device: {next(_model_cache.parameters()).device}")
            print(f"[LLM Service] Model dtype: {next(_model_cache.parameters()).dtype}")
        except Exception as e:
            print(f"[LLM Service] Failed to load LLM model: {e}")
            print("[LLM Service] Falling back to default dimensions")
            return None, None
    
    return _model_cache, _tokenizer_cache


def _get_sentence_model():
    """Get or initialize the sentence transformer model (cached)"""
    global _sentence_model_cache
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    
    if _sentence_model_cache is None:
        try:
            # Use a lightweight sentence transformer model for semantic similarity
            model_name = "all-MiniLM-L6-v2"
            print(f"[LLM Service] Loading sentence transformer model: {model_name}...")
            _sentence_model_cache = SentenceTransformer(model_name)
            print(f"[LLM Service] Sentence transformer model loaded successfully")
            print(f"[LLM Service] Model device: {_sentence_model_cache.device}")
            
            # Pre-compute embeddings for all template class names
            _precompute_template_embeddings()
        except Exception as e:
            print(f"[LLM Service] Failed to load sentence transformer model: {e}")
            return None
    
    return _sentence_model_cache


def _precompute_template_embeddings():
    """Pre-compute embeddings for all KITTI template class names"""
    global _template_embeddings_cache
    
    if _sentence_model_cache is None:
        return
    
    try:
        print(f"[LLM Service] Pre-computing embeddings for {len(KITTI_CUBOID_TEMPLATES)} template classes...")
        template_names = list(KITTI_CUBOID_TEMPLATES.keys())
        _template_embeddings_cache = _sentence_model_cache.encode(
            template_names,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        print(f"[LLM Service] Template embeddings computed successfully")
        print(f"[LLM Service] Embedding shape: {_template_embeddings_cache.shape}")
    except Exception as e:
        print(f"[LLM Service] Failed to pre-compute template embeddings: {e}")
        _template_embeddings_cache = None


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def get_default_dimensions(class_name: Optional[str]) -> Tuple[float, float, float]:
    """
    Get default dimensions for a class name from KITTI_CUBOID_TEMPLATES.
    
    Uses semantic similarity to find the most similar class name if exact match fails.
    Returns dimensions as a tuple.
    
    Args:
        class_name: Semantic class name (case-insensitive, can be variant)
    
    Returns:
        (length, width, height) tuple in meters
    """
    if class_name is None:
        # Use Unknown template as generic default
        template = KITTI_CUBOID_TEMPLATES.get('Unknown', {'length': 2.0, 'width': 1.5, 'height': 1.5})
        return (float(template['length']), float(template['width']), float(template['height']))
    
    # First try direct lookup (case-sensitive)
    template = KITTI_CUBOID_TEMPLATES.get(class_name)
    if template is not None:
        print(f"[LLM Service] Found exact match for '{class_name}' in templates")
        return (float(template['length']), float(template['width']), float(template['height']))
    
    # Try case-insensitive lookup
    class_lower = class_name.lower()
    for key, value in KITTI_CUBOID_TEMPLATES.items():
        if key.lower() == class_lower:
            print(f"[LLM Service] Found case-insensitive match: '{class_name}' -> '{key}'")
            return (float(value['length']), float(value['width']), float(value['height']))
    
    # Try semantic similarity using sentence transformers
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        print(f"[LLM Service] Attempting semantic similarity search for '{class_name}'...")
        sentence_model = _get_sentence_model()
        if sentence_model is None:
            print(f"[LLM Service] Sentence transformer model not available")
        elif _template_embeddings_cache is None:
            print(f"[LLM Service] Template embeddings not pre-computed, computing now...")
            _precompute_template_embeddings()
        
        if sentence_model is not None and _template_embeddings_cache is not None:
            try:
                # Encode the input class name
                print(f"[LLM Service] Encoding query: '{class_name}'")
                query_embedding = sentence_model.encode([class_name], convert_to_numpy=True)[0]
                print(f"[LLM Service] Query embedding shape: {query_embedding.shape}")
                
                # Compute cosine similarity with all template embeddings
                similarities = []
                template_names = list(KITTI_CUBOID_TEMPLATES.keys())
                print(f"[LLM Service] Computing similarities with {len(template_names)} templates...")
                for i, template_name in enumerate(template_names):
                    similarity = _cosine_similarity(query_embedding, _template_embeddings_cache[i])
                    similarities.append((similarity, template_name))
                
                # Sort by similarity (highest first)
                similarities.sort(reverse=True, key=lambda x: x[0])
                
                best_similarity, best_match = similarities[0]
                print(f"[LLM Service] Semantic similarity search for '{class_name}':")
                print(f"[LLM Service]   Best match: '{best_match}' (similarity: {best_similarity:.4f})")
                
                # Show top 3 matches for debugging
                for i, (sim, name) in enumerate(similarities[:3]):
                    print(f"[LLM Service]   Top {i+1}: '{name}' (similarity: {sim:.4f})")
                
                # Use template if similarity is very high (threshold: 0.75)
                if best_similarity >= 0.75:
                    template = KITTI_CUBOID_TEMPLATES.get(best_match)
                    if template is not None:
                        print(f"[LLM Service] Using template dimensions for '{best_match}' (similarity: {best_similarity:.4f})")
                        return (float(template['length']), float(template['width']), float(template['height']))
                else:
                    print(f"[LLM Service] Similarity too low ({best_similarity:.4f} < 0.75), will query LLM")
            except Exception as e:
                print(f"[LLM Service] Error in semantic similarity search: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"[LLM Service] Sentence transformers not available (SENTENCE_TRANSFORMERS_AVAILABLE={SENTENCE_TRANSFORMERS_AVAILABLE})")
    
    # Fallback to Unknown template
    template = KITTI_CUBOID_TEMPLATES.get('Unknown', {'length': 2.0, 'width': 1.5, 'height': 1.5})
    print(f"[LLM Service] No match found for '{class_name}', using Unknown template")
    return (float(template['length']), float(template['width']), float(template['height']))


def query_llm_for_dimensions(class_name: str) -> Tuple[float, float, float]:
    """
    Query an LLM for typical object dimensions.
    
    Uses Hugging Face transformers with a free model to generate a response
    about typical dimensions for the given class name, then parses the output.
    
    Note: GPT-2 is not very good at following instructions, so this function
    should only be used as a last resort when semantic similarity fails.
    
    This function checks the cache first to avoid repeated queries.
    
    Args:
        class_name: Semantic class name (e.g., 'Car', 'Pedestrian', 'Bicycle')
    
    Returns:
        (length, width, height) tuple in meters
    """
    # Check cache first
    cached_dims = _dimension_cache.get(class_name)
    if cached_dims is not None:
        print(f"[LLM Service] Using cached dimensions for '{class_name}': {cached_dims}")
        return cached_dims
    
    # First try to get dimensions using semantic similarity
    # This is more reliable than querying GPT-2
    default_dims = get_default_dimensions(class_name)
    
    # Check if we got a match from semantic similarity (not Unknown template)
    # If we did, return it instead of querying the unreliable LLM
    unknown_template = KITTI_CUBOID_TEMPLATES.get('Unknown', {'length': 2.0, 'width': 1.5, 'height': 1.5})
    unknown_dims = (float(unknown_template['length']), float(unknown_template['width']), float(unknown_template['height']))
    
    if default_dims != unknown_dims:
        print(f"[LLM Service] Semantic similarity found match, skipping LLM query")
        return default_dims
    
    # Only query LLM if semantic similarity didn't find a match
    if not HF_AVAILABLE:
        # Fallback to class-specific default dimensions
        print(f"[LLM Service] LLM not available, using default dimensions for {class_name}: {default_dims}")
        return default_dims
    
    model, tokenizer = _get_llm_model()
    if model is None or tokenizer is None:
        # Fallback to class-specific default dimensions
        print(f"[LLM Service] LLM model not loaded, using default dimensions for {class_name}: {default_dims}")
        return default_dims
    
    try:
        # Few-shot prompt so GPT-2 continues with numbers (it repeats patterns)
        # End with " (" so the model is primed to output digits
        prompt = (
            "Dimensions in meters (length, width, height):\n"
            "Person: (0.5, 0.5, 1.7)\n"
            "Car: (4.0, 1.8, 1.6)\n"
            "Bicycle: (1.8, 0.6, 1.2)\n"
            f"{class_name}: ("
        )
        
        print(f"[LLM Service] Querying LLM for '{class_name}'...")
        print(f"[LLM Service] Prompt: {prompt}")
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        print(f"[LLM Service] Input tokens: {inputs.shape[1]}")
        
        # Generate response: we need just a few more tokens for "X.X, Y.Y, Z.Z)"
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 25,  # Enough for "1.2, 0.8, 1.5)"
                num_return_sequences=1,
                temperature=_llm_temperature,  # Use configurable temperature
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        print(f"[LLM Service] Generated tokens: {outputs.shape[1]}")
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the prompt part and the generated part
        generated_part = generated_text[len(prompt):].strip()
        
        print(f"[LLM Service] Full generated text: {generated_text}")
        print(f"[LLM Service] Generated part: {generated_part}")
        
        # Try to extract tuple directly: (X.XX, Y.YY, Z.ZZ) or [X.XX, Y.YY, Z.ZZ]
        length, width, height = _extract_tuple_from_text(generated_part, class_name)
        
        print(f"[LLM Service] Extracted dimensions: length={length:.2f}m, width={width:.2f}m, height={height:.2f}m")
        
        # Cache the result
        _dimension_cache[class_name] = (length, width, height)
        
        return length, width, height
        
    except Exception as e:
        print(f"[LLM Service] Error querying LLM for {class_name}: {e}")
        import traceback
        traceback.print_exc()
        print("[LLM Service] Falling back to default dimensions")
        return get_default_dimensions(class_name)


def _extract_tuple_from_text(text: str, class_name: str) -> Tuple[float, float, float]:
    """
    Extract tuple (length, width, height) from LLM-generated text.
    
    Looks for tuple format: (X.XX, Y.YY, Z.ZZ) or [X.XX, Y.YY, Z.ZZ]
    """
    # Get class-specific default dimensions as fallback
    default_dims = get_default_dimensions(class_name)
    
    print(f"[LLM Service] Extracting tuple from text: '{text[:200]}'")
    
    # Pattern 1: Tuple format (X.XX, Y.YY, Z.ZZ) or [X.XX, Y.YY, Z.ZZ]
    pattern1 = r'[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]'
    match = re.search(pattern1, text)
    if match:
        try:
            length, width, height = float(match.group(1)), float(match.group(2)), float(match.group(3))
            # Validate reasonable range
            if all(0.1 <= v <= 20.0 for v in [length, width, height]):
                print(f"[LLM Service] Tuple pattern matched: ({length}, {width}, {height})")
                return length, width, height
        except ValueError as e:
            print(f"[LLM Service] Tuple pattern matched but ValueError: {e}")
    
    # Pattern 2: Three numbers separated by commas (first occurrence)
    pattern2 = r'([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)'
    match = re.search(pattern2, text)
    if match:
        try:
            length, width, height = float(match.group(1)), float(match.group(2)), float(match.group(3))
            # Validate reasonable range
            if all(0.1 <= v <= 20.0 for v in [length, width, height]):
                print(f"[LLM Service] Three numbers pattern matched: ({length}, {width}, {height})")
                return length, width, height
        except ValueError:
            print(f"[LLM Service] Three numbers pattern matched but ValueError")
    
    # If no pattern matches, return defaults
    print(f"[LLM Service] Could not extract tuple from LLM text, using defaults for {class_name}")
    return default_dims



