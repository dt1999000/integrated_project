"""
LLM Service for Object Dimension Estimation

Uses Hugging Face transformers to query an LLM for typical object dimensions
when the class name is not in the predefined templates.
"""

from typing import Tuple, Optional, Dict, List
import re
import os

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None
    HF_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers torch")

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

# Global LLM temperature setting (default: 0.3)
_llm_temperature = 0.3

# Dimension cache to avoid repeated queries for the same class names
_dimension_cache: Dict[str, Tuple[float, float, float]] = {}

# Common aliases to avoid unnecessary LLM fallback calls
_CLASS_ALIASES: Dict[str, str] = {
    "bicycle": "Cyclist",
    "bike": "Cyclist",
    "cyclist": "Cyclist",
    "motorcycle": "Cyclist",
    "motorbike": "Cyclist",
    "person": "Person",
    "pedestrian": "Pedestrian",
    "human": "Person",
    "bus": "Tram",
    "coach": "Tram",
    "lorry": "Truck",
}


def _normalize_class_name(class_name: Optional[str]) -> str:
    """Normalize a class name for robust matching and cache keys."""
    if class_name is None:
        return ""
    return " ".join(class_name.strip().lower().split())


def _resolve_template_match(class_name: Optional[str]) -> Optional[str]:
    """Resolve class_name to a known KITTI template key if possible."""
    normalized = _normalize_class_name(class_name)
    if not normalized:
        return None

    # Exact match on known key (case-insensitive)
    for key in KITTI_CUBOID_TEMPLATES:
        if key.lower() == normalized:
            return key

    # Alias match
    alias_key = _CLASS_ALIASES.get(normalized)
    if alias_key is not None and alias_key in KITTI_CUBOID_TEMPLATES:
        return alias_key

    # Lightweight semantic token match for common classes
    if "bus" in normalized and "Tram" in KITTI_CUBOID_TEMPLATES:
        return "Tram"
    if any(token in normalized for token in ("truck", "lorry")) and "Truck" in KITTI_CUBOID_TEMPLATES:
        return "Truck"
    if any(token in normalized for token in ("bike", "bicycle", "cyclist", "motorcycle")) and "Cyclist" in KITTI_CUBOID_TEMPLATES:
        return "Cyclist"
    if any(token in normalized for token in ("person", "pedestrian", "human")):
        if "Person" in KITTI_CUBOID_TEMPLATES:
            return "Person"
        if "Pedestrian" in KITTI_CUBOID_TEMPLATES:
            return "Pedestrian"

    return None


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
    """Get or initialize the LLM model (cached)."""
    global _model_cache, _tokenizer_cache
    
    if not HF_AVAILABLE:
        return None, None
    assert AutoTokenizer is not None and AutoModelForCausalLM is not None and torch is not None

    # Default safety behavior: disabled, so no large model downloads happen unexpectedly.
    # Enable only when needed: export LLM_ENABLE_MODEL_QUERY=1
    llm_enabled = os.getenv("LLM_ENABLE_MODEL_QUERY", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not llm_enabled:
        print("[LLM Service] LLM model loading disabled (LLM_ENABLE_MODEL_QUERY=0). Using template defaults.")
        return None, None

    # Default to local-only model loading (no network fetch).
    # Allow downloads only if explicitly requested: export LLM_ALLOW_DOWNLOAD=1
    allow_download = os.getenv("LLM_ALLOW_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
    local_files_only = not allow_download
    
    if _model_cache is None:
        try:
            model_name = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct").strip()
            print(f"[LLM Service] Loading LLM model: {model_name}...")
            _tokenizer_cache = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)

            load_kwargs: Dict[str, object] = {"low_cpu_mem_usage": True}
            # Respect both CUDA availability and the UI preference exposed via LLM_USE_CUDA
            use_cuda_pref = os.getenv("LLM_USE_CUDA", "0").strip().lower() in {"1", "true", "yes", "on"}
            if torch.cuda.is_available() and use_cuda_pref:
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = "auto"

            _model_cache = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                **load_kwargs,
            )
            
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


def get_default_dimensions(class_name: Optional[str]) -> Tuple[float, float, float]:
    """
    Get default dimensions for a class name from KITTI_CUBOID_TEMPLATES.
    
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

    # Special-case a few common non-KITTI classes so they don't fall back
    # to the very generic Unknown template when the LLM is disabled.
    normalized = _normalize_class_name(class_name)
    if "chair" in normalized:
        # Rough dimensions for a typical chair
        print(f"[LLM Service] Using built-in defaults for '{class_name}' (chair)")
        return 0.5, 0.5, 1.0
    if "door handle" in normalized or "doorknob" in normalized or "door knob" in normalized:
        # Very small object near a door
        print(f"[LLM Service] Using built-in defaults for '{class_name}' (door handle)")
        return 0.2, 0.05, 0.05

    resolved_key = _resolve_template_match(class_name)
    if resolved_key is not None:
        template = KITTI_CUBOID_TEMPLATES[resolved_key]
        print(f"[LLM Service] Found semantic/template match: '{class_name}' -> '{resolved_key}'")
        return (float(template['length']), float(template['width']), float(template['height']))
    
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
    normalized_class_name = _normalize_class_name(class_name)
    cached_dims = _dimension_cache.get(normalized_class_name)
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
    assert torch is not None
    
    model, tokenizer = _get_llm_model()
    if model is None or tokenizer is None:
        # Fallback to class-specific default dimensions
        print(f"[LLM Service] LLM model not loaded, using default dimensions for {class_name}: {default_dims}")
        return default_dims
    
    try:
        # Few-shot prompt so GPT-2 continues with numbers (it repeats patterns)
        # End with " (" so the model is primed to output digits
        prompt = (
            "Return only three numbers for object dimensions in meters as: length, width, height\n"
            "Examples:\n"
            "Person: 0.5, 0.5, 1.7\n"
            "Car: 4.0, 1.8, 1.6\n"
            "Bicycle: 1.8, 0.6, 1.2\n"
            f"{class_name}: "
        )
        
        print(f"[LLM Service] Querying LLM for '{class_name}'...")
        print(f"[LLM Service] Prompt: {prompt}")
        
        # Tokenize input
        tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)
        print(f"[LLM Service] Input tokens: {input_ids.shape[1]}")
        
        # Generate response: we need just a few more tokens for "X.X, Y.Y, Z.Z)"
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=16,
                num_return_sequences=1,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        print(f"[LLM Service] Generated tokens: {outputs.shape[1]}")
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Decode only newly generated tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        generated_part = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        print(f"[LLM Service] Full generated text: {generated_text}")
        print(f"[LLM Service] Generated part: {generated_part}")
        
        # Try to extract tuple directly: (X.XX, Y.YY, Z.ZZ) or [X.XX, Y.YY, Z.ZZ]
        length, width, height = _extract_tuple_from_text(generated_part, class_name)
        
        print(f"[LLM Service] Extracted dimensions: length={length:.2f}m, width={width:.2f}m, height={height:.2f}m")
        
        # Cache the result
        _dimension_cache[normalized_class_name] = (length, width, height)
        
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
    
    # Pattern 0: strict beginning of generated text (most reliable)
    # Supports optional wrapping parentheses.
    strict_start_pattern = r'^\s*[\(\[]?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)'
    match = re.search(strict_start_pattern, text)
    if match:
        try:
            length, width, height = float(match.group(1)), float(match.group(2)), float(match.group(3))
            if all(0.1 <= v <= 20.0 for v in [length, width, height]):
                print(f"[LLM Service] Strict-start pattern matched: ({length}, {width}, {height})")
                return length, width, height
        except ValueError as e:
            print(f"[LLM Service] Strict-start pattern matched but ValueError: {e}")

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



