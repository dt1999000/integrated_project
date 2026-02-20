"""
LLM Service for Object Dimension Estimation

Uses Hugging Face transformers to query an LLM for typical object dimensions
when the class name is not in the predefined templates.
"""

from typing import Tuple, Optional
import re

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers torch")


# Global model cache to avoid reloading
_model_cache = None
_tokenizer_cache = None


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
            print(f"Loading LLM model: {model_name}...")
            _tokenizer_cache = AutoTokenizer.from_pretrained(model_name)
            _model_cache = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set pad token if not present
            if _tokenizer_cache.pad_token is None:
                _tokenizer_cache.pad_token = _tokenizer_cache.eos_token
            
            print(f"LLM model loaded successfully")
        except Exception as e:
            print(f"Failed to load LLM model: {e}")
            print("Falling back to default dimensions")
            return None, None
    
    return _model_cache, _tokenizer_cache


def query_llm_for_dimensions(class_name: str) -> Tuple[float, float, float]:
    """
    Query an LLM for typical object dimensions.
    
    Uses Hugging Face transformers with a free model to generate a response
    about typical dimensions for the given class name, then parses the output.
    
    Args:
        class_name: Semantic class name (e.g., 'Car', 'Pedestrian', 'Bicycle')
    
    Returns:
        (length, width, height) tuple in meters
    """
    if not HF_AVAILABLE:
        # Fallback to default dimensions
        print(f"LLM not available, using default dimensions for {class_name}")
        return 4.0, 1.8, 1.6
    
    model, tokenizer = _get_llm_model()
    if model is None or tokenizer is None:
        # Fallback to default dimensions
        print(f"LLM model not loaded, using default dimensions for {class_name}")
        return 4.0, 1.8, 1.6
    
    try:
        # Create a prompt asking for dimensions
        prompt = f"What are typical dimensions in meters (length, width, height) for a {class_name}? Format: length=X.X width=Y.Y height=Z.Z"
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response (limit to reasonable length)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the prompt part and the generated part
        generated_part = generated_text[len(prompt):].strip()
        
        # Try to parse dimensions from the generated text
        # Look for patterns like "length=3.5 width=1.8 height=1.6" or "3.5m x 1.8m x 1.6m"
        length, width, height = _parse_dimensions_from_text(generated_part, class_name)
        
        print(f"LLM response for {class_name}: {generated_part[:100]}...")
        print(f"Parsed dimensions: length={length:.2f}m, width={width:.2f}m, height={height:.2f}m")
        
        return length, width, height
        
    except Exception as e:
        print(f"Error querying LLM for {class_name}: {e}")
        print("Falling back to default dimensions")
        return 4.0, 1.8, 1.6


def _parse_dimensions_from_text(text: str, class_name: str) -> Tuple[float, float, float]:
    """
    Parse dimensions from LLM-generated text.
    
    Tries multiple patterns to extract length, width, height values.
    """
    # Default fallback dimensions based on class name (heuristic)
    defaults = {
        'car': (4.0, 1.8, 1.6),
        'truck': (8.0, 2.5, 3.0),
        'bus': (12.0, 2.5, 3.5),
        'pedestrian': (0.5, 0.5, 1.7),
        'cyclist': (1.8, 0.7, 1.7),
        'bicycle': (1.8, 0.7, 1.2),
        'motorcycle': (2.0, 0.8, 1.3),
    }
    
    class_lower = class_name.lower()
    default_dims = defaults.get(class_lower, (4.0, 1.8, 1.6))
    
    # Pattern 1: "length=X.X width=Y.Y height=Z.Z"
    pattern1 = r'length[=:\s]+([\d.]+)\s*(?:m|meters?)?[,\s]+width[=:\s]+([\d.]+)\s*(?:m|meters?)?[,\s]+height[=:\s]+([\d.]+)\s*(?:m|meters?)?'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        except ValueError:
            pass
    
    # Pattern 2: "X.Xm x Y.Ym x Z.Zm" or "X.X x Y.Y x Z.Z"
    pattern2 = r'([\d.]+)\s*(?:m|meters?)?\s*[x×]\s*([\d.]+)\s*(?:m|meters?)?\s*[x×]\s*([\d.]+)\s*(?:m|meters?)?'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        except ValueError:
            pass
    
    # Pattern 3: Three consecutive numbers
    pattern3 = r'([\d.]+)\s*(?:m|meters?)?[,\s]+([\d.]+)\s*(?:m|meters?)?[,\s]+([\d.]+)\s*(?:m|meters?)?'
    matches = re.findall(pattern3, text)
    if matches:
        # Take the first match that has reasonable values (between 0.1 and 20 meters)
        for match in matches:
            try:
                vals = [float(m) for m in match]
                if all(0.1 <= v <= 20.0 for v in vals):
                    # Assume order: length, width, height (largest to smallest typically)
                    vals_sorted = sorted(vals, reverse=True)
                    return vals_sorted[0], vals_sorted[1], vals_sorted[2]
            except ValueError:
                continue
    
    # If no pattern matches, return defaults based on class name
    print(f"Could not parse dimensions from LLM text, using defaults for {class_name}")
    return default_dims

