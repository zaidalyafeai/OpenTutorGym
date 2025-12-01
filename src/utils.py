import base64
import hashlib

def image_to_base64(image):
    """
    Convert an image to base64.
    
    Args:
        image: Image to be converted.
        
    Returns:
        str: Base64 encoded image.
    """
    base64_image = base64.b64encode(image).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"

def hash_question(question: str) -> str:
    """
    Hash a question.
    
    Args:
        question: Question to be hashed.
        
    Returns:
        str: Hashed question.
    """
    return hashlib.sha256(question.encode()).hexdigest()
    