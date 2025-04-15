# Constants
VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
VISION_MODEL2 = "llama-3.2-90b-vision-preview"
MAX_TOKENS = 450
TARGET_IMAGE_SIZE = (224, 224)  # Reduced size for token optimization

# Rate limits per key
RATE_LIMITS = {
    VISION_MODEL2: {
        'requests_per_min': 15,
        'tokens_per_min': 7000,
        'requests_per_day': 3500
    },
    VISION_MODEL: {
        'requests_per_min': 30,
        'tokens_per_min': 6000,
        'requests_per_day': 2000
    }
}