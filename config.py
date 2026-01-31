# MODEL & API CONFIGURATION

OPENAI_VISION_MODEL = "gpt-4.1"                 
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

MAX_TOKENS = 150
TEMPERATURE = 0.2

# IMAGE PROCESSING SETTINGS

IMAGE_SIZE = (512, 512)        # Resize images before sending to API
IMAGE_FORMAT = "JPEG"

# SIMILARITY SEARCH SETTINGS

SIMILARITY_THRESHOLD = 0.8     # Minimum cosine similarity for match

DATASET_PATH = "swift-style-embeddings.pkl"   