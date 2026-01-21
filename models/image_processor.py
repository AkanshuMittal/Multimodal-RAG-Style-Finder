import requests 
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI 

class ImageProcessor:
    """
    Handles image processing, embedding and similarity comparisons.
    """

    def __init__(self, image_size=(512, 512)):
        """
        Initialize the image processor.

        Args:
             image_size (tuple): Target size for input images.
        """

        self.image_size = image_size
        self.client = OpenAI()

    def encode_image(self, image_input, is_url=True):
        """
        Encode am image and extract its feature vector.

        Args:
            image_input: URL or local path of the image.
            is_url: Whether the input is a URL (True) or local file path (False).

        Returns:
            dict: Contains 'base64' string and 'vector' (feature embedding).
        """

        try: 
            # Load the image (URL or Local path)
            if is_url:
                response = requests.get(image_input)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")

            # Resize the image for faster processing
            image = image.resize(self.image_size)

            # Convert image to base64
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Generate feacture vector using OpenAI API Embedding model
            embedding_response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=base64_string
            )

            feature_vector = np.array(embedding_response.data[0].embedding)

            return {
                "base64": base64_string,
                "vector": feature_vector    
            }
        
        except Exception as e:
            print(f"Error encoding image: {e}")
            return {"base64": None, "vector": None}
        
    def find_closet_match(self, user_vector, dataset):
        """
        Find the closet match in the dataset based on cosine similarity.
        
        Args:
            user_vector: Feature vector of the user_uploaded image.
            dataset: DataFrame containing precomputed image vectors.
        
        Returns: 
            tuple: (closest matching row, similarity score)
        """

        try:
            # Extract dataset embedding vectors 
            dataset_vectors = np.vstack(dataset['Embedding'].values)

            # Calculate cosine similarities
            similarities = cosine_similarity(
                user_vector.reshape(1, -1),
                dataset_vectors
            )[0]

            # Identify the index of the closest match
            closest_index = np.argmax(similarities)
            similarity_score = similarities[closest_index]

            closest_row = dataset.iloc[closest_index]
            return closest_row, similarity_score
        
        except Exception as e:
            print(f"Error finding closest match: {e}")
            return None, None   





