import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_key)

# Check only for embedding-capable models
print("=== TEXT EMBEDDING MODELS FOR GEMINI ===\n")
embedding_models = []

for model in genai.list_models():
    print(f"Model: {model.name}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"  Methods: {model.supported_generation_methods}")
        if 'embedContent' in model.supported_generation_methods:
            embedding_models.append(model.name)
            print(f"  ✓ This is an EMBEDDING model!")
    print()

if not embedding_models:
    print("\n❌ No embedding models found!")
else:
    print(f"\n✅ Total embedding models available: {len(embedding_models)}")
    print(f"\nUse this model name in your code:")
    for model in embedding_models:
        model_clean = model.replace("models/", "")
        print(f'  model="models/{model_clean}"')
