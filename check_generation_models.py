import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_key)

# List all available models
print("=== TEXT GENERATION MODELS FOR GEMINI ===\n")
generation_models = []

for model in genai.list_models():
    if hasattr(model, 'supported_generation_methods'):
        if 'generateContent' in model.supported_generation_methods:
            generation_models.append(model.name)
            print(f"✓ {model.name}")

if not generation_models:
    print("No generation models found!")
else:
    print(f"\nTotal generation models available: {len(generation_models)}")
    print(f"\nUse this in your code:")
    print(f'genai.GenerativeModel("{generation_models[0].replace("models/", "")}")')
