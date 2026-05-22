import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_key)

# List all available models
print("All models:")
for model in genai.list_models():
    print(f"\n- {model.name}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"  Supports: {model.supported_generation_methods}")

print("\n\n=== EMBEDDING MODELS ONLY ===")
for model in genai.list_models():
    if hasattr(model, 'supported_generation_methods'):
        if 'embedContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")

