import os
from dotenv import load_dotenv

load_dotenv()

# If you used GEMINI_API_KEY in .env, map it:
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    max_tokens=1000,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the national sport of TamilNadu?"},
]

response = model.invoke(messages)
print(response.content)