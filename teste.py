import os
from dotenv import load_dotenv

load_dotenv(override=True)
k = os.getenv("XAI_API_KEY")
print(repr(k[:4]), len(k), repr(k[-1]))
print(repr(k[:12]))