import os
from dotenv import load_dotenv
load_dotenv()
print('env key prefix:', os.getenv('OPENAI_API_KEY')[:10])
