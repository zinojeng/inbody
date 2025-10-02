import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('key prefix:', os.getenv('OPENAI_API_KEY')[:10])
print('base url:', client.base_url)
try:
    client.responses.create(model='gpt-5.0-mini', input='Hello')
except Exception as exc:
    print('error type:', type(exc).__name__)
    print('error:', exc)
