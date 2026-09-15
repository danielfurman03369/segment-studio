import requests
from api_key import MY_API_KEY

def ask_llm(prompt: str) -> str:
    '''
    it sends a prompt to the ollama api and returns the answer of the model as text.
    and the key is kept in api_key.py so it is not written inside the code.
    '''
    API_KEY = MY_API_KEY  # put your key in api_key.py instead of ollama_your_key_here
    # API_KEY = "ollama_your_key_here"  # put your key here instead of ollama_your_key_here

    response = requests.post(
        "https://ollama.com/api/chat",
        headers={
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
            "model": "gpt-oss:120b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
    )

    data = response.json()

    return data["message"]["content"]


