import ollama

response_stream = ollama.chat(model="llama3", messages=[
    {
        "role": "user",
        "content": "Teach me how to use numpy to get cross product of two vectors",
    }],
    stream = True
)

for chunk in response_stream:
    print(chunk["message"]["content"], end='', flush=True)