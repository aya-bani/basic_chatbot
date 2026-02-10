from huggingface_hub import InferenceClient

# You can use a free model like 'tiiuae/falcon-7b-instruct' or any other small public model
MODEL_NAME = "TheBloke/vicuna-7B-1.1-HF"  # Example model

# Replace with your HF token if needed
HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # optional for public models

client = InferenceClient(token=HF_TOKEN)

def chat_with_model(message):
    """
    Send a message to the model and get a response.
    """
    response = client.text_generation(
        model=MODEL_NAME,
        inputs=message,
        max_new_tokens=100
    )
    # The model returns a list of outputs, take the first
    return response[0]["generated_text"]

print("💬 General Chatbot Ready! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    bot_reply = chat_with_model(user_input)
    print("Bot:", bot_reply)
