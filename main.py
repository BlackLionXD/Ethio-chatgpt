from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model and tokenizer (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize object to store conversation history
conversation_history = []

while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)
    
    # Get the input data from the user
    input_text = input("> ")

    if input_text.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break

    # Tokenize the input text and history
    inputs = tokenizer(history_string + input_text, return_tensors="pt", truncation=True, max_length=1024)

    # Generate the response from the model
    outputs = model.generate(**inputs, max_length=1024, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Bot: {response}")

    # Limit the conversation history length to avoid performance issues
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]
