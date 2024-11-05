# Install necessary libraries (uncomment if running in a new environment)
# !pip install transformers torch streamlit

import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
!streamlit run app.py &

# Expose the Streamlit app to the web using ngrok
from pyngrok import ngrok
public_url = ngrok.connect(port=8501)
print(f"Access your Streamlit app at: {public_url}")
# Load the Flan-T5 model and tokenizer
model_name = "google/flan-t5-base"  # You can choose a different version as needed
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to get a response from the model
def get_ai_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app
def main():
    st.title("AI Information Chatbot")
    user_input = st.text_input("Ask me anything about AI:")
    
    if user_input:
        response = get_ai_response(user_input)
        st.write("Bot:", response)

if __name__ == "__main__":
    main()
    # Write the complete Streamlit app code to a Python file with installation commands included



