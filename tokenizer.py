import streamlit as st
from transformers import AutoTokenizer
import random

def choose_tokenizer():
    # Select a pre-trained tokenizer 
    tokenizer_name = st.selectbox("Select a pre-trained tokenizer", ["Llama-3.2-3B-Instruct", "Llama-3.2-3B-Code"])
    
    # Load the selected tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{tokenizer_name}")

    return tokenizer

# Function to tokenize and count tokens
def tokenize_text(text):
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)
    
    # Count the number of tokens
    num_tokens = len(tokens)
    
    return tokens, num_tokens

def tokenize_and_color(text):
    # Tokenize the input text
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=True)  # Convert tokens to IDs
    
    # For consistent color assignment, we'll use a dictionary to store color mappings
    token_color_map = {}
    
    # For visualization: color each token with a unique color
    colored_tokens = []
    for token in tokens:
        # If the token already has a color, reuse it
        if token not in token_color_map:
            # Assign a new random color
            token_color_map[token] = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        color = token_color_map[token]
        colored_tokens.append(f'<span style="color:{color};">{token}</span>')

    # Join all colored tokens into a string
    return " ".join(colored_tokens), len(tokens), token_ids

# Streamlit UI
st.title("Text Tokenization Visualizer")

# Input text field
input_text = st.text_area("Enter text to tokenize", height=75)
tokenizer = choose_tokenizer()

# Button to trigger tokenization
if st.button("Tokenize"):
    if input_text:
        # Tokenize the input text and get the number of tokens
        tokens, num_tokens, token_ids = tokenize_and_color(input_text)
        
        # Display the tokenized text
        st.markdown(f"### Tokenized Text:")
        st.markdown(f"<p style='font-family:monospace'>&lts&gt{tokens}</p>", unsafe_allow_html=True)

        # Display token IDs
        st.markdown("### Token IDs (Tensor):")
        st.text(token_ids)        
        
        # Display the number of tokens
        st.write(f"Total number of tokens: {num_tokens}")
    else:
        st.warning("Please enter some text to tokenize.")
