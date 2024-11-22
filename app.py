import streamlit as st
import torch
import json
import random
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
import nltk
from PIL import Image
import base64

# Download the NLTK 'punkt' package if not already available
nltk.download('punkt')

# Load the intents and trained data
with open(r'C:\Users\User\Desktop\chat_ui_update\intents4.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

data = torch.load(r"C:\Users\User\Desktop\chat_ui_update\data (1).pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Function to get chatbot response
def get_response(sentence):
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
    else:
        return "I'm here to listen, but I might need more guidance."

# Function to add a background image
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image (update with your image path)
add_bg_from_local(r'C:\Users\User\Downloads\backgroundimg.webp')

# Page title and icon
st.image(r'C:\Users\User\Downloads\logoimg.webp', width=80)
st.title("Serenity Chat: Quotes for Well-being")
st.write("Welcome to Serenity Chat 🌿. Feel free to ask for advice or uplifting quotes.")

# Initialize session state for chat history and input
if "history" not in st.session_state:
    st.session_state.history = []

# Function to handle chat input
def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        response = get_response(user_input)
        st.session_state.history.append(f"You: {user_input}")
        st.session_state.history.append(f"Bot: {response}")
        st.session_state.user_input = ""  # Clear the input after processing

# Display chat history
st.write("---")
for chat in st.session_state.history:
    if chat.startswith("You:"):
        st.markdown(f"<div style='color:#2E8B57; font-weight:bold;'>🧑 {chat}</div>", unsafe_allow_html=True)
    elif chat.startswith("Bot:"):
        st.markdown(f"<div style='color:#8B0000;'>🤖 {chat}</div>", unsafe_allow_html=True)

# Text input box with on_change callback
st.text_input(
    "You:",
    value="",
    key="user_input",
    placeholder="Type your message here...",
    on_change=handle_input
)

# Footer section
st.write("---")
st.markdown(
    "<div style='text-align:center; font-size:14px; color:gray;'>"
    "Serenity Chat © 2024. Designed to uplift your spirit.</div>",
    unsafe_allow_html=True
)
