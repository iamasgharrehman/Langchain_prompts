import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API Key
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing! Please set it in your .env file.")
    st.stop()

# Streamlit UI
st.header("Research Tool")
paper_input = st.selectbox("Select Research Paper Name", ['Attention is All You Nedd', 'Bert: Pre-training of Deep Bidirectional Transformers', "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# Initialize the model with API key
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

template = load_prompt('template.json')

chain = template | llm

# Ensure valid user input
if st.button("Summarize"):
        try:
            ai_msg = chain.invoke({
     'paper_input': paper_input,
     'style_input': style_input,
     'length_input': length_input
})
            st.write(ai_msg.content)
        except Exception as e:
            st.error(f"Error: {e}")
