import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

st.header('Research Tool')

user_input = st.text_input("Enter your input prompt")

# Load Hugging Face model pipeline
@st.cache_resource  # Caches the model for efficiency
def load_model():
    return HuggingFacePipeline.from_model_id(
        model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        task='text-generation',
        pipeline_kwargs={
            "temperature": 0.5,
            "max_new_tokens": 100  # Set a reasonable token limit
        }
    )

llm = load_model()
model = ChatHuggingFace(llm=llm)

if st.button("Summarize"):
    if user_input:
        with st.spinner("Generating summary..."):
            result = model.invoke(user_input)  # No need for `await` since we're not in an async function
            st.write(result.content)
    else:
        st.warning("Please enter a prompt before clicking Summarize.")

if st.button("Summerize"):
    result= model.invoke(user_input)
    st.write(result.content)
