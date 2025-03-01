import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')


st.header('Research Tool')

user_input = st.text_input("Enter Your Prompt")
llm = HuggingFaceEndpoint(
    repo_id= 'mistralai/Mistral-7B-Instruct',
    task = 'text-generation',
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
)

model = ChatHuggingFace(llm=llm)
result = model.invoke(user_input)

if st.button('Summerize'):
    st.text(result.content)
    


