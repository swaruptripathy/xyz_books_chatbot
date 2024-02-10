import os
import sys
import streamlit as st
from streamlit_chat import message
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma

import warnings
warnings.filterwarnings("ignore")

st.title("📖 XYZ Books - Personal Assitant")
st.divider()

data_file = "data.txt"
data_persist = False
prompt = None

#containers for the chat 
request_container = st.container()
response_container = st.container()

# Persist and save data to disk using Chroma 
if data_persist and os.path.exists("persist"):
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader(data_file)
    loader.load()
    if data_persist:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model="gpt-3.5-turbo"), retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}))

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! I am your Personal assistant built by XYZ Books"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

def conversational_chat(prompt):
    result = chain({"question": prompt, "chat_history": st.session_state['history']})
    st.session_state['history'].append((prompt, result["answer"]))
    return result["answer"]


with request_container:
    with st.form(key='xyz_form', clear_on_submit=True):
        
        user_input = st.text_input("Prompt:", placeholder="Message XYZBot...", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer", seed=24)
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed=12)
