import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def main():
    st.set_page_config(layout="wide")
    st.title("MediBot: Your Health Assistant")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Ask a Question", "About", "Contact"])

    if page == "Home":
        st.header("Welcome to MediBot! 🤖")
        st.write("""
        MediBot is your AI-powered health assistant, helping you with medical queries 
        by retrieving information from trusted sources.  
        
        - **Ask health-related questions** 🏥  
        - **Get AI-powered responses** 🤖  
        - **View references and sources** 📚  
        
        Try it now by navigating to "Ask a Question" in the sidebar!
        """)
        st.image("healthcare_image.jpg", use_container_width=True)  # Optional: Add an image

    elif page == "Ask a Question":
        ask_question_page()

    elif page == "About":
        st.header("About MediBot 🏥")
        st.write("""
        **MediBot** is an AI-driven chatbot designed to assist users with medical questions.  
        
        - Uses **advanced NLP models** to analyze and answer your queries.  
        - Sources information from **trusted medical databases**.  
        - Offers **conversational AI** to make health information more accessible.  
        
        _Note: This chatbot does not replace professional medical advice. Please consult a doctor for health concerns._
        """)

    elif page == "Contact":
        st.header("Contact Us 📞")
        st.write("""
        If you have any questions, feedback, or suggestions, reach out to us:  
        - 📧 **Email:** shubhangisinha1929@gmail.com 
        - 📍 **Address:** KIIT, Bhubaneshwar
        - 📞 **Phone:** +1 234 567 890  
        """)

def ask_question_page():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Ask Your Question")
        st.write("Feel free to ask advanced medical questions.")
        prompt = st.text_input("Type your question here:")

        if st.button("Submit"):
            st.chat_message('user').markdown(f"🧑‍💻 **You:** {prompt}")
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            CUSTOM_PROMPT_TEMPLATE = """
            You are an AI assistant that keeps track of old conversations.
            Use the provided context and past interactions to generate a response.
            If you don't know the answer, just say that you don't know.

            Context: {context}
            Question: {question}
            """

            HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
            HF_TOKEN = os.environ.get("HF_TOKEN")

            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                sources = response["source_documents"]

                with st.expander("📜 Sources", expanded=False):
                    for doc in sources:
                        st.write(doc)

                st.chat_message('assistant').markdown(f"🤖 **MediBot:** {result}")
                st.session_state.messages.append({'role': 'assistant', 'content': result})

            except Exception as e:
                st.error(f"Error: {str(e)}")

    with col2:
        st.header("Recent Queries")
        for message in st.session_state.messages:
            role_icon = "🧑‍💻" if message['role'] == 'user' else "🤖"
            st.write(f"{role_icon} **{message['role'].capitalize()}**: {message['content']}")

if __name__ == "__main__":
    main()