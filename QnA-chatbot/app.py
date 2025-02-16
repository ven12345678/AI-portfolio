from transformers import pipeline
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#Initialize model
qa_pipeline = pipeline("question-answering", model = "distilbert-base-cased-distilled-squad")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


#Streamlit interface
st.title("LLM-Powered Q&A Chatbot with Context Memory")

#Document upload and processing 
uploaded_file = st.file_uploader("Upload a text file", type = "txt")

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    sentences = text.split(". ")
    embeddings = embedder.encode(sentences)

    #Building FAISS Index
    dimensions = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    index.add(np.array(embeddings))

    st.success("Document processed! Ask me anything about it")
    user_question = st.text_input("Ask a question: ")

    if st.button("Get answer") and user_question:
        #Embed the question
        question_embedding = embedder.encode([user_question])

        #Search for the most relevant sentence
        _, nearest = index.search(np.array(question_embedding), 1)
        relevant_sentences = sentences[nearest[0][0]]

        #Use LLM to answer based on context
        result = qa_pipeline({"question": user_question, "context": relevant_sentences})

        st.subheader("Answer")
        st.write(result["answer"])
