import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import openai

# Khóa API
openai.api_key = "AIzaSyCef1vFxl-W-8Jq9_T271JjyGQpeqsYstI"

# Load chỉ mục FAISS
vector_store = FAISS.load_local("law_faiss_index", OpenAIEmbeddings())

# Hàm tạo câu trả lời
def generate_answer(question):
    # Truy xuất tài liệu liên quan
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Tạo câu trả lời từ API Gemini
    response = openai.ChatCompletion.create(
        model="gemini",
        messages=[
            {"role": "system", "content": "Bạn là luật sư thông thái."},
            {"role": "user", "content": f"Câu hỏi: {question}\n\nThông tin pháp luật liên quan:\n{context}"}
        ]
    )
    return response['choices'][0]['message']['content']

# Giao diện Streamlit
st.title("Chatbot Hỏi Đáp Pháp Luật")
st.write("Nhập câu hỏi liên quan đến pháp luật để nhận câu trả lời.")

# Nhập câu hỏi
question = st.text_input("Câu hỏi của bạn:")
if st.button("Gửi"):
    if question:
        with st.spinner("Đang xử lý..."):
            answer = generate_answer(question)
        st.success("Trả lời:")
        st.write(answer)
