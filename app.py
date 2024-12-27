import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import gemini  # Đừng quên import thư viện Gemini
import openai
st.set_page_config(page_title="Law Chatbot", layout="wide")

# Khóa API
gemini.api_key = "YOUR_API_KEY_HERE"  # Cập nhật API Key của Gemini

# Load chỉ mục FAISS
vector_store = FAISS.load_local("law_faiss_index", OpenAIEmbeddings(openai_api_key=openai.api_key))

# Hàm tạo câu trả lời
def generate_answer(question):
    # Truy xuất tài liệu liên quan
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Tạo câu trả lời từ API Gemini
    response = gemini.ChatCompletion.create(
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
