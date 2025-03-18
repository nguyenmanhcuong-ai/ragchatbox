import os
import re
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import pypdf
import streamlit.components.v1 as components

import concurrent.futures
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import re
from functools import lru_cache
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Chỉ định đường dẫn tới file .env
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# Lấy GOOGLE_API_KEY từ biến môi trường
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY không được tìm thấy trong file .env.")
    raise ValueError("GOOGLE_API_KEY không được tìm thấy trong file .env")

# Khai báo DIRS trước khi sử dụng
DIRS = {
    "cache": Path("cache"),
    "data": Path("data"),
    "history": Path("chat_history"),
    "models": Path("models")
}

# Tạo các thư mục
for dir_path in DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class DocumentMetadata:
    title: str
    file_type: str
    upload_date: datetime
    page_count: int
    file_path: str
    category: str
    effective_date: Optional[str] = None
    issuing_authority: Optional[str] = None
    document_number: Optional[str] = None
    document_year: Optional[int] = None


class DocumentProcessor:
    def __init__(self, batch_size: int = 20):
        self.batch_size = batch_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\nĐiều", "\n\nChương", "\n\nMục", "\n\nPhần", 
                "\n\nPhụ lục", "\n\n", ".\n", "\n", ". ",
                ";\n", ";\\s", "Điều \\d+\\.", "Khoản \\d+\\.",
                "Điều \\d+\\.", "Khoản \\d+\\.", "Điểm [a-z]\\)", 
                "CHƯƠNG [IVX]+", "Mục \\d+", "Phần \\d+"
            ]
        )
        
        self.document_cache = {}
        self.chunk_cache = {}
        self.embedding_cache = {}
        self.text_cache = {}
        
        self._init_model()
        self._compile_patterns()

        
    def _init_model(self):
        try:
            self.model = SentenceTransformer('Cloyne/vietnamese-sbert-v3')
            self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            self.embeddings = HuggingFaceEmbeddings(model_name="Cloyne/vietnamese-sbert-v3")
        except Exception as e:
            logging.error(f"Model initialization error: {str(e)}")
            raise
            
    def _compile_patterns(self):
        """Cải thiện các pattern để nhận diện chính xác hơn"""
        self.patterns = {
            'document_number': re.compile(r'Số[: ]+(\d+/\d{4}/(?:NĐ-CP|TT-BTC|QĐ-TTg|UBTVQH|NQ-CP|QĐ-BTC|VBPQ|CT|VP|CV|TB))'),
            'issue_date': re.compile(r'ngày\s*(\d{1,2}\s*tháng\s*\d{1,2}\s*năm\s*\d{4})|ngày\s*(\d{1,2}/\d{1,2}/\d{4})'),
            'effective_date': re.compile(r'có\s*hiệu\s*lực\s*(từ|kể từ)\s*ngày\s*(\d{1,2}/\d{1,2}/\d{4})'),
            'authority': re.compile(r'(CHÍNH PHỦ|BỘ[^,\n]*|ỦY BAN[^,\n]*|QUỐC HỘI|THỦ TƯỚNG[^,\n]*)'),
            # Pattern cho cấu trúc pháp luật
            'article': re.compile(r'Điều\s+(\d+[a-z]?)\.?\s*([^.]+)'),
            'clause': re.compile(r'(?:^|\n\s*)(\d+)\.(?:\s+|$)([^.]+)'),
            'point': re.compile(r'(?:^|\n\s*)([a-z])\)(?:\s+|$)([^;.]+)'),
            'sub_point': re.compile(r'(?:^|\n\s*)-\s+([^;.]+)')
        }

    def process_document(self, file) -> Optional[tuple[List, DocumentMetadata]]:
        try:
            file_id = hash(file.name)
            
            # Return cached results if available
            if file_id in self.document_cache:
                return self.document_cache[file_id]
            
            # Save and load document
            file_path = DIRS["upload"] / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
                
            loader = PyPDFLoader(str(file_path)) if file.name.endswith('.pdf') else Docx2txtLoader(str(file_path))
            documents = loader.load()
            
            # Clean documents
            with concurrent.futures.ThreadPoolExecutor() as executor:
                clean_futures = [executor.submit(self.clean_vietnamese_text, doc.page_content) 
                               for doc in documents]
                for doc, future in zip(documents, clean_futures):
                    doc.page_content = future.result()
            
            # Extract metadata once
            metadata = self._extract_document_metadata(documents)
            
            # Split into chunks once
            chunks = self.text_splitter.split_documents(documents)
            
            # In các chunk ra terminal
            print("\n=== Kết quả chunk ===")
            for idx, chunk in enumerate(chunks):
                print(f"Chunk {idx+1}:")
                print(chunk.page_content[:500])  # In 500 ký tự đầu tiên của chunk (hoặc có thể điều chỉnh theo yêu cầu)
                print("----")

            
            # Process chunks with detailed metadata
            processed_chunks = self._process_chunks_batch(chunks, metadata)
            
            # Cache results
            result = (processed_chunks, metadata)
            self.document_cache[file_id] = result
            
            # Store chunks separately for reuse
            self.chunk_cache[file_id] = {
                chunk.metadata['chunk_id']: chunk for chunk in processed_chunks
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Document processing error: {str(e)}")
            return None, None
            
    def _process_chunks_batch(self, chunks: List, metadata: DocumentMetadata) -> List:
        """
        Args:
            chunks: List of document chunks to process.
            metadata: Metadata for the entire document.

        Returns:
            A list of processed chunks with enriched metadata.
        """
        processed_chunks = []
        total_chunks = len(chunks)

        # Use tqdm for a progress bar
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            with ThreadPoolExecutor() as executor:
                # Submit tasks for processing each chunk
                futures = {
                    executor.submit(self._process_chunk, chunk, idx, metadata): idx
                    for idx, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        # Process and collect the result
                        result = future.result()
                        processed_chunks.append(result)
                        
                        # Update progress and print status
                        pbar.update(1)
                        print(f"Chunk {chunk_idx} processed successfully.")
                    except Exception as e:
                        logging.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                        pbar.update(1)
        
        return processed_chunks

    def _process_chunk(self, text: Any, chunk_id: int, metadata: DocumentMetadata) -> Any:
        """
        Args:
            text: The content of the chunk to process.
            chunk_id: The unique identifier for the chunk.
            metadata: Metadata for the document.

        Returns:
            The processed chunk with updated metadata.
        """
        content = text.page_content

        # Extract legal references
        legal_refs = self._extract_legal_references(content)

        # Analyze penalties
        penalties = self._extract_penalties(content)

        # Determine section type
        section_type = self._detect_section_type(content)

        # Update metadata
        text.metadata.update({
            "chunk_id": chunk_id,
            "title": metadata.title,
            "document_type": metadata.category,
            "legal_references": legal_refs,
            "keywords": self._get_cached_keywords(content),
            "section_type": section_type,
            "penalties": penalties,
            "has_penalty": bool(penalties),
            "document_number": metadata.document_number,
            "document_year": metadata.document_year
        })

        # Print chunk details to the terminal
        print(f"Processing chunk {chunk_id}:")
        print(f"  Title: {metadata.title}")
        print(f"  Document Type: {metadata.category}")
        print(f"  Document Number: {metadata.document_number}")
        print(f"  Section Type: {section_type}")
        print(f"  Legal References: {legal_refs}")
        print(f"  Keywords: {self._get_cached_keywords(content)}")
        print(f"  Penalties: {penalties}")
        print(f"  Has Penalty: {bool(penalties)}")
        print("----------------------------------------------------")

        return text
        
    def get_cached_chunks(self, file_id: int) -> Optional[dict]:
        """Retrieve cached chunks for a file"""
        return self.chunk_cache.get(file_id)
        
    def clear_caches(self):
        """Clear all caches"""
        self.document_cache.clear()
        self.chunk_cache.clear()
        self.embedding_cache.clear()
        self.text_cache.clear()

    def _extract_document_metadata(self, documents: List) -> DocumentMetadata:
        """Cải thiện trích xuất metadata"""
        first_doc = documents[0]
        content = first_doc.page_content[:2000]  # Tăng phạm vi quét
        
        # Trích xuất số văn bản và năm
        doc_match = re.search(self.patterns['document_number'], content)
        doc_num = None
        doc_year = None
        if doc_match:
            doc_num = doc_match.group(1)
            try:
                doc_year = int(doc_num.split('/')[1])
            except (IndexError, ValueError):
                pass

        # Trích xuất ngày có hiệu lực
        eff_date = None
        eff_match = re.search(self.patterns['effective_date'], content)
        if eff_match:
            eff_date = eff_match.group(2)

        # Trích xuất cơ quan ban hành
        auth_match = re.search(self.patterns['authority'], content)
        authority = auth_match.group(1) if auth_match else None
        
        return DocumentMetadata(
            title=self._extract_title(documents),
            file_type=Path(first_doc.metadata.get('source', '')).suffix.lower()[1:],
            upload_date=datetime.now(),
            page_count=len(documents),
            file_path=first_doc.metadata.get('source', ''),
            category=self._detect_document_type(content),
            document_number=doc_num,
            document_year=doc_year,
            effective_date=eff_date,
            issuing_authority=authority
        )

    def _extract_title(self, documents: List) -> str:
        """Extract title with improved pattern matching"""
        if not documents:
            return ""
            
        content = documents[0].page_content[:1000]
        
        # Enhanced patterns for Vietnamese legal documents
        patterns = [
            r"^(?:LUẬT|NGHỊ ĐỊNH|THÔNG TƯ|QUYẾT ĐỊNH|NGHỊ QUYẾT)[\s\n:]+(.+?)(?=\n\n|\n(?:Số|Căn cứ|Theo|Xét)|\Z)",
            r"^(?:Về việc|V/v)[\s:]+(.+?)(?=\n|\Z)",
            r"^(.{10,150}?(?:luật|nghị định|thông tư|quyết định).+?)(?=\n|\Z)"
        ]
        
        for pattern in patterns:
            if match := re.search(pattern, content, re.I | re.M | re.S):
                return re.sub(r'\s+', ' ', match.group(1)).strip()
                
        return re.sub(r'\s+', ' ', content.split('\n')[0])[:100]

    
    @lru_cache(maxsize=1000)
    def _get_cached_keywords(self, content: str) -> List[str]:
        """Cache kết quả extraction keywords"""
        return self._extract_keywords(content)
        
    @lru_cache(maxsize=1000)
    def _get_cached_section_type(self, content: str) -> str:
        """Cache kết quả detect section type"""
        return self._detect_section_type(content)

    def clean_vietnamese_text(self, text: str) -> str:
        """Cải thiện làm sạch văn bản tiếng Việt"""
        if text in self.text_cache:
            return self.text_cache[text]
            
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        
        # Chuẩn hóa định dạng Điều
        text = re.sub(r'Điều\s+(\d+)', r'Điều \1.', text)
        
        # Chuẩn hóa định dạng Khoản
        text = re.sub(r'(?<=\n)(\d+)\s*\)', r'\1.', text)
        
        # Chuẩn hóa định dạng Điểm
        text = re.sub(r'(?<=\n)([a-z])\s*\)', r'\1)', text)
        
        # Sửa lỗi OCR phổ biến
        replacements = {
            r'nguòi': 'người',
            r'Diều': 'Điều',
            r'Chinh phủ': 'Chính phủ',
            r'nghị đinh': 'nghị định',
            r'(?<=\d)\.(?=\d)': ',',  # Sửa số thập phân
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
            
        result = text.strip()
        self.text_cache[text] = result
        return result


    def _detect_document_type(self, text: str) -> str:
        """Improved document type detection"""
        patterns = {
            'Luật': r'\b(?:LUẬT|Luật)\b',
            'Nghị định': r'\b(?:NGHỊ ĐỊNH|Nghị định)\b',
            'Thông tư': r'\b(?:THÔNG TƯ|Thông tư)\b',
            'Nghị quyết': r'\b(?:NGHỊ QUYẾT|Nghị quyết)\b',
            'Quyết định': r'\b(?:QUYẾT ĐỊNH|Quyết định)\b'
        }
        
        for doc_type, pattern in patterns.items():
            if re.search(pattern, text[:1000]):
                return doc_type
        return "Khác"

    def _extract_legal_references(self, text: str) -> List[dict]:
        """Enhanced legal reference extraction"""
        references = []
        
        patterns = {
            'article': r'Điều\s+(\d+)[A-Za-z]?',
            'clause': r'Khoản\s+(\d+)',
            'point': r'[Đ|d]iểm\s+([a-zA-Z])',
            'document_ref': r'(?:theo|tại|căn cứ)\s+([^.;]+(?:luật|nghị định|thông tư|quyết định)[^.;]+)'
        }
        
        for ref_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.I)
            for match in matches:
                references.append({
                    'type': ref_type,
                    'value': match.group(1),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
                
        return references

    def _extract_keywords(self, text: str) -> List[str]:
        """Cải thiện trích xuất từ khóa pháp lý"""
        legal_terms = [
            r'xử\s+phạt', r'vi\s+phạm', r'bắt\s+buộc', 
            r'quyền', r'nghĩa\s+vụ', r'trách\s+nhiệm',
            r'thẩm\s+quyền', r'giấy\s+phép', r'chứng\s+nhận',
            r'cấm', r'đình\s+chỉ', r'tước\s+quyền',
            r'tạm\s+đình\s+chỉ', r'thu\s+hồi', r'kiểm\s+tra',
            r'thanh\s+tra', r'khiếu\s+nại', r'tố\s+cáo'
        ]
        
        keywords = []
        for term in legal_terms:
            if re.search(rf'\b{term}\b', text, re.I):
                # Lấy ngữ cảnh xung quanh từ khóa
                matches = re.finditer(rf'\b{term}\b', text, re.I)
                for match in matches:
                    context = text[max(0, match.start()-30):min(len(text), match.end()+30)]
                    keywords.append({
                        'term': match.group(0),
                        'context': context.strip(),
                        'position': match.start()
                    })
                
        return keywords


    def _extract_penalties(self, text: str) -> List[dict]:
        """Trích xuất thông tin về hình phạt"""
        penalties = []
        
        # Pattern cho hình phạt tiền
        money_patterns = [
            r'(?:phạt\s+tiền|mức\s+phạt)\s*(?:từ\s*)?(\d+(?:\.\d+)?)\s*(?:đến\s*)?(\d+(?:\.\d+)?)?\s*(?:triệu\s*)?(?:đồng|VNĐ)',
            r'(?:phạt|nộp)\s*(?:từ\s*)?(\d+(?:\.\d+)?)\s*%\s*(?:đến\s*)?(\d+(?:\.\d+)?)?\s*%\s*(?:của|trên|trong)?',
        ]
        
        # Pattern cho hình phạt khác (bao gồm trừ điểm GPLX)
        other_patterns = [
            r'(đình\s*chỉ)\s*(?:hoạt\s*động)?\s*(?:trong\s*thời\s*hạn|thời\s*hạn)?\s*(\d+)\s*(?:tháng|năm)',
            r'(tước\s*quyền)\s*(?:sử\s*dụng)?\s*([^.]+?)\s*(?:trong\s*thời\s*hạn|thời\s*hạn)?\s*(\d+)\s*(?:tháng|năm)',
            r'(tịch\s*thu)\s*([^.]+)',
            r'(buộc)\s*([^.]+)',
            r'(thu\s*hồi)\s*([^.]+)',
            # Thêm pattern cho trừ điểm GPLX
            r'(trừ)\s*(\d+)\s*(?:điểm)?\s*(?:trên|trong|vào)?\s*(?:giấy\s*phép\s*lái\s*xe|GPLX|bằng\s*lái)',
        ]
        
        # Trích xuất hình phạt tiền
        for pattern in money_patterns:
            for match in re.finditer(pattern, text, re.I):
                penalties.append({
                    'type': 'monetary',
                    'amount': match.group(1),
                    'context': match.group(0),
                    'full_text': text[max(0, match.start()-50):match.end()+50]
                })
        
        # Trích xuất hình phạt khác (bao gồm trừ điểm GPLX)
        for pattern in other_patterns:
            for match in re.finditer(pattern, text, re.I):
                penalties.append({
                    'type': 'administrative',
                    'action': match.group(1),
                    'detail': match.group(2) if len(match.groups()) > 1 else None,
                    'context': match.group(0),
                    'full_text': text[max(0, match.start()-50):match.end()+50]
                })
        
        return penalties

    def _detect_section_type(self, text: str) -> str:
        """Cải thiện phát hiện loại mục"""
        type_patterns = {
            "Chế tài": r'xử\s+phạt|vi\s+phạm|chế\s+tài|trách\s+nhiệm\s+(?:hình\s+sự|hành\s+chính)|biện\s+pháp\s+(?:xử\s+lý|ngăn\s+chặn)',
            "Định nghĩa": r'định\s+nghĩa|giải\s+thích|khái\s+niệm|thuật\s+ngữ|quy\s+định\s+chung',
            "Phạm vi": r'phạm\s+vi|đối\s+tượng\s+(?:áp\s+dụng|điều\s+chỉnh)|không\s+áp\s+dụng',
            "Quy trình": r'trình\s+tự|thủ\s+tục|các\s+bước|quy\s+trình|phương\s+thức|cách\s+thức',
            "Tổ chức thực hiện": r'tổ\s+chức\s+thực\s+hiện|điều\s+khoản\s+thi\s+hành|hiệu\s+lực\s+thi\s+hành',
            "Quyền và nghĩa vụ": r'quyền|nghĩa\s+vụ|trách\s+nhiệm|nghĩa\s+vụ\s+của|quyền\s+của',
            "Điều khoản chuyển tiếp": r'chuyển\s+tiếp|điều\s+khoản\s+chuyển\s+tiếp|quy\s+định\s+chuyển\s+tiếp',
            "Thẩm quyền": r'thẩm\s+quyền|có\s+quyền|được\s+quyền|có\s+trách\s+nhiệm',
            "Giấy phép": r'giấy\s+phép|giấy\s+chứng\s+nhận|cấp\s+phép|đăng\s+ký',
            "Thanh tra kiểm tra": r'thanh\s+tra|kiểm\s+tra|giám\s+sát|kiểm\s+soát'

        }
        
        for section_type, pattern in type_patterns.items():
            if re.search(pattern, text, re.I):
                return section_type
                
        return "Nội dung chung"

class TrafficLawAssistant:
    """
    Trợ lý xử lý các câu hỏi liên quan đến Luật Giao thông Việt Nam.
    Sử dụng mô hình Generative AI của Google để trả lời.
    """

    def __init__(self):
        """Khởi tạo trợ lý với mô hình LLM và bộ nhớ cuộc trò chuyện."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            top_p=0.85,
            streaming=True,
            max_tokens=1024,
            system_prompt="Bạn là trợ lý thân thiện và am hiểu về Luật Giao thông đường bộ Việt Nam."
        )
        
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            k=3
        )

    def _create_prompt_template(self) -> PromptTemplate:
        """
        Tạo mẫu prompt cấu trúc cho các câu hỏi liên quan đến Luật Giao thông.
        
        Returns:
            PromptTemplate: Mẫu prompt được cấu hình
        """
        template = """
        Dựa vào các tài liệu sau:
        {context}

        Lịch sử trao đổi:
        {chat_history}

        Câu hỏi: {question}

        Hãy trả lời theo format sau:
                
        
        **PHÂN TÍCH TÌNH HUỐNG**
        [Mô tả ngắn gọn về tình huống và vấn đề chính]
        

        **QUY ĐỊNH PHÁP LUẬT LIÊN QUAN**
        [Chỉ rõ nghị định, điều, khoản, điểm áp dụng cùng giải thích ngắn gọn.]


        **KHUYẾN NGHỊ AN TOÀN**
       [Các khuyến nghị an toàn]

        ⚠️ **LƯU Ý QUAN TRỌNG**
        [Những điểm cần đặc biệt chú ý]
        """
        
        return PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template.strip()
        )
        

    def setup_chain(self, vectorstore):
        """
        Thiết lập chuỗi cuộc trò chuyện với tích hợp kho dữ liệu vector.
        
        Args:
            vectorstore: Kho dữ liệu vector để truy vấn tài liệu
            
        Returns:
            ConversationalRetrievalChain: Chuỗi cuộc trò chuyện được cấu hình
        """
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={
                "prompt": self._create_prompt_template()
            },
            return_source_documents=True,
            verbose=False
        )
        return chain

    def _format_section_content(self, content: str) -> str:
        """
        Định dạng nội dung của từng phần với dòng xuống đúng cách.
        
        Args:
            content (str): Nội dung văn bản cần định dạng
            
        Returns:
            str: Nội dung được định dạng với dòng xuống phù hợp
        """
        # Tách văn bản thành các câu
        sentences = re.split(r'(?<=[.!?])\s+', content)
        # Loại bỏ các câu trống và ghép lại với dòng xuống
        return '\n'.join(sent.strip() for sent in sentences if sent.strip())

    def enhance_response(self, response: str) -> str:
        """
        Tối ưu hóa định dạng câu trả lời với phong cách chuyên nghiệp, xử lý chính xác ký tự **.

        Args:
            response (str): Văn bản câu trả lời thô
            
        Returns:
            str: Câu trả lời được định dạng và tối ưu hóa
        """
        # Các tiêu đề phần
        sections = [
            'PHÂN TÍCH TÌNH HUỐNG',
            'QUY ĐỊNH PHÁP LUẬT LIÊN QUAN',
            'KHUYẾN NGHỊ AN TOÀN',
            'LƯU Ý QUAN TRỌNG'
        ]

        # Chuẩn hóa toàn bộ văn bản trước khi xử lý
        enhanced = self._normalize_bold_markers(response)

        # Định dạng tiêu đề phần
        for section in sections:
            pattern = rf"{section}:"
            replacement = rf"\n\n{section}\n"
            enhanced = enhanced.replace(pattern, replacement)

        # Tách nội dung thành các phần
        sections_content = {}
        current_section = None
        lines = enhanced.split('\n')
        current_content = []

        for line in lines:
            if any(section in line for section in sections):
                if current_section and current_content:
                    sections_content[current_section] = '\n'.join(current_content)
                current_section = line.strip() + ': '
                current_content = []
            elif line.strip():
                current_content.append(line)

        if current_section and current_content:
            sections_content[current_section] = '\n'.join(current_content)

        # Định dạng nội dung từng phần
        formatted_sections = []
        for section, content in sections_content.items():
            # Định dạng bullet point
            if '•' in content:
                bullet_points = content.split('•')
                formatted_bullets = []
                for point in bullet_points:
                    if point.strip():
                        formatted_point = self._format_section_content(point.strip())
                        formatted_bullets.append(f"• {formatted_point}")
                formatted_content = '\n'.join(formatted_bullets)
            else:
                formatted_content = self._format_section_content(content)

            formatted_sections.append(f"{section}\n{formatted_content}")

        # Kết hợp các phần trở lại
        enhanced = '\n\n'.join(formatted_sections)

        # Áp dụng quy tắc định dạng bổ sung
        formatting_rules = [
            # Định dạng danh sách số thứ tự với khoảng cách thích hợp
            (r'^(\d+)\.\s', lambda m: f"{m.group(1)}. ", re.MULTILINE),
            
            # Làm nổi bật các trích dẫn điều luật
            (r'(Điều \d+[^()]*?(?:, Khoản \d+[^()]*?)?(?:, Điểm [a-z][^()]*?)?\))', r'<strong>\1</strong>', 0),
            
            # Làm nổi bật ghi chú quan trọng
            (r'(!.*?[.!?])', r'<strong>\1</strong>', 0),
            
            # Làm sạch không gian thừa
            (r'\n{3,}', '\n\n', 0)
        ]

        # Áp dụng từng quy tắc định dạng
        for pattern, replacement, flags in formatting_rules:
            enhanced = re.sub(pattern, replacement, enhanced, flags=flags)

        return enhanced.strip()

    def _normalize_bold_markers(self, text: str) -> str:
        """
        Chuẩn hóa ký tự ** trong văn bản, xóa tất cả ** không nằm trong cặp hợp lệ và chuyển thành HTML.

        Args:
            text (str): Văn bản cần chuẩn hóa
            
        Returns:
            str: Văn bản đã được chuẩn hóa với <strong> thay vì **
        """
        result = []
        i = 0
        length = len(text)

        while i < length:
            if i + 1 < length and text[i:i+2] == '**':
                # Tìm vị trí đóng của cặp **
                start = i
                i += 2
                found_close = False
                content = []
                
                while i < length:
                    if i + 1 < length and text[i:i+2] == '**':
                        found_close = True
                        i += 2
                        break
                    content.append(text[i])
                    i += 1
                
                content_str = ''.join(content)
                # Chỉ giữ định dạng <strong> nếu nội dung là điều luật, ghi chú, hoặc tiêu đề
                if found_close and content and (
                    content_str.startswith('Điều ') or 
                    content_str.startswith('!') or 
                    any(section in content_str for section in [
                        'PHÂN TÍCH TÌNH HUỐNG', 
                        'QUY ĐỊNH PHÁP LUẬT LIÊN QUAN', 
                        'KHUYẾN NGHỊ AN TOÀN', 
                        'LƯU Ý QUAN TRỌNG'
                    ])
                ):
                    result.append(f"<strong>{content_str}</strong>")
                else:
                    result.append(content_str)
            else:
                result.append(text[i])
                i += 1
        
        return ''.join(result)

    def get_response(self, question: str, vectorstore) -> str:
        """
        Lấy câu trả lời từ trợ lý và tối ưu hóa định dạng.
        
        Args:
            question (str): Câu hỏi của người dùng
            vectorstore: Kho dữ liệu vector để truy vấn tài liệu
            
        Returns:
            str: Câu trả lời đã được tối ưu hóa định dạng
        """
        chain = self.setup_chain(vectorstore)
        result = chain.run(question)
        enhanced_response = self.enhance_response(result['answer'])
        return enhanced_response


class DocumentManager:
    def __init__(self, doc_processor):
        self.doc_processor = doc_processor
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = {}
        if 'failed_files' not in st.session_state:
            st.session_state.failed_files = []
            
    def load_files_from_data_directory(self) -> bool:
        """
        Load and process all files from the data directory automatically
        """
        success = False
        data_dir = Path("data")
        
        # Create data directory if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all PDF and DOCX files from data directory
        files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.docx"))
        
        if not files:
            st.warning("📂 Không tìm thấy tài liệu trong thư mục data. Vui lòng thêm file PDF hoặc DOCX vào thư mục.")
            return False
            
        for file_path in files:
            try:
                with st.status(f"📄 Đang xử lý: {file_path.name}") as status:
                    file_id = hash(file_path.name)
                    
                    if file_id in st.session_state.processed_files:
                        status.info(f"ℹ️ Sử dụng dữ liệu đã xử lý: {file_path.name}")
                        continue
                    
                    # Process document directly from file path
                    texts, metadata = self.process_file(file_path)
                    
                    if texts and metadata:
                        st.session_state.processed_files[file_id] = {
                            'texts': texts,
                            'metadata': metadata,
                            'name': file_path.name
                        }
                        status.success(f"✅ Xử lý thành công: {file_path.name}")
                        success = True
                    else:
                        st.session_state.failed_files.append(file_path.name)
                        status.error(f"❌ Không thể xử lý: {file_path.name}")
                
            except Exception as e:
                st.session_state.failed_files.append(file_path.name)
                st.error(f"❌ Lỗi khi xử lý {file_path.name}: {str(e)}")
        
        if st.session_state.processed_files:
            st.success(f"""
            📊 Tổng kết xử lý:
            - Số file đã xử lý thành công: {len(st.session_state.processed_files)}
            - Số file thất bại: {len(st.session_state.failed_files)}
            """)
            
        return success and len(st.session_state.processed_files) > 0
    
    def process_file(self, file_path: Path) -> tuple:
        """Process a single file from disk"""
        try:
            # Determine loader based on file extension
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.docx':
                loader = Docx2txtLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
                
            documents = loader.load()
            
            # Clean documents
            with concurrent.futures.ThreadPoolExecutor() as executor:
                clean_futures = [executor.submit(self.doc_processor.clean_vietnamese_text, doc.page_content) 
                               for doc in documents]
                for doc, future in zip(documents, clean_futures):
                    doc.page_content = future.result()
            
            # Extract metadata
            metadata = self.doc_processor._extract_document_metadata(documents)
            
            # Split into chunks
            chunks = self.doc_processor.text_splitter.split_documents(documents)
            
            # Process chunks with detailed metadata
            processed_chunks = self.doc_processor._process_chunks_batch(chunks, metadata)
            
            return processed_chunks, metadata
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return None, None

    
    def get_processed_data(self):
        all_texts = []
        all_metadata = []
        
        for file_data in st.session_state.processed_files.values():
            all_texts.extend(file_data['texts'])
            all_metadata.append(file_data['metadata'])
            
        return all_texts, all_metadata
    
    def clear_data(self):
        st.session_state.processed_files = {}
        st.session_state.failed_files = []
        self.doc_processor.chunk_cache.clear()
        self.doc_processor.embedding_cache.clear()

def setup_suggested_questions():
    """Setup suggested questions for the chat interface"""
    return [
        {
            "category": "Vi phạm & Xử phạt",
            "questions": [
                "Lỗi ô tô vượt đèn đỏ bị phạt bao nhiêu tiền?",
                "Điều khiển xe máy không có bằng lái bị phạt như thế nào?",
                "Các mức xử phạt khi lái xe máy có nồng độ cồn?",
                "Mức phạt khi không đội mũ bảo hiểm?",
                "Xe máy đi quá tốc độ xử phạt thế nào?"
            ]
        },
        {
            "category": "Giấy tờ & Thủ tục",
            "questions": [
                "Thủ tục đăng ký xe máy mới cần những gì?",
                "Hồ sơ thi bằng lái xe máy gồm những gì?",
                "Thời hạn đăng kiểm xe ô tô là bao lâu?",
                "Cách tra cứu phạt nguội online?",
                "Thủ tục sang tên xe máy khác tỉnh?"
            ]
        },
        {
            "category": "Quy định & Điều luật",
            "questions": [
                "Quy định về độ tuổi được điều khiển xe máy?",
                "Các trường hợp bị tước bằng lái xe?",
                "Quy định về tốc độ trong khu dân cư?",
                "Điều kiện để được cấp bằng lái xe?",
                "Quy định về số người chở trên xe máy?"
            ]
        }
    ]

def render_search_interface():
    """Render a collapsible search interface with suggestions"""
    suggested_questions = setup_suggested_questions()
    
    # Custom Component for Search Interface
    search_interface_html = """
    <style>
        .search-panel {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
            overflow: hidden;
        }
        
        .search-header {
            padding: 16px;
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
            font-weight: 600;
            color: #1e40af;
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
        }
        
        .questions-container {
            padding: 12px;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .category-accordion {
            margin-bottom: 8px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .category-header {
            padding: 12px 16px;
            background: #f8fafc;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 500;
            color: #475569;
        }
        
        .category-header:hover {
            background: #f1f5f9;
        }
        
        .category-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background: white;
        }
        
        .category-content.active {
            max-height: 500px;
            padding: 12px;
        }
        
        .question-item {
            padding: 10px 16px;
            margin: 4px 0;
            border-radius: 6px;
            cursor: pointer;
            color: #0369a1;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }
        
        .question-item:hover {
            background: #f0f9ff;
        }
        
        .question-item:active {
            background: #e0f2fe;
        }
        
        .toggle-icon {
            transition: transform 0.3s ease;
        }
        
        .category-header.active .toggle-icon {
            transform: rotate(180deg);
        }
        
        /* Scrollbar styling */
        .questions-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .questions-container::-webkit-scrollbar-track {
            background: #f1f5f9;
        }
        
        .questions-container::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 3px;
        }
    </style>
    
    <div class="search-panel">
        <div class="search-header" id="mainToggle">
            <span>💡</span> Câu hỏi thường gặp theo chủ đề
            <span class="toggle-icon">▼</span>
        </div>
        <div class="questions-container" id="questionCategories">
        </div>
    </div>
    """
    
    # Convert suggested questions to JavaScript
    questions_js = f"const suggestedQuestions = {str(suggested_questions).replace('True', 'true').replace('False', 'false')};"
    
    components.html(
        search_interface_html + 
        f"""
        <script>
        {questions_js}
        
        function triggerChatSubmit(question) {{
            // Tìm input field của chat
            const chatInputs = window.parent.document.querySelectorAll('textarea');
            const chatInput = Array.from(chatInputs).find(input => 
                input.placeholder && input.placeholder.includes('Hỏi về nội dung')
            );
            
            if (chatInput) {{
                // Set giá trị cho input
                const nativeTextAreaValue = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
                const nativeInputValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
                
                function setNativeValue(element, value) {{
                    const valueSetter = Object.getOwnPropertyDescriptor(element, 'value')?.set 
                        || nativeTextAreaValue?.set 
                        || nativeInputValue?.set;
                        
                    if (valueSetter) {{
                        const prototype = Object.getPrototypeOf(element);
                        const prototypeValueSetter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
                        
                        if (valueSetter !== prototypeValueSetter) {{
                            prototypeValueSetter.call(element, value);
                        }} else {{
                            valueSetter.call(element, value);
                        }}
                    }}
                }}
                
                setNativeValue(chatInput, question);
                chatInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                
                // Tìm và click nút submit
                const buttons = window.parent.document.querySelectorAll('button');
                const sendButton = Array.from(buttons).find(button => 
                    button.innerHTML.includes('↵')
                );
                
                if (sendButton) {{
                    sendButton.click();
                }}
            }}
        }}

        function createQuestionElement(question) {{
            const div = document.createElement('div');
            div.className = 'question-item';
            div.innerHTML = `<span>❓</span>${{question}}`;
            
            div.onclick = (event) => {{
                event.stopPropagation();
                div.style.background = '#bae6fd';
                setTimeout(() => {{
                    div.style.background = '';
                }}, 200);
                
                triggerChatSubmit(question);
            }};
            
            return div;
        }}
        
        // Render categories function
        function renderCategories() {{
            const categoriesContainer = document.getElementById('questionCategories');
            
            suggestedQuestions.forEach(category => {{
                const categoryDiv = document.createElement('div');
                categoryDiv.className = 'category-accordion';
                
                const header = document.createElement('div');
                header.className = 'category-header';
                header.innerHTML = `
                    ${{category.category}}
                    <span class="toggle-icon">▼</span>
                `;
                
                const content = document.createElement('div');
                content.className = 'category-content';
                
                header.onclick = (event) => {{
                    event.stopPropagation();
                    header.classList.toggle('active');
                    content.classList.toggle('active');
                }};
                
                category.questions.forEach(question => {{
                    content.appendChild(createQuestionElement(question));
                }});
                
                categoryDiv.appendChild(header);
                categoryDiv.appendChild(content);
                categoriesContainer.appendChild(categoryDiv);
            }});
        }}
        
        // Initial render
        renderCategories();
        </script>
        """,
        height=600
    )
   
# Must be the first Streamlit command
st.set_page_config(
    page_title="Trợ Lý Pháp luật",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def setup_styles():
    st.markdown("""
        <style>
            /* Base styles */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            * {
                font-family: 'Inter', sans-serif;
            }
            
            /* Header styling */
            .header {
                background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            
            .app-title {
                color: white;
                font-size: 2.5rem;
                font-weight: 700;
                text-align: center;
                margin-bottom: 0.5rem;
            }
            
            .app-subtitle {
                color: rgba(255, 255, 255, 0.9);
                text-align: center;
                font-size: 1.1rem;
            }
            
            /* Chat interface */
            .chat-container {
                border-radius: 12px;
                background: #f8fafc;
                padding: 1rem;
                margin-bottom: 1rem;
            }
            
            .chat-message {
                padding: 1rem;
                margin-top: -11px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
             
            
            .assistant-message {
                background: white;
                margin-right: 2rem;
            }
            
            /* Chat history styling */
            .chat-history-button {
                background: transparent;
                border: none;
                padding: 0.75rem;
                border-radius: 8px;
                text-align: left;
                width: 100%;
                transition: all 0.2s ease;
                color: #1e293b;
                font-size: 0.95rem;
            }
            
            .chat-history-button:hover {
                background: #f1f5f9;
            }
            
            .new-chat-button {
                background: #3b82f6 !important;
                color: white !important;
                padding: 0.75rem !important;
                border-radius: 8px !important;
                font-weight: 500 !important;
                margin-bottom: 1rem !important;
                text-align: center !important;
            }
            
            .new-chat-button:hover {
                background: #2563eb !important;
            }
            
            /* Source citations */
            .source-citation {
                background: #f1f5f9;
                padding: 0.5rem;
                border-radius: 6px;
                margin-top: 0.5rem;
                font-size: 0.9rem;
                color: #475569;
            }
            
            /* Welcome section */
            .welcome-section {
                background: #f8fafc;
                padding: 2rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }
            
            .welcome-text {
                font-size: 1.1rem;
                line-height: 1.6;
                color: #1e293b;
                text-align: center;
            }
            
            /* Control buttons */
            .control-button {
                background: #ef4444 !important;
                color: white !important;
                padding: 0.6rem !important;
                border-radius: 8px !important;
                font-size: 0.9rem !important;
                font-weight: 500 !important;
            }
            
            .control-button:hover {
                background: #dc2626 !important;
            }
            
            /* Document section */
            .document-info {
                background: #f8fafc;
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 0.5rem;
                border: 1px solid #e2e8f0;
            }
            
            /* Sidebar improvements */
            .css-1d391kg {
                padding: 2rem 1rem;
            }
            
            .sidebar-title {
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: #1e293b;
            }
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state only on first load"""
    # Use a flag to track if initialization has been done
    if "initialized" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.current_chat = None
        st.session_state.processed = False
        st.session_state.initialized = True

def create_new_chat():
    """Create new chat with minimal checks"""
    # Since we know chats exists after initialization, we can directly create new chat
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.chats[chat_id] = {
        "title": "Cuộc hội thoại mới",
        "messages": []
    }
    return chat_id
def render_header():
    st.markdown("""
        <div class="header">
            <div class="app-title">🚦 Trợ Lý Pháp Luật Giao Thông</div>
            <div class="app-subtitle">Tra cứu và tư vấn luật giao thông thông minh</div>
        </div>
    """, unsafe_allow_html=True)

def main():
    setup_styles()
    
    # Initialize components
    doc_processor = DocumentProcessor()
    doc_manager = DocumentManager(doc_processor)
    traffic_assistant = TrafficLawAssistant()
    initialize_session_state()
    
    # Render header
    render_header()

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📚 Tài Liệu Pháp Luật</div>', unsafe_allow_html=True)

        # Document processing section
        if not st.session_state.get("processed", False):
            with st.spinner("Đang tải và xử lý tài liệu..."):
                if doc_manager.load_files_from_data_directory():
                    all_texts, metadata_list = doc_manager.get_processed_data()
                    vectorstore = FAISS.from_documents(all_texts, doc_processor.embeddings)
                    st.session_state.chain = traffic_assistant.setup_chain(vectorstore)
                    st.session_state.document_metadata = metadata_list
                    st.session_state.processed = True

        # Document display
        if "document_metadata" in st.session_state:
            with st.expander("📁 Tài Liệu Đã Xử Lý", expanded=True):
                for meta in st.session_state.document_metadata:
                    st.markdown(f'''
                        <div class="document-info">
                            <div style="font-weight: 500;">📄 {meta.title}</div>
                            <div>📋 Loại: {meta.file_type}</div>
                            <div>📑 Số trang: {meta.page_count}</div>
                            <div>📅 Ngày xử lý: {meta.upload_date.strftime('%Y-%m-%d %H:%M')}</div>
                        </div>
                    ''', unsafe_allow_html=True)

        # Chat history section
        st.divider()
        st.markdown('<div class="sidebar-title">💬 Lịch sử chat</div>', unsafe_allow_html=True)
        
        # New chat button
        if st.button("➕ Cuộc hội thoại mới", key="new_chat", use_container_width=True):
            st.session_state.current_chat = create_new_chat()
            st.rerun()

        # Display chat history
        for chat_id, chat_data in reversed(st.session_state.chats.items()):
            if st.button(
                f"💬 {chat_data['title'][:30]}...",
                key=f"chat_{chat_id}",
                use_container_width=True
            ):
                st.session_state.current_chat = chat_id
                st.rerun()

        # Control buttons
        st.divider()
        cols = st.columns(2)
        with cols[0]:
            if st.button("🗑️ Xóa Lịch Sử", use_container_width=True, key="clear_history"):
                st.session_state.chats = {}
                st.session_state.current_chat = None
                st.success("Đã xóa lịch sử chat!")
                st.rerun()

        with cols[1]:
            if st.button("🔄 Tải Lại", use_container_width=True, key="reload_docs"):
                if "document_metadata" in st.session_state:
                    del st.session_state.document_metadata
                if "chain" in st.session_state:
                    del st.session_state.chain
                if hasattr(doc_manager, 'clear_data'):
                    doc_manager.clear_data()
                st.session_state.processed = False
                st.success("Đang tải lại tài liệu...")
                st.rerun()

    # Chat interface remains the same
    if st.session_state.get("processed", False):
        # Initialize current chat if needed
        if not st.session_state.current_chat:
            st.session_state.current_chat = create_new_chat()

        current_chat = st.session_state.chats[st.session_state.current_chat]
        
        # Render search interface with suggestions
        render_search_interface()

        # Display messages for current chat
        for message in current_chat["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(
                    f"<div class='chat-message {'user-message' if message['role'] == 'user' else 'assistant-message'}'>"
                    f"{message['content']}</div>",
                    unsafe_allow_html=True
                )

        if prompt := st.chat_input("Hỏi về nội dung tài liệu..."):
            # Update chat title with first user message if it's new
            if len(current_chat["messages"]) == 0:
                current_chat["title"] = prompt[:50]

            # Add user message
            current_chat["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(
                    f"<div class='chat-message user-message'>{prompt}</div>",
                    unsafe_allow_html=True
                )

            # Process and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Đang phân tích..."):
                    response = st.session_state.chain({"question": prompt})
                    answer = response["answer"]
                    enhanced_answer = traffic_assistant.enhance_response(answer)

                    referenced_files = set()
                    for doc in response.get("source_documents", []):
                        file_path = doc.metadata.get('source', '')
                        if file_path:
                            file_name = file_path.split('/')[-1]
                            referenced_files.add(file_name)

                    sources = "\n\n**Tham khảo:**"
                    for file_name in referenced_files:
                        sources += f"\n<div class='source-citation'>📄 {file_name}</div>"

                    enhanced_answer = enhanced_answer.replace('**', '<strong>', 1)
                    enhanced_answer = enhanced_answer.replace('**', '</strong>', 1)

                    full_response = f"<div class='chat-message assistant-message'><p>{enhanced_answer}</p>{sources}</div>"
                    st.markdown(full_response, unsafe_allow_html=True)

                    # Add assistant response to chat history
                    current_chat["messages"].append({
                        "role": "assistant",
                        "content": f"{enhanced_answer}{sources}"
                    })
    else:
        st.info("👆 Hãy tải tài liệu lên để bắt đầu.")

def custom_css():
    st.markdown("""
        <style>
        .stButton button {
            text-align: left;
            background-color: transparent;
            border: none;
            padding: 10px;
            line-height: 1.2;
        }
        .stButton button:hover {
            background-color: #f0f2f6;
            border-radius: 5px;
        }
        .source-citation {
            margin-top: 5px;
            color: #666;
            font-size: 0.9em;
        }
        .welcome-section {
            padding: 1rem;
            border-radius: 0.5rem;
            background: #f8f9fa;
            margin-bottom: 1rem;
        }
        .welcome-text {
            font-size: 1.1rem;
            line-height: 1.5;
            color: #1e1e1e;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    custom_css()
    main()