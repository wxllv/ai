# ====================== 标准库导入 ======================
import os
import time
import numpy as np
from pdf2image import convert_from_path  # PDF转图片工具
from typing import List,Iterator # 类型注解支持
import torch
from PIL import Image
from paddleocr import PaddleOCR
from 文本ocr扫描 import ScanPDFLoader
# ====================== 第三方库导入 ======================
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 文本分割器
from langchain_experimental.text_splitter import SemanticChunker
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # 提示词模板
from langchain.schema.runnable import RunnablePassthrough  # 可运行链
from langchain.schema.output_parser import StrOutputParser  # 输出解析器
from langchain_core.documents import Document  # 文档对象
from langchain.agents import initialize_agent, Tool  # 代理系统
from langchain.agents.agent_types import AgentType  # 代理类型
from langchain.memory import ConversationBufferMemory  # 对话记忆
from search import BaiduSearcher
from Googlesearch import GoogleSearcher
from langchain_community.chat_models import ChatOpenAI

# ====================== 向量数据库相关 ======================
from langchain_chroma import Chroma  # Chroma向量数据库
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # 嵌入模型
from langchain.retrievers import BM25Retriever, EnsembleRetriever
#========================web===================================
from flask import Flask, request, jsonify, render_template_string, redirect
import os, re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'NULL知识库'

# 你的 RAG 系统或其他 query 对象
system = None

chat_history = []

def custom_secure_filename(name):
    filename = os.path.basename(name).replace(' ', '_')
    return re.sub(r'[\\/*?:"<>|]', '', filename)

@app.route('/')
def index():
    # 读取并渲染 index.html
    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()
    return render_template_string(html, chat_history=chat_history, documents=get_document_list())

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    q = data.get('question', '').strip()
    if not q:
        return jsonify({'error': 'empty question'}), 400

    chat_history.append({'role': 'user', 'content': q})
    ans = system.query(q) if system else '这是模拟的系统回答。'
    chat_history.append({'role': 'system', 'content': ans})
    return jsonify({'answer': ans})

@app.route('/upload', methods=['POST'])
def handle_upload():
    file = request.files.get('file')
    if not file or not file.filename.lower().endswith('.pdf'):
        return redirect('/')
    fname = custom_secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(path)
    if system:
        system.add_documents([path])
    return redirect('/')

@app.route('/delete', methods=['POST'])
def handle_delete():
    fname = request.form.get('filename')
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    if system:
        system.remove_document(path)
    if os.path.exists(path):
        os.remove(path)
    return redirect('/')


def get_document_list():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    return [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]

def run_flask():
    app.run(host='0.0.0.0', port=5000)

# ====================== RAG系统核心类 ======================
class RAGSystem:
    """基于检索增强生成(RAG)的问答系统"""
    
    def __init__(self, config: dict):
        """
        初始化RAG系统
        :param config: 系统配置字典
        """
        self.config = config
        # 初始化对话记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        # 初始化搜索引擎工具
        # 初始化所有组件
        self._init_components()
        

    # --------------------- 初始化组件 ---------------------
    def _init_components(self):
        """初始化系统所有核心组件"""
        # 1. 文档处理器
        self.pdf_loader = ScanPDFLoader()
        
        # 2. 文本分割器(用于将长文档切分为小块)
        self.text_splitter =self._init_combined_splitter()
        
        # 3. 向量数据库初始化
        if not os.path.exists(self.config['persist_dir']):
            self._init_vector_db()
      
        # 加载已存在的向量数据库
        self.vector_store = Chroma(
            persist_directory=self.config['persist_dir'],
            embedding_function=HuggingFaceBgeEmbeddings(
                model_name="./models/bge-small-zh-v1.5"
            ),
            collection_metadata={  # 新增HNSW调优参数
                "hnsw:space": "cosine",
                "hnsw:M": 32,             
                "hnsw:ef_construction": 200,
                 "hnsw:ef": 200
            }
            
)
        
        # 4. 大语言模型初始化
        self.llm = ChatOpenAI(
                    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://maas-api.cn-huabei-1.xf-yun.com/v1",
                    model="xdeepseekv32",
                    temperature=0.3,
                    streaming=True
                )
        # 5. 初始化混合检索器
        self._init_hybrid_retriever()
        self._init_reranker()
        # 7. 初始化智能代理
        self.rag_chain = self._build_optimized_rag_chain()
        self.agent = self._build_agent()
    #-----------文档分割---------------------------
    def _init_base_splitter(self):
        """初始化基础字符分割器"""
        return RecursiveCharacterTextSplitter(
            chunk_size=1500,  # 较大的初始块
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", ""],
            keep_separator=True
        )

    def _init_semantic_splitter(self):
        """初始化语义分割器"""
        return SemanticChunker(
            embeddings=HuggingFaceBgeEmbeddings(
                model_name="./models/bge-small-zh-v1.5"
            ),
            buffer_size=5,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.6,
            sentence_split_regex=r"(?<=[。！？])\s+",
            min_chunk_size=800,
        )

    def _init_combined_splitter(self):
        """初始化组合分割器：先字符分割，再语义分割"""
        base_splitter = self._init_base_splitter()
        semantic_splitter = self._init_semantic_splitter()
        class CombinedSplitter:
            def __init__(self, base_splitter, semantic_splitter):
                self.base_splitter = base_splitter
                self.semantic_splitter = semantic_splitter
                
            def split_documents(self, documents):
                # 处理文档列表
                all_chunks = []
                for doc in documents:
                    # 先进行字符分割
                    base_chunks = self.base_splitter.split_documents([doc])
                    # 对每个字符分割块再进行语义分割
                    for chunk in base_chunks:
                        all_chunks.extend(self.semantic_splitter.split_documents([chunk]))
                return all_chunks
        
        return CombinedSplitter(base_splitter, semantic_splitter)
    # --------------------- 混合检索器 ---------------------
    def _init_hybrid_retriever(self):
        """初始化混合检索器：结合语义检索与关键词检索，提升召回率"""
            # 从Chroma获取原始数据
        db_data = self.vector_store.get()
        
        # 重建Document对象列表
        documents = [
            Document(
                page_content=content,
                metadata=meta
            ) for content, meta in zip(
                db_data['documents'],
                db_data['metadatas']
            )
        ]

        # 初始化BM25检索器
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 20
        
        # 集成检索器
        self.base_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_store.as_retriever(search_kwargs={"k": 20}),
                bm25_retriever
            ],
            weights=[0.5, 0.5],
            k=15
        )
    def _init_reranker(self):
        """初始化重排序组件：使用LLM对检索结果进行相关性过滤"""
        from langchain.retrievers import ContextualCompressionRetriever
        from rerankers import Reranker
        ranker = Reranker(
        "BAAI/bge-reranker-base",  # 从base换成small
        dtype='fp16',
        batch_size=512
    )
        compressor=ranker.as_langchain_compressor(k=10)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.base_retriever,

        )
    # --------------------- 向量数据库初始化 ---------------------
    def _init_vector_db(self):
        """初始化并填充向量数据库"""
        print("正在初始化向量数据库...")
        all_chunks = []
        
        # 处理PDF路径(兼容单个路径和路径列表)
        pdf_paths = [self.config['pdf_paths']] if isinstance(self.config['pdf_paths'], str) else self.config['pdf_paths']

        for pdf_path in pdf_paths:
            print(f"正在处理: {pdf_path}")
            try:
                # 加载并分割PDF文档
                docs = self.pdf_loader.load(pdf_path)
                chunks = self.text_splitter.split_documents(docs)
                
                # 添加文档元数据
                for chunk in chunks:
                    chunk.metadata.update({
                        'source': pdf_path,
                        'filename': os.path.basename(pdf_path)
                    })
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"处理 {pdf_path} 时出错: {str(e)}")
                continue
        
        if not all_chunks:
            raise ValueError("没有成功加载任何PDF文档")
        
        # 创建新的向量数据库
        Chroma.from_documents(
            documents=all_chunks,
            embedding=HuggingFaceBgeEmbeddings(
                model_name="./models/bge-small-zh-v1.5",
                 model_kwargs={"device": "cuda"}  
            ),
            persist_directory=self.config['persist_dir'],
            collection_metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        print(f"向量数据库初始化完成，共加载 {len(all_chunks)} 个文本块")

# ====================== RAG处理链 ======================
    def _build_optimized_rag_chain(self):
    
        prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """你是专业问答助手，请严格遵循以下规则：
     1. 必须参考之前的对话历史
     2. 回答需与历史对话保持逻辑连贯
     3. 当前对话历史：{chat_history}
     ---
     问题：{input}
     上下文：{context}"""),
        
        MessagesPlaceholder(variable_name="chat_history"),
        
        ("human", 
         """请分析以下问题并提供专业解答：
[问题] {input}"""),
        
        ("human", 
         """可参考的上下文资料：
{context}""")
    ])
    
        return (
             {
        "context": self.retriever, 
        "input": RunnablePassthrough(),
        "chat_history": lambda _: self.memory.load_memory_variables({})["chat_history"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    # ====================== 优化后的代理构建 ======================
    def google_search(self,input: str) -> str:
        """使用Google搜索获取最新信息"""
        searcher = BaiduSearcher(headless=True)  # 生产环境使用无头模式
        #searcher =GoogleSearcher(headless=False)
        results = searcher.search(input, num_results=3)
        return self.llm.invoke(results).content
    
    def _build_agent(self):
        """优化版智能代理：集成多工具和增强提示"""
        
        # 强化版工具集
        tools = [
            Tool(
                name="知识库检索",
                func=lambda query:self.rag_chain.invoke(query),
                description="用于查询专业领域知识、内部文档内容"
            ),
            Tool(
                name="网络搜索",
                func=self.google_search,
                description="用于获取实时信息或补充背景知识"
            )
        ]

        from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS, PREFIX
        system_message = (
        "你是一个聪明、专业且细致的中文智能问答助手，能够根据用户的问题合理选择工具。\n\n"
        "你可以访问以下工具：\n"
        "1. 知识库检索：用于查询专业资料、公司文档、内部知识。\n"
        "2. 网络搜索：用于实时信息、新闻、广泛背景资料，或知识库无法回答的问题。\n\n"
        "在作答时，请根据问题内容判断是否需要使用工具；"
        "如果需要使用工具，请严格按照格式调用工具，并等待结果后再继续回答。\n"
        "如果用户提问内容明确可直接回答，可以直接作答。\n"
        "你应使用自然、简洁且专业的中文回答。\n"
    )
       
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            max_iterations=3,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={
            "system_message": system_message,
            "format_instructions": FORMAT_INSTRUCTIONS,
            "prefix": PREFIX,
        }
        )
    # --------------------- 核心查询接口 ---------------------
    def query(self, query: str) -> str:
        try:
            # 执行查询
            response = self.agent.run({"input":query})
            return response
        except Exception as e:
            return f"处理失败: {str(e)}"
        
    # --------------------- 知识库管理 ---------------------
    def add_documents(self, new_pdf_paths: List[str]):
        """增量添加新文档到知识库"""
        print("开始增量更新知识库...")
        new_chunks = []
        
        for pdf_path in new_pdf_paths:
            # 检查文档是否已存在
            existing = self.vector_store.get(
                where={"source": pdf_path},
                limit=1
            )
            if existing['ids']:
                print(f"检测到 {pdf_path} 已存在，将执行覆盖更新")
                self.remove_document(pdf_path)  # 先删除旧版本
                
            try:
                # 加载并处理新文档
                docs = self.pdf_loader.load(pdf_path)
                chunks = self.text_splitter.split_documents(docs)
                # 添加元数据
                for chunk in chunks:
                    chunk.metadata.update({
                        'source': pdf_path,
                        'filename': os.path.basename(pdf_path),
                    })
                new_chunks.extend(chunks)
            except Exception as e:
                print(f"处理 {pdf_path} 时出错: {str(e)}")
                continue
        
        if new_chunks:
            # 增量添加到向量数据库
            self.vector_store.add_documents(new_chunks)
            print(f"成功添加 {len(new_chunks)} 个新文本块")
            
        else:
            print("未添加新内容")

    def remove_document(self, pdf_path: str) -> bool:
        """从知识库删除指定文档"""
        try:
            # 获取规范化路径（确保与存储格式一致）
            normalized_path = os.path.normpath(pdf_path)
            
            # 先查询匹配的文档ID
            existing = self.vector_store.get(
                where={"source": normalized_path},
                limit=None  # 获取所有匹配项
            )
            
            if existing['ids']:
                # 通过ID删除更可靠
                self.vector_store.delete(ids=existing['ids'])
                print(f"已成功删除文档：{os.path.basename(pdf_path)} (共{len(existing['ids'])}个块)")
                return True
            else:
                print(f"未找到匹配的文档：{pdf_path}")
                return False
        except Exception as e:
            print(f"删除失败: {str(e)}")
            return False

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 系统配置
    directory = "NULL知识库"  # PDF文档目录
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    config = {
        'pdf_paths': [os.path.join(directory, f) for f in pdf_files],
        'persist_dir': "./NULL_chroma_db",  # 向量数据库存储路径
    }
    
    # 初始化RAG系统
    system = RAGSystem(config)
    
    # 自动打开浏览器
    import webbrowser
    webbrowser.open('http://localhost:5000')
    run_flask()