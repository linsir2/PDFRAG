# 私人PDF问答机器人
"""
langchain相应包的作用：
langchain:核心库，包含Chain,Runnable等组装工具，类似于nn.Sequential
langchain-openai:用来连接大模型API
langchain-community:社区扩展，包含Redis、Mongo等储存历史的东西
langchain-text-splitters:切文本的工具(PDF太长得切成块)
langchain-chroma + chromadb:向量数据库，存PDF内容的“智能索引”
pypdf:读PDF的工具
tiktoken:计算token数的工具
"""
from pathlib import Path
from pydantic import SecretStr
from langchain_openai import ChatOpenAI # chatopenai:调用大模型聊天；openaiembeddings:把文本转成向量（数学表示，便于搜索）
from langchain_community.document_loaders import PyPDFDirectoryLoader # PyPDFDirectoryLoader:自动加载文件夹中的PDF文件
from langchain_text_splitters import RecursiveCharacterTextSplitter # RecursiveCharacterTextSplitter:把长文本切成小块(太长模型会吃不下)
from langchain_chroma import Chroma # Chroma:向量数据库，存放切好的文本块+向量
from langchain_core.prompts import ChatPromptTemplate # ChatPromptTemplate:写Prompt的模板(让模型按照你的格式回答)
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda # RunnablePassthrough:"原封不动传出"输入(链里用);RunnableMap:同时跑多个东西(并行处理)
from langchain_core.output_parsers import StrOutputParser # StrOutputParser:把模型输出转化为纯字符串
from langchain_core.runnables.history import RunnableWithMessageHistory # RunnableWithMessageHistory:带记忆的链(记住聊天历史)
from langchain_core.chat_history import InMemoryChatMessageHistory #InMemoryChatMessageHistory:内存存聊天记录
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatOpenAI(
    model = "deepseek-chat",
    api_key = SecretStr("sk-****************************"),
    base_url = "https://api.deepseek.com/v1",
    temperature = 0.3    # 温度低：回答更稳定
)

embeddings = HuggingFaceEmbeddings(
    #model_name="sentence-transformers/all-MiniLM-L6-v2",  # 英文推荐
    model_name="BAAI/bge-small-zh-v1.5",  # ← 中文推荐用这个
    model_kwargs={'device': 'cpu'},  # 指定CPU运行（没有GPU的话）
)
"""
Document是一个结构化对象：核心文本内容+元数据(来源、页码、时间等)
doc = Document(
    page_content="LangChain 是一个用于开发 LLM 应用的框架。",
    metadata={
        "source": "langchain_guide.pdf",
        "page": 5,
        "author": "Harrison Chase",
        "date": "2023-10-01"
    }
)
"""
current_dir = Path(__file__).parent
print(f"脚本所在目录{current_dir}")
docs_dir = current_dir / "docs"
print("正在加载你的PDF文件...")
loader = PyPDFDirectoryLoader(str(docs_dir))
docs = loader.load()    #load方法:把PDF转成Document对象（一段文本内容+其附加信息）
if not docs:
    print("Loading failed")
    exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, # 每块1000字符
    chunk_overlap = 200 # 块间重叠200字符（防止断句）
)
chunks = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents( # from_documents():把切块+向量存进数据库
    documents = chunks,
    embedding = embeddings,
    persist_directory = "./my_db" # 保存到本地，下次直接用
)
retriever = vectorstore.as_retriever(search_kwargs = {"k": 5}) # as_retriever():转成检索器(能搜索相似内容)

# prompt模板       其中的{context}{question}都是占位符，本身没有值，等待后续通过PromptTemplate填充
template = """
你是一个PDF专家，根据一下内容回答问题。如果不知道，就说“我不知道”。
相关内容：
{context}

问题：{question}
回答：
"""
prompt = ChatPromptTemplate.from_template(template) #from_template():从字符串转换为Prompt对象

# 组装RAG链 + 带记忆
"""
用户问题 
→ 检索器（找资料） 
→ Prompt模板（写任务书） 
→ 大模型（思考） 
→ 解析器（整理答案） 
→ 给你
"""
rag_chain = (
    RunnableParallel({ # RunnableMap:并行跑两个东西
        "context": RunnableLambda(lambda x: retriever.invoke(x["question"])), # 检索PDF内容
        "question": RunnablePassthrough() # 透传用户问题
    })
    | prompt # | 是链式连接
    | llm
    | StrOutputParser() # 转为字符串
)

# 加上记忆功能(用RunnableWithMessageHistory包裹整个链)
store = {}

def get_session_history(session_id: str):
    if session_id  not in store:
        store[session_id] = InMemoryChatMessageHistory() # 每个用户一个历史
    return store[session_id]

# rag_chain本身就可以invoke了，但要有记忆功能就还的给rag_chain套上一层runnablewithmessagehistory方法
chain_with_history = RunnableWithMessageHistory(
    runnable = rag_chain,
    get_session_history = get_session_history,
    input_messages_key = "question",
    history_messages_key = "history"
)

print("\n你的 PDF 机器人已就绪！问任何问题吧（输入 'quit' 退出）\n")

while True:
    inputs = input("you:")
    if inputs.strip().lower() == "quit":
        break

    response = chain_with_history.invoke(
        {"question": inputs},
        config = {"configurable": {"session_id": "user_001"}}
    )

    print(f"AI: {response}\n")