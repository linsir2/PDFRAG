import os
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

api_key = os.environ["DEEPSEEK_API_KEY"] = "sk-************************"
if not api_key:
    raise ValueError("DEEPSEEK_API_KEY环境变量未设置")

llm = ChatOpenAI(
    model = "deepseek-chat",
    api_key = SecretStr(api_key),
    base_url = "https://api.deepseek.com/v1",
    temperature = 0.8
)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 组装链(langchain最核心的玩法)
chain = RunnableWithMessageHistory(
    runnable = llm,
    get_session_history = get_session_history,
    # chatopenai 不支持传入字典
    #input_messages_key = "input",
    #history_messages_key = "history",
)

print("AI is ready!Enter [quit] to end the chat.\n")
while True:
    user_input = input("You:")
    if user_input.strip().lower() in ["quit"]: break

    response = chain.invoke(
        user_input,
        config = {"configurable": {"session_id": "user_001"}}
    )

    print(f"AI: {response.content}")
