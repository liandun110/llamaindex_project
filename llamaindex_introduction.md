# Llama Index 项目介绍
- llama_index 项目地址：https://github.com/run-llama/llama_index/
- 截至2025年12月3日，该项目获得star数为45.6k。
- 项目文档地址：https://developers.llamaindex.ai/python/framework/

# 安装
```bash
pip install llama_index -i https://mirrors.aliyun.com/pypi/simple/
```

# 调用 ollama 实现 agent 和 tools

```bash
pip install llama-index-llms-ollama -i https://mirrors.aliyun.com/pypi/simple/
```

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
    llm=Ollama(
        model="Qwen3:8b",
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    ),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
```

# 调用 Ollama：方法2

```python
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
llm=Ollama(
        model="Qwen3:8b",
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    )
Settings.llm = llm
```

# 调用 ollama 利用 FunctionAgent 实现股价查询

```bash
pip install llama-index-tools-yahoo-finance -i https://mirrors.aliyun.com/pypi/simple/
```

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

llm=Ollama(
        model="Qwen3:8b",
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    )
Settings.llm = llm

finance_tools = YahooFinanceToolSpec().to_tool_list()

workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    tools=finance_tools,
    system_prompt="You are a helpful assistant.",
)

async def main():
    response = await workflow.run(
        user_msg="What's the current stock price of NVIDIA?"
    )
    print(response)

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
```

# 调用维基百科工具

```bash
pip install llama-index-tools-wikipedia -i https://mirrors.aliyun.com/pypi/simple/
```

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.tools.wikipedia import WikipediaToolSpec

llm=Ollama(
        model="Qwen3:8b",
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    )
Settings.llm = llm

tool_spec = WikipediaToolSpec()

workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    tools=tool_spec.to_tool_list(),
    system_prompt="You are a helpful assistant.",
)

async def main():
    response = await workflow.run(
        user_msg="Who is Ben Afflecks spouse? 用中文回复"
    )
    print(response)

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
```

# 智能体调用 tavily 网络搜索工具

```bash
pip install llama-index-tools-tavily-research -i https://mirrors.aliyun.com/pypi/simple/
```

```python
import os
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.tools.tavily_research import TavilyToolSpec

os.environ["TAVILY_API_KEY"] = "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" # 替换你的 Tavily Key

llm=Ollama(
        model="Qwen3:8b",
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    )
Settings.llm = llm

tool_spec = TavilyToolSpec(
    api_key=os.environ["TAVILY_API_KEY"],
)

workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    tools=tool_spec.to_tool_list(),
    system_prompt="You are a helpful assistant.",
)

async def main():
    response = await workflow.run(
        user_msg="Who is Ben Afflecks spouse? 用中文回复"
    )
    print(response)

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
```

# 基于 fastapi 后端，调用智能体

```bash
pip install llama-index-llms-ollama -i https://mirrors.aliyun.com/pypi/simple/
```

```python
import os
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.tools.tavily_research import TavilyToolSpec
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

os.environ["TAVILY_API_KEY"] = "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" # 替换你的 Tavily Key

llm=Ollama(
        model="Qwen3:8b",
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    )
Settings.llm = llm

tool_spec = TavilyToolSpec(
    api_key=os.environ["TAVILY_API_KEY"],
)

workflow = FunctionAgent(
    name="Agent",
    description="Useful for performing financial operations.",
    tools=tool_spec.to_tool_list(),
    system_prompt="You are a helpful assistant.",
)

@app.post("/api/ask")
async def main():
    print('开始查询')
    response = await workflow.run(
        user_msg="Who is Ben Afflecks spouse? 用中文回复"
    )
    print(response)
    return response
```

运行方式：`python -m uvicorn main:app --reload --port 8001`

调用方式：`curl -X POST http://localhost:8001/api/ask`

# 基于 RAG 的问答

注意，下面的方法默认使用 OpenAI。

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```

# 基于 RAG 的问答 —— 设置不同嵌入提取方法：基于HuggingFace

安装：`pip install llama-index-embeddings-huggingface -i https://mirrors.aliyun.com/pypi/simple/`

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```

# 基于 RAG 的问答 —— 基于 DashScope 在线 API
```bash
pip install llama-index-embeddings-dashscope -i https://mirrors.aliyun.com/pypi/simple/
```

```python
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.core import Settings

Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key="sk-efe1c9004f7e4de0a8ade26120301c6d"
)
```

# 嵌入向量的保存

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)

Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key="sk-efe1c9004f7e4de0a8ade26120301c6d"
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./full_index_storage")
```

# 嵌入向量的读取

```python
from llama_index.core import load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)

llm = Ollama(
        model="Qwen3:8b",
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    )
Settings.llm = llm

Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key="sk-efe1c9004f7e4de0a8ade26120301c6d"
)

# 从本地目录加载索引
storage_context = StorageContext.from_defaults(persist_dir="./full_index_storage")
index = load_index_from_storage(storage_context)
print("知识库加载完成")
# 使用索引查询
query_engine = index.as_query_engine()
response = query_engine.query("输出《邓稼先》全文")
print(response)
```

效果：不如网页版好。比如我问：哪句话描述了邓稼先的一生。网页版答案准确：鞠躬尽瘁、死而后已。但是代码结果不对。

但是，改正如下实现后，回答是正确的：
```python
from llama_index.core import load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)

# 配置LLM
llm = Ollama(
        model="Qwen3:8b",
        request_timeout=360.0,
        context_window=8000,
    )
Settings.llm = llm

# 配置嵌入模型
Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key="sk-efe1c9004f7e4de0a8ade26120301c6d"
)

# 从本地目录加载索引
storage_context = StorageContext.from_defaults(persist_dir="./full_index_storage")
index = load_index_from_storage(storage_context)
print("知识库加载完成")

# 创建查询引擎，可以配置返回更多检索结果
query_engine = index.as_query_engine(
    similarity_top_k=5,  # 返回最相似的5个结果
    response_mode="tree_summarize"  # 使用树状总结模式
)

# 执行查询
response = query_engine.query("哪句话描述了邓稼先的一生")

# 打印最终回答
print("=== 最终回答 ===")
print(response)
print("\n" + "="*80)

# 打印检索结果详情
print("\n=== 检索结果详情 ===")
for i, node in enumerate(response.source_nodes, 1):
    print(f"\n【检索结果 {i}】")
    print(f"相似度得分: {node.score:.4f}")
    print(f"文本内容: {node.node.get_text()[:1000]}...")  # 显示前1000个字符
    if node.node.metadata:
        print(f"元数据: {node.node.metadata}")
    print("-" * 60)

# 如果想要获取所有检索到的文本内容
print("\n=== 所有检索到的完整文本 ===")
all_text = "\n\n".join([node.node.get_text() for node in response.source_nodes])
print(all_text)
```

# 流式输出

```python
from llama_index.core import load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
import sys

# 配置LLM
llm = Ollama(
    model="Qwen3:8b",
    request_timeout=360.0,
    context_window=8000,
)
Settings.llm = llm

# 配置嵌入模型
Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key="sk-efe1c9004f7e4de0a8ade26120301c6d"
)

# 从本地目录加载索引
storage_context = StorageContext.from_defaults(persist_dir="./full_index_storage")
index = load_index_from_storage(storage_context)
print("知识库加载完成")

# 创建普通查询引擎，但手动处理流式响应
query_engine = index.as_query_engine(
    similarity_top_k=5,
    streaming=True,  # 启用流式
    response_mode="tree_summarize"
)


def stream_response(query_text):
    """处理流式响应的函数"""
    response = query_engine.query(query_text)

    # 逐段输出
    print("\n回答：", end="", flush=True)
    full_response = ""

    for text in response.response_gen:
        print(text, end="", flush=True)
        sys.stdout.flush()
        full_response += text

    print()
    return full_response, response


# 主循环
while True:
    query_text = input("\n输入你的问题（输入'exit'退出）：")
    if query_text.lower() == 'exit':
        break

    try:
        full_response, response_obj = stream_response(query_text)

        # 可选：显示检索结果
        show_sources = input("\n是否查看检索结果？(y/n): ")
        if show_sources.lower() == 'y':
            print("\n=== 检索结果详情 ===")
            for i, node in enumerate(response_obj.source_nodes, 1):
                print(f"\n结果 {i} (相似度: {node.score:.4f}):")
                print(f"{node.node.get_text()[:500]}...")

    except Exception as e:
        print(f"\n发生错误: {e}")
```

# 实现联网搜索

意义：书中有：课外阅读《太阳吟》，可内容是什么？

```bash
pip install llama-index-readers-web
```