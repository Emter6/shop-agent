"""
Embedding 客户端管理器

负责按配置初始化 Embedding 服务客户端，并为字段、指标和用户问题的向量化
提供统一访问入口。当前使用 Ollama 本地 Embedding 服务。
"""

import asyncio
from typing import Optional

from langchain_ollama import OllamaEmbeddings

from app.conf.app_config import EmbeddingConfig, app_config


class EmbeddingClientManager:
    """管理 Embedding 服务客户端的初始化与复用"""

    def __init__(self, config: EmbeddingConfig):
        self.client: Optional[OllamaEmbeddings] = None
        self.config = config

    def init(self):
        """显式初始化客户端，避免模块导入时立即建立外部连接"""
        self.client = OllamaEmbeddings(
            model=self.config.model,
            base_url=self.config.base_url,
        )


# 模块级单例，供整个项目复用同一套 Embedding 客户端管理器
embedding_client_manager = EmbeddingClientManager(app_config.embedding)


if __name__ == "__main__":
    embedding_client_manager.init()
    client = embedding_client_manager.client

    async def test():
        """执行一次最小化向量化调用，验证服务是否可用"""
        text = "What is deep learning?"
        query_result = await client.aembed_query(text)
        print(query_result[:3])

    asyncio.run(test())
