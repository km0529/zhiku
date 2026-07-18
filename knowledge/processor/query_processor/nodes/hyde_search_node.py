from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.exceptions import StateFieldError, LLMError, EmbeddingError, MilvusError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompts.query_prompt import HYDE_SYSTEM_PROMPT_TEMPLATE, HYDE_USER_PROMPT_TEMPLATE
from knowledge.test.query.milvus.test_hybrid_search import generate_hybrid_embeddings
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.milvus_util import _item_names_filter, create_hybrid_search_requests, execute_hybrid_search_query


class HydeSearchNode(BaseNode):
    name = "hyde_search_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        rewritten_query,item_names = self._validate_state(state)
        
        if not item_names:
            self.logger.info("item_names为空，跳过HyDE检索")
            return {"hyde_embedding_chunks": []}
            
        try:
            hy_document = self._generate_hy_document(rewritten_query,item_names)
        except Exception as e:
            self.logger.error(f"HyDE搜索节点生成假设性文档失败: {e}")
            return {"hyde_embedding_chunks": []}

        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"获取嵌入模型对象失败,{e}")
            return {"hyde_embedding_chunks": []}

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取milvus客户端对象失败,{e}")
            return {"hyde_embedding_chunks": []}

        try:
            embedding_input = f"{rewritten_query}\n{hy_document}"
            embedding_result = generate_hybrid_embeddings(embedding_client, [embedding_input])
            expr, expr_params = _item_names_filter(item_names)
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=embedding_result.get("dense")[0],
                sparse_vector=embedding_result.get("sparse")[0],
                expr=expr,
                expr_params=expr_params,
                limit=self.config.hyde_search_limit
            )
            hyde_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_requests,
                limit=self.config.hyde_search_limit,
                output_fields=["item_name","title","content"]
            )
            return {"hyde_embedding_chunks":hyde_search_result[0]}
        except Exception as e:
            self.logger.error(f"HyDE搜索节点执行检索失败: {e}")
            return {"hyde_embedding_chunks": []}

    def _validate_state(self, state:QueryGraphState):
        # 1. 获取rewritten_query,item_names
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        # 2. 校验rewritten_query,item_names
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"rewritten_query不能为空以及类型必须是str")
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not isinstance(item_names, list):
            self.logger.error("item_names类型必须是list")
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names or []

    # 调用大模型生成假设性文档
    def _generate_hy_document(self, rewritten_query:str, item_names:List[str]):
        #1. 获取llm客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except Exception as e:
            self.logger.error(f"创建大模型客户端失败,{e}")
            raise LLMError(node_name=self.name,message=f"创建大模型客户端失败,{e}")

        #2. 构建SystemMessage和HumanMessage
        hyde_system_prompt = HYDE_SYSTEM_PROMPT_TEMPLATE.format(item_names=item_names)
        hyde_user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names=item_names,
            rewritten_query=rewritten_query
        )

        system_message = SystemMessage(content=hyde_system_prompt)
        human_message = HumanMessage(content=hyde_user_prompt)

        #3. 调用大模型
        try:
            llm_result = llm_client.invoke([system_message,human_message])
        except Exception as e:
            self.logger.error(f"调用大模型客户端失败,{e}")
            raise LLMError(node_name=self.name, message=f"调用大模型客户端失败,{e}")


        return llm_result.content

if __name__ == '__main__':
    # 1. 创建state
    state = {
        "rewritten_query": "RS PRO RS-12 数字万用表的使用方法",
        "item_names": ["RS PRO RS-12 数字万用表"]
    }
    node = HydeSearchNode()
    final_state = node.process(state)
    print(final_state)