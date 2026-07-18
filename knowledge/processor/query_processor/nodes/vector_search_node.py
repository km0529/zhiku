from typing import Tuple, List

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError, MilvusError, EmbeddingError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.clients.ai_clients import AIClients
from knowledge.utils.clients.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import _item_names_filter, create_hybrid_search_requests, execute_hybrid_search_query

"""
最终经过混合检索后,state中多了一个embedding_chunks属性,它的值是:
[{
    "id":""
    "distance":分数,
    "entity":{"item_name":"","title":"","content":""}
}]
"""
class VectorSearchNode(BaseNode):
    name = "vector_search_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        rewritten_query,item_names = self._validate_state(state)
        
        if not item_names:
            self.logger.info("item_names为空，跳过向量检索")
            return {"embedding_chunks": []}
            
        try:
            embedding_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"获取嵌入模型对象失败,{e}")
            return {"embedding_chunks": []}

        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取milvus客户端对象失败,{e}")
            return {"embedding_chunks": []}

        try:
            embedding_result = generate_bge_m3_hybrid_vectors(embedding_client,[rewritten_query])
            expr, expr_params = _item_names_filter(item_names)
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=embedding_result.get("dense")[0],
                sparse_vector=embedding_result.get("sparse")[0],
                expr=expr,
                expr_params=expr_params,
                limit=self.config.embedding_search_limit
            )
            hybrid_search_results = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_requests,
                limit=self.config.embedding_search_limit,
                output_fields=["item_name","title","content"]
            )
            return {"embedding_chunks":hybrid_search_results[0]}
        except Exception as e:
            self.logger.error(f"向量搜索节点执行检索失败: {e}")
            return {"embedding_chunks": []}

    def _validate_state(self, state:QueryGraphState) -> Tuple[str, List[str]]:
        #1. 获取rewritten_query,item_names
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        #2. 校验rewritten_query,item_names
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error(f"rewritten_query不能为空以及类型必须是str")
            raise StateFieldError(node_name=self.name,field_name="rewritten_query",expected_type=str)

        if not isinstance(item_names, list):
            self.logger.error("item_names类型必须是list")
            raise StateFieldError(node_name=self.name,field_name="item_names",expected_type=list)

        return rewritten_query,item_names or []

if __name__ == '__main__':
    #1. 创建state
    state = {
        "rewritten_query":"RS PRO RS-12 数字万用表的使用方法是什么?",
        "item_names":["RS PRO RS-12 数字万用表"]
    }
    node = VectorSearchNode()
    final_state = node.process(state)
    print(final_state)