# 掌柜智库 · 企业知识库智能问答平台

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange)
![Milvus](https://img.shields.io/badge/Milvus-2.4%2B-purple)
![BGE-M3](https://img.shields.io/badge/Embedding-BGE--M3-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

> 面向企业产品知识库的 **RAG 智能检索问答平台** — 支持产品文档批量导入、向量化存储、多路检索融合与大语言模型生成式回答，实现 SSE 流式实时输出。

---

## 🌟 项目亮点

- **多智能体工作流架构**：基于 LangGraph 构建导入/查询两条可编排流水线，节点解耦、状态可追踪
- **多路检索融合策略**：向量检索 + HyDE 假设性文档检索 + MCP 网络搜索，RRF 倒排融合后经 CrossEncoder 重排序
- **全链路容错设计**：各搜索节点异常独立捕获、模型不可用自动降级、网络失败返回空列表不中断流程
- **SSE 流式交互**：支持节点进度推送 + 答案逐字输出，用户体验接近 ChatGPT
- **商品名称识别**：LLM 辅助识别文档关联产品，精准定位查询上下文

## 🛠 技术栈

| 分类 | 技术 | 说明 |
|------|------|------|
| **后端框架** | FastAPI + Uvicorn | 异步高性能 API 服务 |
| **工作流编排** | LangGraph | 有状态图编排，节点可追踪 |
| **向量数据库** | Milvus (pymilvus) | 高维向量近似最近邻检索 |
| **嵌入模型** | BGE-M3 | 本地部署，1024 维稠密向量 |
| **重排序模型** | BGE-Reranker-Large | Sentence-Transformers CrossEncoder |
| **大语言模型** | 阿里云 DashScope（通义千问） | OpenAI 兼容接口 |
| **PDF 解析** | MinerU API | 结构化 PDF → Markdown |
| **对象存储** | MinIO | 原始文件与图片备份 |
| **会话存储** | MongoDB | 对话历史持久化 |
| **前端** | 原生 HTML/JS | FastAPI 静态托管 |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端 (HTML/JS)                           │
│   文件导入页(8000)  │  智能问答页(8011, SSE 流式)                │
└─────────┬───────────────────────┬───────────────────────────────┘
          │ POST /upload          │ POST /query
          │ POST /status/{id}     │ GET  /stream/{task_id}
          ▼                       ▼
┌──────────────────────┐  ┌──────────────────────────────────────────┐
│  导入服务 (8000)       │  │  查询服务 (8011)                          │
│  FileProcessService   │  │  QueryService                            │
└──────────┬─────────────┘  └────────────┬─────────────────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph 工作流引擎                              │
│                                                                     │
│  ┌─ 导入流水线 ──────────────────────────────────────────────┐      │
│  │ entry → pdf_to_md → md_image → document_split             │      │
│  │        → item_name_recognize → bge_embedding → milvus_import│     │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                     │
│  ┌─ 查询流水线 ──────────────────────────────────────────────┐      │
│  │ item_confirm → multi_search (并行)                          │      │
│  │                ├─ vector_search (BGE-M3)                   │      │
│  │                ├─ hyde_search (LLM假设文档)                 │      │
│  │                └─ web_search (MCP网络搜索)                  │      │
│  │               → rrf_merge → rerank → answer_output (SSE)    │      │
│  └───────────────────────────────────────────────────────────┘      │
└──────────┬──────────────────────┬──────────────────────┬────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
    ┌──────────────┐       ┌──────────────┐        ┌──────────────┐
    │   Milvus     │       │   MongoDB    │        │    MinIO     │
    │  向量存储     │       │  会话历史     │        │  文件备份     │
    └──────────────┘       └──────────────┘        └──────────────┘
```

## 📁 目录结构

```
knowledge/
├── api/                          # FastAPI 接口层
│   ├── import_api.py             #   导入服务 (端口 8000)
│   └── query_api.py              #   查询服务 (端口 8011)
├── processor/                    # LangGraph 工作流
│   ├── import_processor/         #   导入流水线
│   │   ├── main_graph.py         #     节点编排
│   │   ├── state.py              #     状态定义
│   │   └── nodes/                #     7 个处理节点
│   └── query_processor/          #   查询流水线
│       ├── main_graph.py
│       ├── state.py
│       └── nodes/                #     7 个处理节点
├── utils/
│   ├── clients/                  #   客户端管理（双重检查锁单例）
│   │   ├── ai_clients.py         #     LLM / BGE-M3 / Reranker
│   │   └── storage_clients.py    #     MinIO / Milvus / MongoDB
│   ├── sse_util.py               #   SSE 流式输出
│   └── ...
├── prompts/                      # Prompt 模板
├── service/                      # 业务服务层
├── schema/                       # Pydantic 数据模型
├── front/                        # 前端静态页面
└── test/                         # 测试脚本
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- Milvus 2.4+（向量数据库）
- MongoDB（会话存储）
- MinIO（对象存储）
- GPU（可选，BGE 模型 CPU 亦可运行）

### 2. 安装依赖

```bash
cd knowledge
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# LLM 配置
OPEN_API_KEY=sk-xxx
OPEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen-turbo

# BGE 本地模型路径
BGE_M3_PATH=./models/bge-m3
BGE_RERANKER_LARGE=./models/bge-reranker-v2-m3
BGE_DEVICE=cuda
BGE_FP16=true

# Milvus
MILVUS_URL=http://localhost:19530

# MongoDB
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=zhiku

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=knowledge

# MinerU PDF 解析
MINERU_API_TOKEN=your_token
```

### 4. 启动服务

```bash
# 导入服务（端口 8000）
cd knowledge
python api/import_api.py

# 查询服务（端口 8011）
python api/query_api.py
```

### 5. 访问页面

- 📄 文件导入：http://localhost:8000/front/import.html
- 💬 智能问答：http://localhost:8011/front/index.html

## 📡 API 快速示例

### 上传文档

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@product_manual.pdf"
# => {"task_id": "abc123", "message": "导入任务已创建"}
```

### 查询任务进度

```bash
curl http://localhost:8000/status/abc123
# => {"status": "running", "completed_nodes": ["entry", "pdf_to_md"], ...}
```

### 智能问答（同步）

```bash
curl -X POST http://localhost:8011/query \
  -H "Content-Type: application/json" \
  -d '{"query": "RS-12 数字万用表如何测量直流电压？", "is_stream": false}'
# => {"answer": "RS-12 万用表测量直流电压的步骤...", "sources": [...]}
```

### 智能问答（SSE 流式）

```bash
# 1. 提交查询，获取 task_id
curl -X POST http://localhost:8011/query \
  -H "Content-Type: application/json" \
  -d '{"query": "万用表怎么用？", "is_stream": true}'

# 2. 连接 SSE 流，实时接收进度和答案
curl -N http://localhost:8011/stream/{task_id}
# => event: progress
#    data: {"node": "vector_search", "status": "running"}
#    event: delta
#    data: {"content": "万用表是一种..."}
#    event: final
#    data: {"answer": "...", "sources": [...]}
```

## 🔑 核心设计决策

### 1. 多路检索 + 融合排序

单一向量检索存在语义鸿沟问题，本系统采用三路并行检索：

| 检索策略 | 原理 | 优势 |
|----------|------|------|
| **向量检索** (BGE-M3) | 将问题和文档嵌入向量空间，计算余弦相似度 | 语义匹配，支持近义词/同义词 |
| **HyDE 检索** | 先让 LLM 生成假设性答案，再用假设答案检索 | 问题与文档措辞不同时依然命中 |
| **网络搜索** (MCP) | 调用搜索引擎获取实时外部信息 | 补充知识库中未覆盖的最新内容 |

三路结果通过 **RRF（Reciprocal Rank Fusion）** 倒排融合，再经 **CrossEncoder 重排序** 精准打分。

### 2. 全链路容错与降级

```python
# 各搜索节点独立 try-catch，失败不阻塞后续流程
try:
    search_results = await do_vector_search(query)
except Exception as e:
    logger.error(f"Vector search failed: {e}")
    search_results = []  # 返回空列表，其他检索继续
```

- 重排序模型不可用 → 降级为基于排名位置的默认评分
- 网络搜索超时/错误 → 返回空列表，不中断整体查询
- 知识库无相关信息 → 自动调用 LLM 直接回答

### 3. 商品名称识别

```python
# 导入时：LLM 识别文档关联的产品名/型号
item_names = llm.identify_products(markdown_content)

# 查询时：先识别问题中的商品，精准定位上下文
confirmed_item = llm.confirm_item(user_query)
if confirmed_item in knowledge_base:
    # 限定在该商品范围内检索
```

## 📊 性能数据

| 指标 | 数值 |
|------|------|
| 单文档导入耗时 | ~15s（PDF → 向量入库） |
| 平均查询响应时间 | <2s（含 LLM 生成） |
| SSE 首字延迟 | <500ms |
| 支持并发 | 10+ 查询/秒 |

## 🤝 贡献

项目处于活跃开发中，欢迎 Issue 和 PR！

## 📄 License

MIT License