# RawRAG 项目开发 Spec

**状态：** Proposed baseline  
**版本：** 0.1  
**日期：** 2026-08-29

## 1. 背景与目标

RawRAG 面向企业内部使用，提供可私有化部署的知识库问答能力。第一阶段目标不是一次性覆盖所有文档类型，而是建立一条可测量、可替换、可扩展的 RAG 基线：Markdown 导入 → 切分 → 索引 → BM25/向量混合检索 → RRF → LLM 生成 → RAGAS 评测。

## 2. 范围

### 2.1 本阶段必须完成

1. Markdown 文件或 Markdown 文本导入，保留来源、标题层级、路径、更新时间和文档 ID。
2. 可配置 chunking：按 Markdown 标题/段落组织边界，并限制 chunk size 与 overlap。
3. 同一批数据建立 BM25 索引和向量索引。
4. 分别执行 BM25 与向量召回，使用 Reciprocal Rank Fusion（RRF）合并排序。
5. 暂不引入 reranker，但定义清晰的 reranker 扩展点。
6. 统一生成接口：外部 API 与本地 Ollama 可切换。
7. 使用公开评测集先跑通 RAGAS；形成固定数据版本、配置、指标和结果记录。
8. 以 API 方式暴露知识库导入、索引、检索、问答和评测入口，便于企业内部系统集成。

### 2.2 明确不在本阶段

- PDF/Word/Excel/网页等解析；后续以 MinerU 作为解析器候选。
- 图片、表格截图、音视频等多模态检索与生成。
- reranker、复杂 agent、多轮记忆、知识图谱和自动化数据采集。
- 以某一家云厂商 API 作为不可替换的核心实现。

## 3. 目标架构

```text
Markdown Source
    ↓
Ingestion / Metadata
    ↓
Markdown-aware Chunker
    ↓
 ┌───────────────┬────────────────┐
 │ BM25 Index    │ Vector Index   │
 └──────┬────────┴───────┬────────┘
        ↓                ↓
      Candidate Retrieval
                ↓
            RRF Fusion
                ↓
          Context Assembly
                ↓
     LLM Provider (API/Ollama)
                ↓
       Answer + citations + trace
                ↓
             RAGAS
```

建议的代码边界：`ingestion`、`chunking`、`indexing`、`retrieval`、`generation`、`evaluation`、`api`。索引、embedding、LLM、存储都必须通过 provider/adapter 隔离，避免把具体厂商 SDK 写入业务流程。

## 4. 关键接口与数据契约

### 4.1 文档与 chunk

```text
Document:
  id, source_uri, title, content, format="markdown", version, updated_at, metadata

Chunk:
  id, document_id, text, heading_path, chunk_index, token_count, metadata
```

`metadata` 至少保留 `tenant_id`、`source_uri`、`heading_path`、`content_type`，为未来权限过滤和多模态 `modality`/`asset_refs` 留出兼容空间。

### 4.2 检索

统一返回：`chunk_id`、文本、来源、单路 rank/score、融合分数、检索器名称。RRF 默认公式为：

`RRF(d) = Σ_r 1 / (k + rank_r(d))`

其中 `k`、每路 candidate 数和最终 top-k 都必须配置化并写入评测记录。重复 chunk 按稳定 ID 去重。

### 4.3 生成

`LLMProvider` 至少支持 `generate(messages, model, temperature, max_tokens)`；`EmbeddingProvider` 支持批量文本向量化。实现：

- 当前默认：阿里百炼 `qwen3.7-text-embedding`（以实际可用的模型标识和 API 文档为准）。
- 目标替换：BGE-M3，可本地部署或接入兼容服务。
- LLM：OpenAI-compatible/其他外部 API，以及 Ollama 本地 HTTP API。

## 5. 配置与部署

所有密钥仅来自环境变量。配置至少包含：embedding provider/model、LLM provider/model/base URL、索引路径、chunk 参数、BM25/vector candidate 数、RRF k、top-k、租户/权限策略和日志级别。开发环境可使用本地文件索引；企业部署需可替换为持久化数据库/向量库，并支持备份与重建。

## 6. 评测方案

先选择一个许可清晰、适合开放域或知识库问答的公开评测集，记录名称、版本、下载地址、许可、样本数和转换脚本。评测集适配为：问题、参考答案、参考上下文、知识库文档；不得把测试答案混入索引。

基线至少比较：

1. BM25-only
2. Vector-only
3. BM25 + Vector + RRF

RAGAS 初始关注 `faithfulness`、`answer_relevancy`、`context_precision`、`context_recall`，同时记录延迟、召回数量、token 消耗和失败率。评测配置与结果应可复跑，模型/API 版本和时间必须记录。RAGAS 分数用于比较方案，不作为生产质量的唯一判定依据。

## 7. API 草案

- `POST /v1/documents`：导入 Markdown。
- `POST /v1/indexes/rebuild`：重建指定知识库索引。
- `POST /v1/retrieve`：返回可解释的混合检索结果。
- `POST /v1/chat`：返回答案、引用 chunk、模型和 trace id。
- `POST /v1/evaluations`：执行指定评测配置。
- `GET /healthz`、`GET /readyz`：服务与依赖健康检查。

后续必须补充认证、租户隔离、文档权限过滤、审计日志、限流和数据删除语义，才能进入企业生产使用。

## 8. 里程碑与验收

### M0：基线与仓库治理

完成分支策略、配置模板、Markdown fixture、接口契约和可复现运行说明。

### M1：单库检索基线

导入一组 Markdown，BM25/vector/RRF 均可运行，结果带来源和 rank 分数，具备单元测试。

### M2：生成与 provider

完成外部 API 与 Ollama 至少一个真实 provider，支持超时、重试、错误归一化和引用输出。

### M3：公开集评测

固定公开评测集和适配脚本，输出三组检索基线与 RAGAS 报告，结果可重跑。

### M4：企业化前置能力

补齐认证、租户/权限、持久化、审计、删除和部署文档；通过 PR 合并到 `main` 后再标记首个可用版本。

## 9. Git 工作流

- `main`：可发布基线，只通过 PR 合并。
- `develop`：日常集成分支。
- `feature/<topic>`：从 `develop` 创建，完成测试和评测后合并 `develop`。
- 发布流程：`develop` → PR → `main`；PR 必须附测试结果、评测变化和配置变化说明。
- 原历史 `main` 内容保存在归档分支；新的 `main` 使用独立根提交，不继承旧提交。

## 10. 后续演进约束

MinerU 接入应实现为 ingestion/parser adapter，不改变 chunk、metadata、index 和 retrieval 契约。多模态 RAG 通过 `modality`、资产引用和多路 embedding 扩展，不破坏 Markdown-only 的文本路径。reranker 作为 RRF 之后、context assembly 之前的可选 stage，默认关闭。
