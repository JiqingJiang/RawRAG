# RawRAG Agent Guide

## 项目目标

RawRAG 是企业内部自用的私有知识库与 RAG 服务。当前阶段只接收 Markdown，先验证混合检索与评测闭环；后续再接入 MinerU 文档解析和多模态 RAG。

## 当前技术决策

- 输入：Markdown-only；暂不做 PDF、Office、网页等文档解析。
- 检索：BM25 + 向量检索，融合排序使用 RRF。
- 暂不使用 reranker；保留后续扩展接口。
- Embedding：开发验证阶段使用阿里百炼 `qwen3.7-text-embedding`；目标模型为 BGE-M3。
- 生成：支持外部 API，也支持本地 Ollama；通过统一 LLM provider 接口隔离实现。
- 评测：RAGAS；先选用公开评测集建立可复现基线。
- 暂不做多模态；数据模型需为未来图文/表格/版面信息保留扩展位。

## 开发约定

- 主干分支：`main` 只接收已验证、可发布的变更。
- 开发分支：`develop`；功能分支从 `develop` 创建，完成验证后提 PR 合并到 `main`。
- 所有架构、行为或评测口径变化必须同步更新 `docs/DEVELOPMENT_SPEC.md`。
- 代码按 ingest/chunk/index/retrieve/generate/evaluate 分层，provider 通过接口注入。
- 配置从环境变量或配置文件读取，禁止提交密钥、真实知识库数据和评测私有数据。
- 新功能至少补充单元测试；检索或生成行为变化必须运行离线评测并记录指标。

## 验证最低要求

提交前运行项目已有测试/脚本；新增模块需覆盖正常路径、空输入、配置错误和 provider 异常。涉及检索时至少报告 BM25、向量、RRF 三组可比较结果，并固定数据切分、top-k、随机种子和模型版本。

## 上下文加载

先读本文件，再读 `docs/DEVELOPMENT_SPEC.md` 及与当前任务相关的文档。当前运行环境无法创建受平台保护的 `.agents/` 目录，因此暂不建立该目录下的 skills/notes 结构；后续在可写环境中补齐。
