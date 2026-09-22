# RiskLens · 金融智能风控分析平台

基于天池 [贷款违约预测赛题](https://tianchi.aliyun.com/competition/entrance/531830) 的本地演示项目。从原始贷款数据出发，串起数据质量、特征探索、模型实验、风险分层、阈值决策、客户画像、客群分析、特征漂移和有数据证据的 AI 问答。

页面中的客户数、违约率、AUC、阈值结果、风险分层与 PSI 均由本地比赛数据计算。当前留出验证 AUC 约 **0.7313**，这是本地参考值，不是天池线上分数。

> 本项目用于学习和功能演示，不应用于真实信贷审批或客户资格决策。

## 启动

要求 Python 3.12。仓库使用 FastAPI 提供 API 和本地页面，前端为原生 HTML、CSS、JavaScript，无需 Node 构建。依赖版本见 [`requirements.txt`](requirements.txt)。

在 PowerShell 中，从仓库根目录安装依赖：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

登录天池后从 [赛题与数据页](https://tianchi.aliyun.com/competition/entrance/531830/information) 下载 `train.csv`、`testA.csv`、`sample_submit.csv`，放到 `data/`。仓库不包含比赛数据，目录结构应为：

```text
data/
├── train.csv
├── testA.csv
└── sample_submit.csv
```

首次运行先训练模型并生成页面所需聚合数据：

```powershell
.\.venv\Scripts\python.exe baseline.py
.\.venv\Scripts\python.exe prepare_demo.py
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/>。首次训练会读取 80 万条训练记录，耗时取决于机器配置。以后只查看演示时，执行最后一条启动命令即可。生成的模型、验证集预测、提交文件和页面聚合数据都在 `outputs/`。

如需 DeepSeek 总结，在启动服务前配置进程环境变量 `DEEPSEEK_API_KEY`。默认模型为 `deepseek-flash`，可通过 `DEEPSEEK_MODEL` 改写。调用使用 [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)；密钥只在服务端读取，不写入项目文件。未配置或调用失败时，问答保留本地计算证据，并显示在线总结不可用。

## 首版功能

- **风险驾驶舱**：真实训练标签占比、留出验证 AUC、ROC、测试集风险分布。
- **数据中心**：字段质量与关键字段分布；CSV/Parquet 文件预览上限 25 MB，不保存上传文件。
- **Feature Lab**：五种业务特征的公式与 5 万条样本预览。
- **Model Lab**：完整基线指标；可运行 5,000–50,000 条样本的梯度提升树、逻辑回归、随机森林后台任务。小样本实验与完整基线不能直接公平比较。
- **阈值模拟**：在 16 万条留出验证集上计算召回、精确、漏判、误拒与假设成本。
- **客户画像**：测试集 A 客户查询，模型概率与单变量替换敏感度。敏感度并非 SHAP，也不表示因果关系。
- **客群分析**：训练集历史违约率分组；测试集预测风险筛选与 CSV 导出。
- **模型监控**：训练集与无标签测试集 A 的特征 PSI。无标签数据无法计算线上 AUC 或真实违约率趋势。
- **AI Analyst**：先路由到本地聚合计算，再将问题与聚合结果发给 DeepSeek 总结，并展示计算依据。原始客户记录不会发送给模型。当前不包含文档检索，因此无需 embedding/reranker。

## 数据与输出

`data/`、`outputs/`、`.env` 已加入 `.gitignore`。不要把原始贷款数据、模型文件、客户导出或 API 密钥提交到代码仓库。天池提交文件位于 `outputs/submission.csv`，格式与官方样例一致。

## 实现与边界

```text
baseline.py       分层留出验证、完整训练、比赛提交文件
prepare_demo.py   数据质量、客群指标、PSI、客户特征预计算
app.py            FastAPI 接口、后台样本实验、DeepSeek 问答
static/           演示页面
```

风险预测由本地机器学习模型计算；AI Analyst 先查询本地聚合指标，再把**问题与聚合结果**交给 DeepSeek 做文字总结。当前问答路由覆盖等级违约率、特征漂移、阈值影响和整体概览；没有文档库，所以暂不需要 embedding 或 reranker。

本版只供本地演示：训练任务结果保存在进程内，服务重启后会清空；上传文件仅预览，不作为新的训练数据集；模型监控比较训练集与无标签测试集的输入分布，不代表生产环境实时监控；客户画像使用单变量替换敏感度，而非 SHAP。权限、多租户、模型注册中心及线上自动决策流程均未实现。
