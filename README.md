# RiskLens · AI 信贷风险决策助手

基于天池 [贷款违约预测赛题](https://tianchi.aliyun.com/competition/entrance/531830) 的本地演示项目。帮助演示人员回答三个业务问题：**谁有风险？为什么值得关注？审核策略该怎么调整？**

页面中的客户数、违约率、AUC、阈值结果、风险分层、SHAP 与 PSI 均由本地比赛数据和模型计算。同一 16 万条留出验证集上，LR AUC 约 **0.7113**，XGBoost AUC 约 **0.7354**；这是本地参考值，不是天池线上分数。

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

首次运行先训练 LR 与 XGBoost，再生成页面所需聚合数据及 SHAP 摘要：

```powershell
.\.venv\Scripts\python.exe baseline.py
.\.venv\Scripts\python.exe prepare_demo.py
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/>。如果 8000 端口已被占用，把启动命令中的 `--port 8000` 改为 `--port 8001`，再打开对应地址。首次训练会读取 80 万条训练记录，耗时取决于机器配置。默认使用 CPU；有可用的 NVIDIA CUDA 环境时可执行 `python baseline.py --device cuda`。以后只查看演示时，执行最后一条启动命令即可。生成的模型、验证集预测、提交文件和 SHAP 数据都在 `outputs/`。

如需 DeepSeek 总结，在启动服务前配置进程环境变量 `DEEPSEEK_API_KEY`。默认模型为 `deepseek-flash`，可通过 `DEEPSEEK_MODEL` 改写。调用使用 [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)；密钥只在服务端读取，不写入项目文件。未配置或调用失败时，问答保留本地计算证据，并显示在线总结不可用。

## 演示路线

1. **风险总览**：按信用等级、贷款用途、收入和风险等级筛选 20 万待评估客户；查看风险分布、描述性风险线索与优先审查客户。
2. **客户审查**：查看 XGBoost 预测违约概率、关键资料和概率空间 SHAP 贡献路径；与 LR 基准的同客户预测对照。模拟审核标记只保存在当前页面。
3. **策略模拟**：改变审核阈值及成本假设，观察需审核人数、违约识别率、正常客户额外审核率与假设成本。技术混淆矩阵收在详情中。
4. **AI Analyst**：支持单客户 SHAP 解释、高风险及信用等级客群的抽样 SHAP 聚合、PD 筛选、阈值比较及模型健康问答；展示计算证据与操作入口。单客户解释在本地生成，不发送给 DeepSeek。

右上角 **高级 / 模型与数据** 保留数据质量、特征公式与全局 SHAP 洞察、LR/XGBoost 公平比较、AUC/KS/PR-AUC/Brier 与校准曲线、全局 `mean(|SHAP|)` 和散点分布、小样本实验，以及输入特征 PSI。业务首页不展示 ROC、缺失单元格或 PSI。

## 数据与输出

`data/`、`outputs/`、`.env` 已加入 `.gitignore`。不要把原始贷款数据、模型文件、客户导出或 API 密钥提交到代码仓库。天池提交文件位于 `outputs/submission.csv`，格式与官方样例一致。

## 实现与边界

```text
baseline.py       LR/XGBoost 同划分验证、完整训练、比赛提交文件
prepare_demo.py   数据质量、客群指标、PSI、概率空间 SHAP 摘要
app.py            FastAPI 接口、后台样本实验、DeepSeek 问答
static/           演示页面
```

风险预测由本地 XGBoost 计算；LR 用作同数据划分的透明基准。XGBoost 的轮数由训练部分中的内部切分选择，外层 16 万条留出集只用于两模型的最终比较。SHAP 使用 100 条训练背景样本的 interventional TreeExplainer，解释的是**概率空间**，因此基础概率与全部特征贡献可加总到当前模型预测。全局图和客群贡献来自测试集 A 的固定 300 条随机样本，并标注样本口径。AI Analyst 的单客户查询在本地用预测与 SHAP 回答；其他问题只把问题与聚合证据交给 DeepSeek。当前没有文档库，所以暂不需要 embedding 或 reranker。

本版只供本地演示：训练任务结果保存在进程内，服务重启后会清空；上传文件仅预览，不作为新的训练数据集；模型监控比较训练集与无标签测试集的输入分布，不代表生产环境实时监控。**SHAP 解释模型预测，不代表违约的因果原因，也不能单独用于信贷决策。**测试集没有真实标签，不能报告线上 AUC 或真实违约率变化。权限、多租户、模型注册中心及线上自动决策流程均未实现。SHAP 的方法与输出空间见 [官方 TreeExplainer 文档](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)。
