# 来源、修改与材料状态

整理日期：2026-11-22。

1. 实际读取来源：用户 Mac 的 `Desktop/my-math-agent`，复制 60 个 Python/JSON/CSV 文件；没有复制 `.env`、虚拟环境、缓存或整个 Git 历史。逐文件原始 SHA-256 见 `source_manifest.json`。
2. 远程先核查 `cicilem429-sys/robust-localization-agent`，main 树为 `c467b37e5bc52db93e700978a8ad8756c1bb5849`，仅含早期离线原型与进展 PDF。它不是此次 V1–V6 的最终代码来源。
3. 修改 `research_agent_v6.py`：将 dotenv/OpenAI 导入、读取密钥、创建 API 客户端推迟到实际调用模型时。数值公式、候选参数与优化流程保持原样。
4. 修改 `final_statistical_validation.py`：新增 `--output-dir` 参数，默认保存到 `results/reproduced/`。原始指标文件留在根目录。
5. 新增 README、依赖、忽略规则、核验程序、报告生成器、SVG 图表与最终 PDF。图表来自保存的数值，不是示意数据。
6. V1 对应 `localization_agent.py`；无版本后缀的 `research_agent.py` 是更早的回归研究练习，并非与 V3–V6 同环境的定位实验。各版本数据配置不同，不将它们的得分当成单一递增曲线。
7. 缺失：V1 完整模型对话、其他版本完整 API 请求/响应轨迹、原始服务账单、600 组逐样本定位误差存档、真实设备观测。现有结构化历史不能替代这些材料。原先没有最终 PDF，本次依据实际源码与保存结果新写研究报告。
8. 密钥防护：按文件白名单复制，排除环境配置，扫描常见 token/私钥模式及代码中的凭据赋值。环境变量名称不是密钥；不生成或上传 `.env`。
