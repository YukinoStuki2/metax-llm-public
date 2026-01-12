---
title: 本地评测（eval_local.py）
sidebar_position: 40
---

# 本地评测脚本详解

## 一、脚本功能概述

### 1.1 核心作用

`eval_local.py` 是项目中用于**本地复现线上评测流程**的核心工具。它通过 HTTP 接口调用正在运行的模型服务,使用 RougeL-F1 指标(基于 jieba 分词)对模型的问答能力进行量化评估,并输出详细的性能统计数据。

### 1.2 主要功能

- **问答评测**：从 `.docx` 格式的问答文档中加载题目和参考答案,调用模型接口获取预测答案,计算 RougeL-F1 分数。
- **性能统计**：统计推理耗时、吞吐量(tokens/s)、请求成功率等关键性能指标。
- **批量推理模式**：支持单条请求和批量请求两种模式,批量模式更接近线上评测机的实际行为。
- **Token 计数**：可加载 tokenizer 对输入输出进行 token 统计,便于分析模型的真实吞吐性能。
- **详细日志**：将每道题目的详细评测结果保存为 JSONL 格式,方便后续分析和调试。
- **调试功能**：支持打印前 N 个或随机 N 个样本的详细信息,用于快速检查模型输出质量。

### 1.3 在项目中的定位

- **调优依据**：`auto_tune.py` 自动调参脚本会循环调用本脚本,解析其输出的准确率和吞吐量指标,作为参数搜索的优化目标。
- **本地验证**：在提交到线上评测平台之前,可使用本脚本在本地验证模型的准确率和性能表现。
- **问题诊断**：通过详细的 JSONL 输出,可快速定位模型在哪些问题上表现不佳,辅助模型优化。

### 1.4 输入与输出

**输入：**
- `.docx` 格式的问答文档(如 `basic.docx`、`plus.docx`)
- 运行中的模型服务的 HTTP 端点
- (可选)本地 tokenizer 路径(用于 token 统计)

**输出：**
- 控制台打印：准确率、吞吐量、耗时等统计信息
- JSONL 文件：每道题的详细评测结果(问题、参考答案、预测答案、分数、耗时、token 数等)

**不改变：**
- 不修改模型文件、配置文件或任何持久化状态
- 仅进行只读评测,不影响模型服务的运行状态

---

## 二、参数与环境变量详解

### 2.1 命令行参数

#### 网络相关参数

**`--endpoint`**
- **作用**：模型推理接口的完整 URL
- **默认值**：`http://127.0.0.1:8000/predict`
- **格式要求**：完整的 HTTP/HTTPS URL,服务端需实现 `POST /predict` 接口
- **接口协议**：
  - 请求：`{"prompt": "问题文本"}` 或 `{"prompt": ["问题1", "问题2", ...]}`(batch 模式)
  - 响应：`{"response": "答案文本"}` 或 `{"response": ["答案1", "答案2", ...]}`(batch 模式)
- **使用示例**：`--endpoint http://192.168.1.100:8000/predict`

**`--health`**
- **作用**：健康检查接口的 URL
- **默认值**：`http://127.0.0.1:8000/`
- **说明**：脚本启动时会先访问此接口,检查服务是否可用。检查失败不会终止脚本,仅输出警告。
- **使用示例**：`--health http://localhost:8000/health`

**`--timeout`**
- **作用**：单次 HTTP 请求的超时时间(秒)
- **默认值**：`300`(5 分钟)
- **适用场景**：
  - 模型推理速度较慢时,可适当增大此值
  - Batch 模式下处理大量问题时,建议设置更大的超时时间
- **使用示例**：`--timeout 600`

#### 数据集相关参数

**`--basic_docx`**
- **作用**：基础题集的 `.docx` 文件路径
- **默认值**：`basic.docx`(当前目录下)
- **格式要求**：文档中每道题需遵循固定格式：
  ```
  问题：题目文本
  答案：参考答案文本
  ```
  问题和答案之间可有其他段落,脚本会自动识别并配对。
- **使用示例**：`--basic_docx /data/eval/basic.docx`

**`--bonus_docx`**
- **作用**：加分题集的 `.docx` 文件路径
- **默认值**：`plus.docx`
- **说明**：用于评测更高难度或额外的问题集
- **使用示例**：`--bonus_docx /data/eval/plus.docx`

**`--which`**
- **作用**：选择要评测的数据集
- **可选值**：`basic`(仅基础题)、`bonus`(仅加分题)、`all`(两个集合都测)
- **默认值**：`all`
- **使用示例**：`--which basic`

**`--max_n`**
- **作用**：限制每个数据集最多评测的题目数量
- **默认值**：`0`(不限制,评测全部题目)
- **适用场景**：快速验证或调试时,只评测前几道题
- **使用示例**：`--max_n 10`(每个数据集只测前 10 题)

#### 评测模式参数

**`--batch`**
- **作用**：启用批量推理模式
- **默认值**：`False`(逐条请求模式)
- **区别**：
  - **逐条模式**：每道题发送一次独立的 HTTP 请求,`prompt` 为单个字符串
  - **批量模式**：将所有题目一次性打包发送,`prompt` 为字符串列表
- **推荐场景**：批量模式更接近线上评测机的行为,且能更准确地测试模型的并发处理能力
- **使用示例**：`--batch`

**`--sleep`**
- **作用**：逐条模式下,每次请求后等待的时间(秒)
- **默认值**：`0.0`(不等待)
- **适用场景**：避免过快的连续请求对服务造成压力,或用于模拟生产环境的请求间隔
- **注意**：批量模式下此参数无效
- **使用示例**：`--sleep 0.5`(每次请求后等待 0.5 秒)

#### Token 统计相关参数

**`--model_dir_for_tokenizer`**
- **作用**：指定 tokenizer 的本地路径或模型 ID
- **默认值**：`""`(空字符串,不加载 tokenizer)
- **说明**：
  - 如果提供路径,脚本会尝试使用 `transformers.AutoTokenizer` 加载 tokenizer
  - 加载成功后,会对每道题的问题文本和答案文本进行 token 统计
  - Token 统计用于计算真实的吞吐量(tokens/s)
  - 如果不提供,token 统计会显示为 0,吞吐量指标不可用
- **兼容性**：需要本地安装 `transformers` 库,且模型 tokenizer 需支持 `AutoTokenizer`
- **使用示例**：
  - `--model_dir_for_tokenizer ./model/YukinoStuki/Qwen3-4B-Plus-LLM`
  - `--model_dir_for_tokenizer Qwen/Qwen-7B`

#### 输出与日志参数

**`--save_jsonl`**
- **作用**：保存详细评测结果的 JSONL 文件路径
- **默认值**：`eval_details.jsonl`
- **记录内容**：每行一个 JSON 对象,包含：
  - `dataset`：数据集名称(basic/bonus)
  - `idx`：题目索引
  - `question`：问题文本
  - `ref`：参考答案
  - `pred_raw`：模型原始预测答案
  - `pred_clean`：清理后的预测答案(如果启用 `--strip_q_suffix`)
  - `ok`：请求是否成功
  - `latency_s`：单题推理耗时(秒)
  - `rougeL_f1_raw`：原始 RougeL-F1 分数
  - `rougeL_f1_clean`：清理后的分数(可选)
  - `prompt_tokens`：问题的 token 数
  - `output_tokens_raw`：原始答案的 token 数
  - `output_tokens_clean`：清理后答案的 token 数
  - `batch_total_latency_s`：批量模式下的总耗时(仅批量模式)
- **使用示例**：`--save_jsonl results/eval_20250112.jsonl`

**`--overwrite_jsonl`**
- **作用**：覆盖写入模式(而非追加模式)
- **默认值**：`False`(追加模式)
- **说明**：
  - **追加模式**：每次运行将结果追加到已有文件末尾,适合多次实验对比
  - **覆盖模式**：每次运行清空文件后重新写入,适合单次完整评测
- **推荐**：重复实验时建议使用 `--overwrite_jsonl`,避免文件越来越大
- **使用示例**：`--overwrite_jsonl`

**`--strip_q_suffix`**
- **作用**：启用后,额外计算一个"清理题目重复后缀"的分数
- **默认值**：`False`
- **背景**：某些模型在输出答案时,会在末尾重复题目文本(如"答案是...矩阵乘法中使用分块技术的优势是什么？")
- **清理逻辑**：检测并移除答案末尾与题目重复的部分
- **输出**：会同时输出 `RAW`(原始)和 `CLEAN`(清理后)的分数和吞吐量
- **适用场景**：诊断模型是否存在重复输出问题,对比清理前后的准确率变化
- **使用示例**：`--strip_q_suffix`

#### 调试参数

**`--debug_first_n`**
- **作用**：打印前 N 道题的详细信息
- **默认值**：`0`(不打印)
- **输出内容**：
  - 问题文本
  - 预测答案预览(截断到 240 字符)
  - RougeL-F1 分数
  - Token 统计(如果加载了 tokenizer)
- **适用场景**：快速查看模型在最初几道题上的表现
- **使用示例**：`--debug_first_n 5`

**`--debug_random_n`**
- **作用**：从第 `debug_first_n` 题之后,随机抽取 N 道题打印详细信息
- **默认值**：`0`(不打印)
- **说明**：采样范围是 `[debug_first_n, 总题数)`,避免与前 N 题重复
- **适用场景**：检查模型在整个数据集上的表现,而不仅仅是前几题
- **使用示例**：`--debug_random_n 10`(随机抽取 10 道题)

**`--debug_random_seed`**
- **作用**：控制随机采样的种子
- **默认值**：`None`(每次运行使用不同的随机种子)
- **说明**：
  - 如果不指定,脚本会自动生成一个随机种子并打印到控制台
  - 指定种子后,每次运行会采样相同的题目,便于复现
- **使用示例**：`--debug_random_seed 42`

### 2.2 环境变量

本脚本**不依赖任何环境变量**,所有配置均通过命令行参数传递。这确保了脚本的可移植性和可重复性。

### 2.3 依赖的外部服务

**模型服务要求：**
- 必须提供 `GET /`(健康检查)和 `POST /predict`(推理)两个接口
- `/predict` 接口的请求和响应格式需符合脚本的协议约定
- 服务需在脚本启动前已经运行

**依赖文件：**
- `.docx` 格式的问答文档(需符合固定格式)
- (可选)本地 tokenizer 文件

---

## 三、代码实现详解

### 3.1 导入的核心库

#### 标准库
- **`argparse`**：解析命令行参数
- **`json`**：处理 JSON 数据的序列化与反序列化
- **`os`**：文件路径检查和随机数生成
- **`re`**：正则表达式,用于解析 `.docx` 文档中的问答格式
- **`time`**：测量推理耗时
- **`random`**：随机采样调试样本

#### 第三方库
- **`requests`**：发送 HTTP 请求到模型服务
- **`docx` (python-docx)**：解析 `.docx` 格式的问答文档
- **`jieba`**：中文分词,用于 RougeL-F1 计算
- **`rouge_score`**：计算 RougeL-F1 分数
- **`tqdm`**：显示进度条
- **`transformers`**：加载 tokenizer 进行 token 统计(可选依赖)

### 3.2 核心函数说明

#### `parse_docx_qa(docx_path: str) -> List[Tuple[str, str]]`

**功能：** 从 `.docx` 文档中解析问答对。

**实现逻辑：**
1. 使用 `python-docx` 读取文档的所有段落
2. 通过正则表达式匹配 `问题：` 和 `答案：` 开头的段落
3. 支持多段问题和多段答案的续行(答案较长时可能跨越多个段落)
4. 将每对问答以 `(问题, 答案)` 元组的形式存入列表

**关键正则：**
- 问题模式：`^\s*问题[:：]\s*(.*)\s*$`
- 答案模式：`^\s*答案[:：]\s*(.*)\s*$`

**续行处理：**
- 如果当前段落既不是问题也不是答案,但存在正在收集的答案,则将该段落追加到答案中
- 同理,如果存在正在收集的问题但答案尚未开始,则追加到问题中

**返回值：** 问答对列表 `[(问题1, 答案1), (问题2, 答案2), ...]`

---

#### `rougeL_f1(pred: str, ref: str, scorer: rouge_scorer.RougeScorer) -> float`

**功能：** 计算预测答案和参考答案之间的 RougeL-F1 分数。

**实现逻辑：**
1. 使用 `jieba.lcut()` 对预测答案和参考答案进行中文分词
2. 将分词结果用空格连接成字符串(`rouge_scorer` 要求输入为空格分隔的 token)
3. 调用 `scorer.score(ref_tokens, pred_tokens)` 计算 RougeL 分数
4. 返回 F1 值(Precision 和 Recall 的调和平均数)

**特殊处理：**
- 如果预测答案或参考答案为空,直接返回 0.0

**注意事项：**
- 参数顺序为 `scorer.score(ref_tokens, pred_tokens)`,与某些库的习惯相反,需保持与评测机一致

---

#### `strip_question_suffix(answer: str, question: str) -> str`

**功能：** 启发式清理答案末尾重复附带的题目文本。

**背景：** 某些模型在生成答案时,会在末尾无意重复输出题目,如：
```
答案：矩阵分块技术可以提高计算效率。矩阵乘法中使用分块技术的优势是什么？
```

**实现逻辑：**
1. **精确后缀匹配**：检查答案是否以题目文本结尾,如果是,直接移除
2. **去除空白后匹配**：将答案和题目都去除所有空白符后比较,避免因空格差异导致匹配失败
3. **启发式定位**：如果检测到重复但位置不在最末尾,仅在重复位置超过答案长度一半时才移除

**使用场景：**
- 配合 `--strip_q_suffix` 参数使用
- 用于诊断模型是否存在重复输出问题,以及评估清理后的准确率提升

---

#### `load_tokenizer(model_dir_or_id: Optional[str])`

**功能：** 加载 tokenizer 用于 token 统计。

**实现逻辑：**
1. 检查是否安装了 `transformers` 库
2. 使用 `AutoTokenizer.from_pretrained()` 加载 tokenizer
3. 优先使用 `use_fast=False` 避免某些兼容性问题
4. 如果加载失败,输出警告但不中断脚本

**参数处理：**
- `trust_remote_code=True`：允许加载包含自定义代码的模型
- `use_fast=False`：使用慢速但更稳定的 tokenizer

**返回值：** tokenizer 对象或 `None`(加载失败时)

---

#### `count_tokens(tokenizer, text: str) -> int`

**功能：** 统计文本的 token 数量。

**实现逻辑：**
1. 如果 tokenizer 未加载,返回 0
2. 使用 `tokenizer.encode(text, add_special_tokens=False)` 编码文本
3. 返回 token 列表的长度

**注意：**
- `add_special_tokens=False` 排除 `[CLS]`、`[SEP]` 等特殊 token,只统计实际内容

---

#### `call_predict(endpoint: str, prompt: str, timeout: int) -> str`

**功能：** 单条推理模式,发送单个问题到模型服务。

**请求格式：**
```json
POST {endpoint}
Content-Type: application/json

{"prompt": "问题文本"}
```

**响应处理：**
1. 期望响应为 `{"response": "答案文本"}`
2. 如果响应格式不符合预期,将整个响应序列化为字符串返回
3. 如果请求失败,抛出异常由调用者处理

**返回值：** 模型预测的答案字符串

---

#### `call_predict_batch(endpoint: str, prompts: List[str], timeout: int) -> List[str]`

**功能：** 批量推理模式,一次性发送多个问题到模型服务。

**请求格式：**
```json
POST {endpoint}
Content-Type: application/json

{"prompt": ["问题1", "问题2", "问题3"]}
```

**响应处理：**
1. 期望响应为 `{"response": ["答案1", "答案2", "答案3"]}`
2. 如果 `response` 是单个字符串,复制成与 `prompts` 同长度的列表
3. 如果 `response` 长度不足,用空字符串补齐
4. 如果响应格式不符合预期,尝试多种兜底方案

**兼容性：**
- 兼容返回字符串列表的标准协议
- 兼容返回单个字符串的简化实现
- 兼容直接返回列表而非字典的非标准实现

**返回值：** 答案字符串列表,长度与 `prompts` 一致

---

#### `main()` 函数

**功能：** 脚本的主执行流程。

**执行步骤：**

1. **解析命令行参数**：使用 `argparse` 获取所有配置
2. **健康检查**：尝试访问 `--health` 接口,检查服务是否可用(失败不中断)
3. **清空 JSONL 文件**(如果指定 `--overwrite_jsonl`)：确保每次运行从空白开始
4. **加载 tokenizer**(如果指定 `--model_dir_for_tokenizer`)：用于后续的 token 统计
5. **初始化 RougeL scorer**：创建 `rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)`
6. **确定评测数据集**：根据 `--which` 参数决定评测哪些数据集
7. **循环处理每个数据集**：
   - 加载问答对
   - 截取前 `max_n` 题(如果指定)
   - 根据 `--batch` 参数选择推理模式
8. **逐条模式循环**(非 batch)：
   - 对每道题调用 `call_predict()`
   - 测量推理耗时
   - 计算 RougeL-F1 分数
   - 统计 token 数量
   - 如果启用 `--strip_q_suffix`,额外计算清理后的分数
   - 将结果写入 JSONL 文件
   - 如果指定 `--sleep`,等待一段时间
9. **批量模式处理**(batch)：
   - 将所有问题打包成列表
   - 一次性调用 `call_predict_batch()`
   - 测量总耗时并平均分配到每道题
   - 后续处理与逐条模式相同
10. **统计汇总**：
    - 计算平均准确率(总分数 / 题目数量)
    - 计算吞吐量(总 token 数 / 总耗时)
    - 打印统计结果
11. **调试输出**(如果启用 `--debug_first_n` 或 `--debug_random_n`)：
    - 打印指定题目的详细信息
    - 显示问题、答案预览、分数、token 统计

**异常处理：**
- 健康检查失败：仅警告,不中断
- 推理请求失败：记录为 `ok=False`,分数为 0,答案为 `[ERROR] ...`
- tokenizer 加载失败：仅警告,token 统计为 0

---

### 3.3 评测指标计算

#### RougeL-F1 计算流程

1. **分词**：使用 `jieba.lcut()` 对预测答案和参考答案进行分词
2. **空格连接**：将分词结果用空格连接成字符串
3. **计算 RougeL**：调用 `rouge_scorer.score(ref_tokens, pred_tokens)`
4. **提取 F1**：从结果中提取 `rougeL.fmeasure` 字段

**RougeL 的含义：**
- **Longest Common Subsequence (LCS)**：最长公共子序列
- **Recall**：LCS 长度 / 参考答案长度
- **Precision**：LCS 长度 / 预测答案长度
- **F1**：Precision 和 Recall 的调和平均数

#### 吞吐量计算

1. **Answer tokens/s**：`总输出 token 数 / 总耗时`
   - 仅统计模型生成的答案部分
2. **(Prompt+Answer) tokens/s**：`(总输入 token 数 + 总输出 token 数) / 总耗时`
   - 统计完整的输入输出

**注意：** 吞吐量指标依赖 tokenizer,如果未加载则为 0

---

### 3.4 JSONL 输出格式

每行一个 JSON 对象,包含以下字段：

```json
{
  "dataset": "basic",
  "idx": 0,
  "question": "问题文本",
  "ref": "参考答案",
  "pred_raw": "模型输出",
  "pred_clean": "清理后的输出",
  "ok": true,
  "latency_s": 1.234,
  "rougeL_f1_raw": 0.8765,
  "rougeL_f1_clean": 0.8900,
  "prompt_tokens": 25,
  "output_tokens_raw": 150,
  "output_tokens_clean": 140,
  "batch_total_latency_s": 30.0
}
```

---

## 四、常见问题

### 4.1 评测相关问题

**Q1: 为什么我的准确率和线上评测结果不一致？**

A: 可能的原因：
- **分词差异**：确保使用的 `jieba` 版本与评测机一致
- **推理模式差异**：线上评测使用批量模式,建议本地也使用 `--batch` 参数
- **参数配置差异**：检查模型服务的配置参数(如 `temperature`、`top_p` 等)是否与线上一致
- **数据集差异**：确认使用的 `.docx` 文件与评测机完全相同

**Q2: 为什么有些题目得分为 0？**

A: 可能的原因：
- **推理失败**：检查 JSONL 文件中的 `ok` 字段,如果为 `false` 说明请求失败
- **空答案**：模型输出了空字符串或 `None`
- **答案完全不匹配**：模型输出与参考答案没有任何公共子序列(极少见)

**Q3: `--strip_q_suffix` 后准确率为什么反而下降了？**

A: 这说明模型输出的答案本身就很短,或者清理逻辑误删了部分有效内容。此参数仅用于诊断,不建议作为正式评测依据。

---

### 4.2 性能相关问题

**Q4: 吞吐量显示为 0 或很低,但模型推理很快？**

A: 可能的原因：
- **未加载 tokenizer**：检查是否指定了 `--model_dir_for_tokenizer` 参数
- **tokenizer 加载失败**：查看控制台是否有警告信息
- **tokenizer 与模型不匹配**：尝试使用正确的 tokenizer 路径

**Q5: 批量模式和逐条模式的吞吐量差异很大？**

A: 这是正常现象：
- **批量模式**：将所有题目的耗时平均分配,无法反映单题的真实延迟
- **逐条模式**：每题有独立的耗时,但总吞吐量可能低于批量模式
- 建议：使用批量模式评估整体吞吐能力,使用逐条模式分析单题性能

---

### 4.3 网络与服务问题

**Q6: 健康检查失败,但脚本继续运行了？**

A: 健康检查失败不会中断脚本,因为某些服务可能不提供健康检查接口。脚本会继续尝试调用 `/predict` 接口。

**Q7: 请求超时,应该如何调整？**

A: 
- 增大 `--timeout` 参数(默认 300 秒)
- 批量模式下,超时时间应设置为 `单题推理时间 × 题目数量 + 一定冗余`
- 如果经常超时,考虑优化模型推理性能

**Q8: 如何测试远程服务器上的模型？**

A: 将 `--endpoint` 和 `--health` 参数改为远程服务器的地址：
```bash
python3 eval_local.py \
  --endpoint http://192.168.1.100:8000/predict \
  --health http://192.168.1.100:8000/
```

---

### 4.4 数据相关问题

**Q9: 如何添加自定义的评测题目？**

A: 
1. 准备一个 `.docx` 文件
2. 按照格式要求添加问答对：
   ```
   问题：你的问题1
   答案：参考答案1
   
   问题：你的问题2
   答案：参考答案2
   ```
3. 使用参数指定文件路径：
   ```bash
   python3 eval_local.py --basic_docx your_custom.docx --which basic
   ```

**Q10: 评测文档的格式要求严格吗？**

A: 
- **必须包含**：`问题：` 和 `答案：` 标记(支持中英文冒号)
- **可选内容**：标题、说明、代码块等其他段落会被自动忽略
- **续行支持**：问题和答案可以跨越多个段落,脚本会自动拼接

---

### 4.5 调试相关问题

**Q11: 如何快速查看模型在某些题目上的输出？**

A: 使用调试参数：
```bash
# 查看前 5 题
python3 eval_local.py --debug_first_n 5

# 查看前 5 题 + 随机 10 题
python3 eval_local.py --debug_first_n 5 --debug_random_n 10

# 固定随机种子以复现
python3 eval_local.py --debug_first_n 5 --debug_random_n 10 --debug_random_seed 42
```

**Q12: 如何分析某道题的详细信息？**

A: 
1. 运行评测并保存 JSONL：`python3 eval_local.py --save_jsonl results.jsonl`
2. 使用 `jq` 或 Python 查询特定题目：
   ```bash
   # 查看第 10 题
   cat results.jsonl | jq 'select(.idx == 10)'
   
   # 查看得分低于 0.3 的题目
   cat results.jsonl | jq 'select(.rougeL_f1_raw < 0.3)'
   ```

**Q13: 如何对比不同参数配置的评测结果？**

A: 
1. 每次评测使用不同的 JSONL 文件名：
   ```bash
   python3 eval_local.py --save_jsonl run1.jsonl --overwrite_jsonl
   python3 eval_local.py --save_jsonl run2.jsonl --overwrite_jsonl
   ```
2. 使用脚本对比分数：
   ```python
   import json
   
   def avg_score(jsonl_path):
       scores = []
       with open(jsonl_path) as f:
           for line in f:
               rec = json.loads(line)
               if rec["ok"]:
                   scores.append(rec["rougeL_f1_raw"])
       return sum(scores) / len(scores) if scores else 0
   
   print("Run1:", avg_score("run1.jsonl"))
   print("Run2:", avg_score("run2.jsonl"))
   ```

---

### 4.6 与 auto_tune 的集成问题

**Q14: auto_tune.py 无法解析 eval_local 的输出？**

A: 
- 确保未修改 `eval_local.py` 中的输出格式
- `auto_tune.py` 依赖以下关键行的格式：
  - `Accuracy (RougeL-F1 mean, RAW): 0.xxxx`
  - `Throughput RAW: answer_tokens/s=xx.xx, (prompt+answer)_tokens/s=xx.xx`
  - `Total time: xx.xxs`
- 如需自定义输出,同步修改 `auto_tune.py` 中的正则表达式

**Q15: 如何单独运行 eval_local 而不触发 auto_tune？**

A: 直接运行 `eval_local.py` 脚本即可,它是独立的评测工具：
```bash
python3 eval_local.py --which bonus --batch
```

---

### 4.7 依赖与环境问题

**Q16: 提示 `transformers` 未安装？**

A: 
- 如果不需要 token 统计,可忽略此警告
- 如需 token 统计,安装 transformers：
  ```bash
  pip install transformers
  ```

**Q17: 提示找不到 `basic.docx` 或 `plus.docx`？**

A: 
- 确保文件存在于脚本运行目录
- 或使用绝对路径指定：
  ```bash
  python3 eval_local.py --basic_docx /path/to/basic.docx
  ```

**Q18: `jieba` 分词结果不符合预期？**

A: 
- 确保 `jieba` 版本与评测机一致
- 如需自定义词典,在脚本开头添加：
  ```python
  import jieba
  jieba.load_userdict("custom_dict.txt")
  ```
