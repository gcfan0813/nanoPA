[TOC]

------

# 书生·普语大模型实战营第二期——用茴香豆搭建个人的RAG知识助手

## 一、RAG的基础知识

### 1. 什么是RAG

​	RAG（Retrieval Augmented Generation）技术，通过检索与用户输入相关的信息片段，并结合***外部知识库***来生成更准确、更丰富的回答。解决 LLMs 在处理知识密集型任务时可能遇到的挑战, 如幻觉、知识过时和缺乏透明、可追溯的推理过程等。提供更准确的回答、降低推理成本、实现外部记忆。

![image-20240408094755689](./assets/image-20240408094755689.png)

**特点**：可以解决大型模型在处理知识密集型任务时面临的各种挑战，如生成幻觉等问题；可以让大型模型具备外部记忆功能，在不需要额外训练的情况下就能获取新知识，降低了整体的成本。

![image-20240408095500160](./assets/image-20240408095500160.png)

**应用**：问答系统、文本生成系统、信息检索，以及在结合了多模态大模型之后，RAG技术也能够用于图片的描述等。

![image-20240408095527430](./assets/image-20240408095527430.png)

### 2. RAG的工作原理

- 经典的RNG由三个部分组成：**索引(indexing)、检索(retrieval)、生成(generation)**
- **索引**部分负责处理外部知识，将知识源（如文档、网页）分割成trunk，然后编码成向量，并存储在专用的向量数据库中
- **检索**部分负责接收用户的问题，然后将问题也编码成向量，在向量数据库中找出与问题最相关的内容
- **生成**部分负责将检索到的内容和原始问题一起作为提示，输入到大模型中，生成最终的答案

![image-20240408095743502](./assets/image-20240408095743502.png)

​	**向量数据库（vector database）**

​	向量数据库是RAG技术当中**专门储存外部数据**的地方，主要是将文本及相关的数据，通过预训练的模型转换为固定长度的向量，这些向量要能够很好地捕捉到我们文本和知识的语义信息及内部联系。

​	向量数据库是**实现快速准确回答的基础**，要能够高效地实现相似性检索，根据用户的查询，快速找出最相关的向量。

​	在面向大规模数据的时候以及需要高速响应的需求的时候，向量数据库是需要进行优化的，很重要的就是对**向量表示**的优化，例如使用更高级的文本编码技术、使用更好的预训练模型等，或去尝试不同的句子嵌入或段落嵌入方法等。

![image-20240410121401322](./assets/image-20240410121401322.png)

### 3. RAG工作流程

1. 用户输入一个问题或查询。
2. 预处理部分将用户的输入进行筛选和转换，将其转化为合适的问询。
3. 在外部知识库中搜寻相关的内容，并结合大语言模型的能力生成回答。

![image-20240408101449275](./assets/image-20240408101449275.png)

​	针对一些专业性知识或者高时效信息，只需要**不断的更新这个向量数据库**，就可以将大模型的生成能力和我们向量数据库当中的知识内容很好地结合在一起，无需任何训练就能让大模型拥有并处理这些新的知识。

### 4. RAG的发展进程

![image-20240408101100065](./assets/image-20240408101100065.png)

从提出到现在不到4年的时间已经出现了三种RAG的范式

#### Naive RAG

> 只有索引、检索、生成三个部分构成的最基础的方式

应用：简单的问答系统和信息检索场景

<img src="./assets/image-20240408101226538.png" alt="image-20240408100923416" style="zoom:67%;" />

#### Advanced RAG

> 在三个基础部分之外对检索前后都进行了增强,在检索之前对用户的问题进行路由扩展、重写等处理力**（pre-retrieval）**，对于检索到的信息进行重排序、总结融合等处理**（post-retrieval）**，使信息收集和处理效率更高

应用：摘要生成、内容推荐等

![image-20240408101240144](./assets/image-20240408101240144.png)

#### Modular RAG

> 将RAG的基础部分和后续各种优化技术和功能**模块化**

应用：可以根据实际业务需求定制完成如多模态任务对话系统等更高级的应用

![image-20240408101258914](./assets/image-20240408101258914.png)

### 5. RAG常见的优化方法

**嵌入式优化**是通过结合稀疏编码器、密集检索器以及多任务的方式来增强嵌入的性能；

**索引优化**是通过增强数据力度、优化索引结构等多种策略来提升索引的质量。

> 嵌入式优化和索引优化是是用来提高向量数据库的质量，从而对RAG的性能进行提升

**查询过程优化**是通过查询扩展和转换等方式使用户的原始问题更适合检索任务；

**上下文管理**是通过重排和上下文选择压缩来减少检索的冗余信息并提高大模型的处理效率。

> 查询过程优化和上下文管理，既Advanced RAG范式中的前检索（pre-retrieval）和后检索（post-retrieval）部分

**迭代检索**是在RAG过程中根据检索结果多次迭代检索知识，为大模型生成提供全面的知识基础；

**递归检索**是通过迭代细化查询，来改进搜索结果的*深度和相关性*，使用了链式推理来指导检索过程，并根据检索结果细化推理过程；

**自适应检索**是用Flare、Self-RAG等，让大模型能够自主的决定他所要检索的内容、最佳时机等因素。

> 迭代检索、递归检索、自适应检索是RAG的检索部分（retrieval）优化的三种常见方式

**LLM微调**可以根据场景和数据特征对大模型进行定向微调，也可以根据大模型对于检索或生成的参与进行有针对性的微调。

### 6. RAG vs. Fine-tuning

|             | 场景                               | 优势                           | 局限                                               |
| ----------- | ---------------------------------- | ------------------------------ | -------------------------------------------------- |
| RAG         | 需要结合最新信息和实时数据的任务   | 动态知识更新，处理长尾知识问题 | 依赖于外部知识库的质量和覆盖范围。依赖大模型的能力 |
| Fine-tuning | 数据可用且需要模型高度专业化的任务 | 模型能针对特定任务优化         | 需要大量的标注数据，且对新任务的适应性较差         |

> 虽然微调可以在一定程度上改善模型的表现，但其也需要面临诸多挑战和限制。例如需要大量的标注数据才能有效实施微调；可能导致过度拟合等问题影响泛化能力；并且每次信息更新都可能需要再次进行

![image-20240408105140075](./assets/image-20240408105140075.png)

### 7. 如何评价一个RAG技术

> 通常将RAG技术当中的检索阶段和生成阶段进行单独的分别评价

- 传统NLP领域的经典评估指标可以用于RAG的检索过程和生成过程的评价；

- 专门的RAG评测框架

![image-20240410132319377](./assets/image-20240410132319377.png)

### 8. 总结

![image-20240408105402976](./assets/image-20240408105402976.png)

------

## 二、茴香豆

​	茴香豆（豆哥）是一款基于Retrieval Augmented Generation（RAG）技术的知识助手应用。RAG技术通过检索与用户输入相关的信息片段，并结合外部知识库来生成更准确、更丰富的回答，它能够帮助用户快速获取知识，且无需训练就可以掌握新领域的知识，从而解决大型语言模型在处理知识密集型任务时可能遇到的挑战。

### 1. 应用特点

​	茴香豆应用能够通过RAG技术，让基础模型实现非参数知识更新，无需训练就可以掌握新领域的知识。此外，茴香豆还支持从本地向量数据库中检索内容进行回答，也可以加入网络的搜索结果，生成回答。

### 2. 应用场景

​	茴香豆可以应用于各种需要知识问答的场景，如客服机器人、智能家居等。通过茴香豆，用户可以快速、高效地获取到他们所需要的知识。

![image-20240408105956354](./assets/image-20240408105956354.png)

> 它具有开放源代码、可免费商业使用的优点

![image-20240408110153059](./assets/image-20240408110153059.png)

> 具备良好的安全性及扩展性，它可以被应用于各类即时通信软件或交流社区中，也能与其他大型模型及其云接口相结合，提供了非常灵活的选择空间

![image-20240408110208418](./assets/image-20240408110208418.png)

### 3. 构成及工作流

![image-20240408110744772](./assets/image-20240408110744772.png)

- **知识库**：通常为企业内部或个人所在领域的专业技术文档。支持markdown、Pdf、Word、txt等常用文件格式。

- **前端**：问答助手用于读取和回答用户提问的平台。如微信、飞书等。

- **大模型**：支持本地调用大模型，如书生浦语和通义千问的模型格式，以及远端大模型的API及API集成工具，如kimi、chat gtp、deepsick、chatglm等。

- **工作流**：通过设置问题相关性阈值，对用户问题进行判断，决定是否需要进行回答。

  ​	**预处理**(preprocess)：将用户的输入筛选，然后转换为合适的问询

  ​	**拒答**(rejection pipeline)：通过对该问询本身的分析，以及该问询与数据库当中示例问题的比较来给出问题相关性的得分，根据得分来判断该问题是否要进入回答环节

  ​	**应答**(response pipeline)：大模型根据问询和检索到的知识内容进行答案的生成并返回给用户

  ​	<img src="./assets/image-20240408111350752.png" alt="image-20240408111350752" style="zoom:50%;" />

- **应答模块**：采用多来源检索、混合模型及安全评估等方式，保证回答内容的准确性。在生成部分，既可以使用本地模型，也可以使用远端模型，使用混合的模式来共同处理生成任务。

- **安全检测模块**：在直接输出给最终用户前，会对回答内容进行安全检测，确保符合要求。

![image-20240410135212427](./assets/image-20240410135212427.png)

------

## 三、实践

### （一）、茴香豆web版

#### 1. 登录茴香豆web版页面

https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web

![image-20240409084400239](./assets/image-20240409084400239.png)

#### 2. 创建（进入）自己的知识库

![image-20240409084555775](./assets/image-20240409084555775.png)

> 知识库的名称要大于8个字符

![image-20240409084706323](./assets/image-20240409084706323.png)

#### 3. 上传自己领域的知识文档到服务器上

> web版目前支持的文档格式很丰富

![image-20240409084810905](./assets/image-20240409084810905.png)

打开上传窗口

<img src="./assets/image-20240409084821085.png" alt="image-20240409084821085" style="zoom:67%;" />

选择需要的文件进行上传

<img src="./assets/image-20240409091611650-1712729148279-1.png" alt="image-20240409091611650" style="zoom: 67%;" />

#### 4. 设置正反例

<img src="./assets/image-20240409091907374.png" alt="image-20240409091907374" style="zoom: 67%;" /><img src="./assets/image-20240409092015746.png" alt="image-20240409092015746" style="zoom: 67%;" />

#### 5. 在线知识助手对话

<img src="./assets/image-20240409101804350.png" alt="image-20240409101804350" style="zoom:67%;" />

> 可以提示回答所参考的文档
>
> 也出现过一个问题👇

<img src="./assets/image-20240409102536546.png" alt="image-20240409102536546" style="zoom:67%;" />

<img src="./assets/image-20240409102608779.png" alt="image-20240409102608779" style="zoom:67%;" />

> 分析应该是提问的方法不对，本身“茴香豆”在日常生活总就是一个食品，在我们这里是一个应用，我用了“怎么用”来进行提问，这个确实感觉有歧义。
>
> 更换了问法，回答就没问题了👇

<img src="./assets/image-20240409103020606.png" alt="image-20240409103020606" style="zoom:67%;" />

<img src="./assets/image-20240409103037195.png" alt="image-20240409103037195" style="zoom:67%;" />

> 所以，“提问的智慧”不光是在与人交流中，与大模型交流也一样需要。

#### 6. 调用端口部署到飞书群

- 创建bot应用

<img src="./assets/image-20240410160848902.png" alt="image-20240410160848902" style="zoom:67%;" />

- 将应用凭证信息填入茴香豆web版

  <img src="./assets/image-20240410161005552.png" alt="image-20240410161005552" style="zoom:67%;" />

  <img src="./assets/image-20240409131627677.png" alt="image-20240409131627677" style="zoom:67%;" />

- bot配置页面填入加密策略

<img src="./assets/image-20240409131747138.png" alt="image-20240409131747138" style="zoom:67%;" />

- 填入时间回调地址

<img src="./assets/image-20240409132028792.png" alt="image-20240409132028792" style="zoom:67%;" />

- 添加【接收消息】事件

<img src="./assets/image-20240409132232678.png" alt="image-20240409132232678" style="zoom:67%;" />

- 开通权限：im:chat:readonly 和 im:message:send_as_bot

<img src="./assets/image-20240409132402750.png" alt="image-20240409132402750" style="zoom:67%;" />

- 发布bot应用，并添加进群聊

![image-20240409132631472](./assets/image-20240409132631472.png)

- 给飞书群名加上后缀

![image-20240409132805046](./assets/image-20240409132805046.png)

开始对话

<img src="./assets/image-20240409133001170.png" alt="image-20240409133001170" style="zoom:67%;" />

<img src="./assets/image-20240409133058730.png" alt="image-20240409133058730" style="zoom:67%;" />

Done.

------

### （二）、在 `InternLM Studio` 上部署茴香豆

#### 1. 环境准备

**进入开发机，创建并激活基础环境**

```bash
# 从官方环境复制运行 InternLM 的基础环境
studio-conda -o internlm-base -t InternLM2_Huixiangdou

# 在本地查看环境列表
conda env list

# 激活 InternLM2_Huixiangdou python 虚拟环境
conda activate InternLM2_Huixiangdou
```

![image-20240409124934446](./assets/image-20240409124934446.png)

#### 2. 下载基础文件

**复制茴香豆所需模型文件**

```bash
# 复制BCE模型
ln -s /root/share/new_models/maidalun1020/bce-embedding-base_v1 /root/models/bce-embedding-base_v1
ln -s /root/share/new_models/maidalun1020/bce-reranker-base_v1 /root/models/bce-reranker-base_v1

# 复制大模型参数
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
```

![image-20240409124943102](./assets/image-20240409124943102.png)

#### 3. 下载安装茴香豆

**安装依赖**

```bash
# 新建requirements.txt
cd /root
echo "
	protobuf==4.25.3
	accelerate==0.28.0 
	aiohttp==3.9.3 
	auto-gptq==0.7.1 
	bcembedding==0.1.3 
	beautifulsoup4==4.8.2 
	einops==0.7.0 
	faiss-gpu==1.7.2 
	langchain==0.1.14 
	loguru==0.7.2 
	lxml_html_clean==0.1.0
    openai==1.16.1 
    openpyxl==3.1.2
    pandas==2.2.1 
    pydantic==2.6.4
    pymupdf==1.24.1 
    python-docx==1.1.0
    pytoml==0.1.21
    readability-lxml==0.8.1 
    redis==5.0.3 
    requests==2.31.0 
    scikit-learn==1.4.1.post1 
    sentence_transformers==2.2.2 
    textract==1.6.5 tiktoken==0.6.0 
    transformers==4.39.3 
    transformers_stream_generator==0.0.5 
    unstructured==0.11.2
" > requirements.txt

# 安装python依赖 (不含Word文件解析)
 pip install -r requirements.txt
```

<img src="./assets/image-20240408123432161.png" alt="image-20240408123432161" style="zoom:67%;" />

**下载茴香豆并对齐版本**

```bash
cd /root
# 下载茴香豆repo
git clone https://github.com/internlm/huixiangdou && cd huixiangdou
git checkout 447c6f7e68a1657fce1c4f7c740ea1700bde0440
```

![image-20240409124952485](./assets/image-20240409124952485.png)

#### 4. 修改配置文件 

> `/root/huixiangdou/config.ini` 

```ini
# 修改用于向量数据库和词嵌入的模型
- embedding_model_path = "maidalun1020/bce-embedding-base_v1"
+ embedding_model_path = "/root/models/bce-embedding-base_v1" 

# 修改用于检索的重排序模型
- reranker_model_path = "maidalun1020/bce-reranker-base_v1"
+ reranker_model_path = "/root/models/bce-reranker-base_v1"

# 修改选用的大模型
- local_llm_path = "internlm/internlm2-chat-7b"
+ local_llm_path = "/root/models/internlm2-chat-7b"
```

![image-20240409125000381](./assets/image-20240409125000381.png)

#### 5. 创建知识库

> 使用 **InternLM** 的 **Huixiangdou** 文档作为新增知识数据检索来源

```bash
cd /root/huixiangdou && mkdir repodir
# 下载 Huixiangdou 语料
git clone https://github.com/internlm/huixiangdou --depth=1 repodir/huixiangdou
```

> 除了语料知识的向量数据库，茴香豆建立接受和拒答两个向量数据库，用来在检索的过程中更加精确的判断提问的相关性，这两个数据库的来源分别是：
>
> - 接受问题列表，希望茴香豆助手回答的示例问题
>   - 存储在 `/root/huixiangdou/resource/good_questions.json` 中
> - 拒绝问题列表，希望茴香豆助手拒答的示例问题
>   - 存储在 `/root/huixiangdou/resource/bad_questions.json` 中
>   - 其中多为技术无关的主题或闲聊
>   - 如："nihui 是谁", "具体在哪些位置进行修改？", "你是谁？", "1+1"

```bash
# 增加茴香豆相关的问题到接受问题示例中
cd /root/huixiangdou
mv resource/good_questions.json resource/good_questions_bk.json
echo '[
    "mmpose中怎么调用mmyolo接口",
    "mmpose实现姿态估计后怎么实现行为识别",
    "mmpose执行提取关键点命令不是分为两步吗，一步是目标检测，另一步是关键点提取，我现在目标检测这部分的代码是demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth   现在我想把这个mmdet的checkpoints换位yolo的，那么应该怎么操作",
    "在mmdetection中，如何同时加载两个数据集，两个dataloader",
    "如何将mmdetection2.28.2的retinanet配置文件改为单尺度的呢？",
    "1.MMPose_Tutorial.ipynb、inferencer_demo.py、image_demo.py、bottomup_demo.py、body3d_pose_lifter_demo.py这几个文件和topdown_demo_with_mmdet.py的区别是什么，\n2.我如果要使用mmdet是不是就只能使用topdown_demo_with_mmdet.py文件，",
    "mmpose 测试 map 一直是 0 怎么办？",
    "如何使用mmpose检测人体关键点？",
    "我使用的数据集是labelme标注的，我想知道mmpose的数据集都是什么样式的，全都是单目标的数据集标注，还是里边也有多目标然后进行标注",
    "如何生成openmmpose的c++推理脚本",
    "mmpose",
    "mmpose的目标检测阶段调用的模型，一定要是demo文件夹下的文件吗，有没有其他路径下的文件",
    "mmpose可以实现行为识别吗，如果要实现的话应该怎么做",
    "我在mmyolo的v0.6.0 (15/8/2023)更新日志里看到了他新增了支持基于 MMPose 的 YOLOX-Pose，我现在是不是只需要在mmpose/project/yolox-Pose内做出一些设置就可以，换掉demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py 改用mmyolo来进行目标检测了",
    "mac m1从源码安装的mmpose是x86_64的",
    "想请教一下mmpose有没有提供可以读取外接摄像头，做3d姿态并达到实时的项目呀？",
    "huixiangdou 是什么？",
    "使用科研仪器需要注意什么？",
    "huixiangdou 是什么？",
    "茴香豆 是什么？",
    "茴香豆 能部署到微信吗？",
    "茴香豆 怎么应用到飞书",
    "茴香豆 能部署到微信群吗？",
    "茴香豆 怎么应用到飞书群",
    "huixiangdou 能部署到微信吗？",
    "huixiangdou 怎么应用到飞书",
    "huixiangdou 能部署到微信群吗？",
    "huixiangdou 怎么应用到飞书群",
    "huixiangdou",
    "茴香豆",
    "茴香豆 有哪些应用场景",
    "huixiangdou 有什么用",
    "huixiangdou 的优势有哪些？",
    "茴香豆 已经应用的场景",
    "huixiangdou 已经应用的场景",
    "huixiangdou 怎么安装",
    "茴香豆 怎么安装",
    "茴香豆 最新版本是什么",
    "茴香豆 支持哪些大模型",
    "茴香豆 支持哪些通讯软件",
    "config.ini 文件怎么配置",
    "remote_llm_model 可以填哪些模型?"
]' > /root/huixiangdou/resource/good_questions.json
```

```bash
# 创建一个测试用的问询列表
cd /root/huixiangdou
echo '[
"huixiangdou 是什么？",
"你好，介绍下自己"
]' > ./test_queries.json
```

#### 6. 创建向量数据库

```bash
# 创建向量数据库存储目录
cd /root/huixiangdou && mkdir workdir 

# 分别向量化知识语料、接受问题和拒绝问题中后保存到 /root/huixiangdou/workdir
python3 -m huixiangdou.service.feature_store --sample ./test_queries.json
```

![image-20240410200339080](./assets/image-20240410200339080.png)

![image-20240409125020888](./assets/image-20240409125020888.png)

![image-20240408132452501](./assets/image-20240408132452501.png)

#### 7. 运行茴香豆

```bash
# 填入问题
sed -i '74s/.*/queries = ["huixiangdou 是什么？", "茴香豆怎么部署到微信群", "今天天气怎么样？"]/' /root/huixiangdou/huixiangdou/main.py
```

![image-20240408133012322](./assets/image-20240408133012322.png)

```bash
# 运行茴香豆
cd /root/huixiangdou/
python3 -m huixiangdou.main --standalone
```

![image-20240408134013684](./assets/image-20240408134013684.png)

![image-20240408134115474](./assets/image-20240408134115474.png)

![image-20240408134317369](./assets/image-20240408134317369.png)

Done.

------

### （三）、进阶

#### 1. 加入网络搜索

> 登录 [Serper](https://serper.dev/) ，获取API-key

<img src="./assets/image-20240408141832048.png" alt="image-20240408141832048" style="zoom:67%;" />

> 修改 `/root/huixiangdou/config.ini`

```ini
[web_search]
# check https://serper.dev/api-key to get a free API key
- x_api_key = "${YOUR-API-KEY}"
+ x_api_key = "1848********************"
```

![image-20240408141927912](./assets/image-20240408141927912.png)

#### 2. 使用远程大模型

> 修改 `/root/huixiangdou/config.ini`

```ini
enable_local = 0 # 关闭本地模型
enable_remote = 1 # 启用云端模型
```

> 修改 `remote_` 相关配置，填写 API key、模型类型等参数

![image-20240410202613402](./assets/image-20240410202613402.png)

> 修改 `huixiangdou/huixiangdou/service/llm_server_hybrid.py`
>
> 将ChatGPT的base_url强行指向`https://api.nextapi.fun/v1`接口

![image-20240410203156096](./assets/image-20240410203156096.png)

> 这里启用的远程模型，只用在问答分析和问题生成，依然需要本地嵌入、重排序模型进行特征提取。
>
> 运行效果👇

![image-20240408223447147](./assets/image-20240408223447147.png)

![image-20240408223633180](./assets/image-20240408223633180.png)

![image-20240408223726088](./assets/image-20240408223726088.png)

![image-20240408223827681](./assets/image-20240408223827681.png)

#### 3. 利用 Gradio 搭建网页 Demo

> 安装 **Gradio** 依赖

```bash
pip install gradio==4.25.0 redis==5.0.3 flask==3.0.2 lark_oapi==1.2.4
```

> 启动茴香豆对话 Demo 服务

```bash
cd /root/huixiangdou
python3 -m tests.test_query_gradio 
```

> 配置本地power shell端口映射

```powershell
ssh -p 41686 root@ssh.intern-ai.org.cn -CNg -L 7860:127.0.0.1:7860 -o StrictHostKeyChecking=no
```

![image-20240408161727009](./assets/image-20240408161727009.png)

> 本地浏览器访问http://127.0.0.1:7860 进入Gradio对话页面

![image-20240408225012855](./assets/image-20240408225012855.png)

![image-20240408230243304](./assets/image-20240408230243304.png)

Done.

