<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  Baichuan-13B
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/baichuan-inc/Baichuan-13B-Base" target="_blank">Baichuan-13B-Base</a> 
  • 
🤗 <a href="https://huggingface.co/baichuan-inc/Baichuan-13B-Chat" target="_blank">Baichuan-13B-Chat</a> 
  • 
🤖 <a href="https://modelscope.cn/organization/baichuan-inc" target="_blank">ModelScope</a> 
  • 
💬 <a href="https://github.com/baichuan-inc/Baichuan-13B/blob/main/media/wechat.jpeg?raw=true" target="_blank">WeChat</a>
</p>

<div align="center">

[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/Baichuan-inc/baichuan-13B/blob/main/LICENSE)
<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="https://github.com/baichuan-inc/Baichuan-13B/blob/main/README_EN.md">English</a>
    <p>
</h4>
</div>

# 目录

- [介绍](#介绍)
- [Benchmark结果](#Benchmark结果)
- [模型细节](#模型细节)
- [推理和部署](#推理和部署)
- [对模型进行微调](#对模型进行微调)
- [声明](#声明)
- [协议](#协议)

  
# 介绍
Baichuan-13B 是由百川智能继 [Baichuan-7B](https://github.com/baichuan-inc/baichuan-7B) 之后开发的包含 130 亿参数的开源可商用的大规模语言模型，在权威的中文和英文 benchmark 上均取得同尺寸最好的效果。本次发布包含有预训练 ([Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base)) 和对齐 ([Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)) 两个版本。Baichuan-13B 有如下几个特点：

  1. **更大尺寸、更多数据**：Baichuan-13B 在 [Baichuan-7B](https://github.com/baichuan-inc/baichuan-7B) 的基础上进一步扩大参数量到 130 亿，并且在高质量的语料上训练了 1.4 万亿 tokens，超过 LLaMA-13B 40%，是当前开源 13B 尺寸下训练数据量最多的模型。支持中英双语，使用 ALiBi 位置编码，上下文窗口长度为 4096。
  2. **同时开源预训练和对齐模型**：预训练模型是适用开发者的『 基座 』，而广大普通用户对有对话功能的对齐模型具有更强的需求。因此本次开源我们同时发布了对齐模型（Baichuan-13B-Chat），具有很强的对话能力，开箱即用，几行代码即可简单的部署。
  3. **更高效的推理**：为了支持更广大用户的使用，我们本次同时开源了 int8 和 int4 的量化版本，相对非量化版本在几乎没有效果损失的情况下大大降低了部署的机器资源门槛，可以部署在如 Nvidia 3090 这样的消费级显卡上。
  4. **开源免费可商用**：Baichuan-13B 不仅对学术研究完全开放，开发者也仅需邮件申请并获得官方商用许可后，即可以免费商用。


# Benchmark结果

我们在各个权威大语言模型的中英文 benchmark 上进行了`5-shot`评测。结果如下：

## [C-Eval](https://cevalbenchmark.com/index.html#home)

| Model 5-shot            | STEM  | Social Sciences | Humanities | Others | Average |
|-------------------------|:-----:|:---------------:|:----------:|:------:|:-------:|
| Baichuan-7B             | 38.2  | 52.0            | 46.2       | 39.3   | 42.8    |
| Chinese-Alpaca-Plus-13B | 35.2  | 45.6            | 40.0       | 38.2   | 38.8    |
| Vicuna-13B              | 30.5  | 38.2            | 32.5       | 32.5   | 32.8    |
| Chinese-LLaMA-Plus-13B  | 30.3  | 38.0            | 32.9       | 29.1   | 32.1    |
| Ziya-LLaMA-13B-Pretrain | 27.6  | 34.4            | 32.0       | 28.6   | 30.0    |
| LLaMA-13B               | 27.0  | 33.6            | 27.7       | 27.6   | 28.5    |
| moss-moon-003-base (16B)| 27.0  | 29.1            | 27.2       | 26.9   | 27.4    |
| **Baichuan-13B-Base**   | **45.9** | **63.5** | **57.2**    | **49.3** | **52.4** |
| **Baichuan-13B-Chat**   | **43.7** | **64.6** | **56.2**    | **49.2** | **51.5** |


## [MMLU](https://arxiv.org/abs/2009.03300)

| Model 5-shot            | STEM  | Social Sciences | Humanities | Others | Average |
|-------------------------|:-----:|:---------------:|:----------:|:------:|:-------:|
| Vicuna-13B              | 40.4  | 60.5            | 49.5       | 58.4   | 52.0    |
| LLaMA-13B               | 36.1  | 53.0            | 44.0       | 52.8   | 46.3    |
| Chinese-Alpaca-Plus-13B | 36.9  | 48.9            | 40.5       | 50.5   | 43.9    |
| Ziya-LLaMA-13B-Pretrain | 35.6  | 47.6            | 40.1       | 49.4   | 42.9    |
| Baichuan-7B             | 35.6  | 48.9            | 38.4       | 48.1   | 42.3    |
| Chinese-LLaMA-Plus-13B  | 33.1  | 42.8            | 37.0       | 44.6   | 39.2    |
| moss-moon-003-base (16B)| 22.4  | 22.8            | 24.2       | 24.4   | 23.6    |
| **Baichuan-13B-Base**   | **41.6** | **60.9** | **47.4**    | **58.5** | **51.6** |
| **Baichuan-13B-Chat**   | **40.9** | **60.9** | **48.8**    | **59.0** | **52.1** |
> 说明：我们采用了 MMLU 官方的[评测方案](https://github.com/hendrycks/test)。

## [CMMLU](https://github.com/haonan-li/CMMLU)

| Model 5-shot            | STEM  | Humanities | Social Sciences | Others | China Specific | Average |
|-------------------------|:-----:|:----------:|:---------------:|:------:|:--------------:|:-------:|
| Baichuan-7B             | 34.4  | 47.5       | 47.6            | 46.6   | 44.3           | 44.0    |
| Vicuna-13B              | 31.8  | 36.2       | 37.6            | 39.5   | 34.3           | 36.3    |
| Chinese-Alpaca-Plus-13B | 29.8  | 33.4       | 33.2            | 37.9   | 32.1           | 33.4    |
| Chinese-LLaMA-Plus-13B  | 28.1  | 33.1       | 35.4            | 35.1   | 33.5           | 33.0    |
| Ziya-LLaMA-13B-Pretrain | 29.0  | 30.7       | 33.8            | 34.4   | 31.9           | 32.1    |
| LLaMA-13B               | 29.2  | 30.8       | 31.6            | 33.0   | 30.5           | 31.2    |
| moss-moon-003-base (16B)| 27.2  | 30.4       | 28.8            | 32.6   | 28.7           | 29.6    |
| **Baichuan-13B-Base**   | **41.7** | **61.1** | **59.8** | **59.0**          | **56.4** | **55.3** |
| **Baichuan-13B-Chat**   | **42.8** | **62.6** | **59.7** | **59.0**          | **56.1** | **55.8** |
> 说明：CMMLU 是一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力。我们采用了其官方的[评测方案](https://github.com/haonan-li/CMMLU)。

# 模型细节

| 模型名称      | 隐藏层维度  | 层数 | 注意力头数 |词表大小 | 总参数量        | 训练数据（tokens）| 位置编码                                     | 最大长度 |
|--------------|:---------:|:---:|:--------:|:------:|:--------------:|:---------------:|:------------------------------------------:|:-------:|
| Baichuan-7B  | 4,096     | 32  | 32       | 64,000 | 7,000,559,616  | 1.2 万亿         | [RoPE](https://arxiv.org/abs/2104.09864)   | 4,096  |
| Baichuan-13B | 5,120     | 40  | 40       | 64,000 | 13,264,901,120 | 1.4 万亿         | [ALiBi](https://arxiv.org/abs/2108.12409)  | 4,096  |

# 推理和部署

推理所需的模型权重、源码、配置已发布在 Hugging Face：[Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base) 和 [Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)。下面以 Baichuan-13B-Chat 为例示范多种推理方式。程序会自动从 Hugging Face 下载所需资源。

推理前请安装依赖：
```shell
pip install -r requirements.txt
```

## Python代码方式

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from transformers.generation.utils import GenerationConfig
>>> tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-13B-Chat", use_fast=False, trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
>>> model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan-13B-Chat")
>>> messages = []
>>> messages.append({"role": "user", "content": "世界上第二高的山峰是哪座"})
>>> response = model.chat(tokenizer, messages)
>>> print(response)
乔戈里峰。世界第二高峰———乔戈里峰西方登山者称其为k2峰，海拔高度是8611米，位于喀喇昆仑山脉的中巴边境上
```

> 在上述代码中，模型加载指定 `device_map='auto'`，会使用所有可用显卡。如需指定使用的设备，可以使用类似 `export CUDA_VISIBLE_DEVICES=0,1`（使用了0、1号显卡）的方式控制。


## 命令行工具方式

```shell
python cli_demo.py
```
最后输出示例如下：

<p align="center">
    <img src="media/cn-cli.png" width="70%"/>
</p>

## 网页 demo 方式

依靠streamlit运行以下命令，会在本地启动一个 web 服务，把控制台给出的地址放入浏览器即可访问。

```shell
streamlit run web_demo.py
```

效果如下：

<p align="center">
    <img src="media/cn-web.gif" width="70%"/>
</p>

## Baichuan-13B-Chat 示例输出

<details><summary><b>内容创作</b></summary>

```
用户：
你是一个内容创作者，编写一段关于对马斯克和比尔盖茨访问中国视角的短视频脚本

Baichun-13B-Chat：
探索马斯克与比尔·盖茨的中国之旅

镜头1：(背景音乐)画面切入，展示中国的繁华都市景象。

旁白：欢迎来到我们的节目，今天我们将带您一起了解两位科技巨头——埃隆·马斯克和比尔·盖茨在中国的一次特殊访问。

镜头2：(背景音乐)画面切换至马斯克和比尔·盖茨抵达中国的机场。

旁白：这次访问是他们首次共同踏上中国的土地，他们将在这里展开一场关于科技创新、环保和教育的对话。

镜头3：(背景音乐)画面展示马斯克和比尔·盖茨参观一家电动汽车制造厂。

旁白：在这次访问中，他们首先参观了一家电动汽车制造厂，了解了中国在新能源汽车领域的最新发展。

镜头4：(背景音乐)画面展示马斯克和比尔·盖茨参加一场关于人工智能的论坛。

旁白：随后，他们参加了关于人工智能的一场论坛，探讨了这一领域的发展趋势及其对未来的影响。

镜头5：(背景音乐)画面展示马斯克和比尔·盖茨参观一所学校。

旁白：在这次访问中，他们还参观了一所学校，了解了中国在教育领域的创新实践。

镜头6：(背景音乐)画面展示马斯克和比尔·盖茨接受媒体采访。

旁白：最后，他们在媒体采访环节表示，希望通过这次访问，能够为中美两国在科技创新、环保和教育等领域的合作提供一些启示。

镜头7：(背景音乐)画面切回繁华都市景象。

旁白：这就是马斯克和比尔·盖茨的中国之行，他们的访问不仅让我们看到了中国在科技创新方面的巨大潜力，也为我们提供了关于未来发展的更多思考。感谢收看，我们下期再见！
```

</details>

<details><summary><b>广告文案</b></summary>
  
```
用户：
请以销售员的角色，将洗衣液的干净、无残留、无异味以及除菌等优点进行推销

Baichun-13B-Chat：
尊敬的顾客，您好！我是一名洗衣液销售员，非常高兴为您介绍我们这款产品。我们的洗衣液具有以下几个显著优点：

干净：我们的洗衣液采用了先进的洗涤技术，能够深入衣物纤维，有效去除各种污渍，让您的衣物焕然一新。同时，它不会对衣物的颜色和质地造成任何损害，让您的衣物保持原有的品质。

无残留：与传统的洗衣粉相比，我们的洗衣液在清洗过程中不会产生过多的泡沫，因此清洗后的衣物更加柔软，不会有黏腻的感觉。此外，它的去污能力强，能彻底洗净衣物上的污渍，避免了传统洗衣粉容易残留在衣物上的问题。

无异味：我们的洗衣液采用天然植物提取物作为原料，不含有害化学物质，因此在使用过程中不会产生刺鼻的气味。即使是敏感肌肤的人群也可以放心使用，让您的衣物散发出淡雅的清香。

除菌：为了满足现代人对健康生活的需求，我们的洗衣液特别添加了高效除菌成分，能够有效杀灭衣物上的细菌，保护您和家人的身体健康。无论是婴幼儿的衣物还是贴身内衣，都可以放心使用。

总之，我们的洗衣液凭借其干净、无残留、无异味以及除菌等优点，成为了广大消费者的首选。如果您还没有尝试过我们的产品，那么现在就是最好的时机。相信它一定会给您带来全新的洗衣体验！
```

```
用户：
准备一份以宠物食品为主题的报告，包括潜在消费者，市场规模，市场机遇，及盈利模式等内容


Baichun-13B-Chat：

一、引言

随着人们生活水平的提高和对宠物的关爱程度不断加深，宠物食品行业得到了快速发展。本报告将对宠物食品行业的潜在消费者、市场规模、市场机遇以及盈利模式等方面进行分析，以期为相关企业提供有益的参考信息。

二、潜在消费者

1. 养宠人群：随着养宠人数的增加，宠物食品的需求也在不断增长。根据统计数据显示，2019年中国城镇养宠(犬猫)人口数量已达7355万人，同比增长4.6%。预计到2023年，这一数字将达到8742万人。

2. 新生代消费者：新生代消费者在消费观念上更加注重健康、环保和个性化，这使得他们更愿意为宠物购买高品质的食品。此外，他们也更倾向于通过网络平台了解和购买宠物食品。

三、市场规模

1. 根据国家统计局数据，2019年我国宠物食品市场规模达到了1,020亿元人民币，同比增长约10%。预计到2023年，市场规模将达到1,  500亿元人民币。

2. 从产品类型来看，宠物干粮市场占比最高，约占总市场的70%;其次是宠物湿粮，占比约为20%;宠物零食市场占比约为10%。

四、市场机遇

1. 功能性宠物食品：随着消费者对健康的关注度不断提高，具有特定功能的宠物食品如处方粮、减肥粮等需求逐渐增大。

2. 天然有机宠物食品：越来越多的消费者倾向于选择天然、无添加剂的宠物食品，这也为市场带来了新的商机。

3. 定制化宠物食品：针对不同品种、年龄、体型的宠物，推出定制化的宠物食品，满足消费者个性化需求。

五、盈利模式

1. 传统销售渠道：通过超市、宠物店、电商平台等传统渠道销售宠物食品。

2. 直销模式：通过自建网站或与第三方平台合作，直接向消费者销售宠物食品。

3. 会员制：通过设立会员制度，为会员提供专属优惠和服务，增加客户粘性。

4. 增值服务：提供宠物美容、寄养、训练等附加服务，实现多元化盈利。

六、结论

宠物食品行业具有巨大的市场潜力和发展空间。企业应抓住市场机遇，关注消费者需求变化，创新产品
```

</details>

<details><summary><b>精准问答</b></summary>

```
用户：
世界上第二高的山是什么山

Baichun-13B-Chat：
乔戈里峰(K2) 海拔8611米，位于喀喇昆仑山脉的中巴边境上，是中国以外登山者的最大考验，也是世界上最难征服的山峰之一。它的高度仅次于珠穆朗玛峰，位居世界第二。
```

```
用户：
意兴阑珊是什么意思

Baichun-13B-Chat：
"意兴阑珊"是一个汉语成语，它的意思是形容一个人兴致不高，兴趣减弱。这个词通常用来描述一个人在某个活动中失去了原有的热情和兴趣。
```

```
用户：
请问强化学习里面的Reward Shaping是什么意思

Baichun-13B-Chat：
奖励塑造(Reward Shaping)是强化学习中的一个概念，它是指通过调整环境或设计特定的奖励函数来引导智能体(Agent)学习到更符合人类期望的行为。在传统的强化学习中，智能体通过与环境交互来学习最优策略，但这种方法可能导致智能体学习到一些次优甚至不良行为。

奖励塑造的目的是为了解决这个问题，通过人为地调整奖励函数，使得智能体更容易学到我们希望它学会的行为。这可以通过以下几种方式实现：

1. 增加正向奖励：当智能体表现出期望行为时，给予其更多的奖励，以鼓励这种行为。
2. 减少负向奖励：当智能体表现出不期望行为时，减少对其进行惩罚，降低其对这种行为的偏好。
3. 引入新奖励：为智能体提供额外的奖励，以便更好地引导其学习特定行为。
4. 改变奖励分布：通过调整奖励函数，使其具有不同的奖励分布，从而影响智能体的学习过程。

总之，奖励塑造是一种通过调整环境和奖励函数来引导智能体学习更好行为的方法，有助于提高强化学习的性能。
```

</details>

<details><summary><b>语言理解</b></summary>

```
用户：
小张给领导买了一个礼物说，说“意思一下”，领导说“什么意思”，小张回答说：“没什么意思，就是随便意思意思”。请依次解读一下小张说的这几个“意思”的含义。

Baichun-13B-Chat：
1. 第一个“意思”：指的是小张送给领导的那个礼物，表示他已经买了这个礼物并准备送给他。
2. 第二个“意思”：指的是小张对领导的一种礼貌和尊重，表示自己并没有忘记他的身份和地位，在购买礼物时也考虑到了这一点。
3. 第三个“意思”：指的是小张对于这次赠送礼物的行为本身的态度，即不希望这件事情引起过多的关注或争议，只是想表达一种心意而已。
```
</details>


## 推理性能
Baichuan-13B 使用了 ALiBi 线性偏置技术，相对于 Rotary Embedding 计算量更小，对推理性能有显著提升；与标准的 LLaMA-13B 相比，平均推理速度 (tokens/s) 实测提升 31.6%：

| Model       | tokens/s |
|-------------|:--------:|
| LLaMA-13B   | 19.4     |
| Baichuan-13B| 25.4     |

> 测试环境和参数：GPU A100-SXM4-80G, PyTorch 2.0.0+cu117, transformers 4.29.1, batch size = 1, 生成长度 = 2048, 精度 fp16, 基于 Baichuan-13B-Base


## 量化部署

Baichuan-13B 支持 int8 和 int4 量化，用户只需在推理代码中简单修改两行即可实现。

### 使用量化的用户请务必注意！

***请仔细阅读接下来的的示例代码，尤其是第一行模型加载部分，和上面的推理示例是不同的。***

***开发者可以按照自己的需求修改模型的加载方式，但是请注意：如果是为了节省显存而进行量化，应加载原始精度模型到 CPU 后再开始量化；避免在`from_pretrained`时添加`device_map='auto'`或者其它会导致把原始精度模型直接加载到 GPU 的行为的参数。***

如需使用 int8 量化：
```python
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", torch_dtype=torch.float16, trust_remote_code=True)
model = model.quantize(8).cuda() 
```

同样的，如需使用 int4 量化：
```python
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", torch_dtype=torch.float16, trust_remote_code=True)
model = model.quantize(4).cuda()
```

另外，如果你不想调用 quantize 在线量化，我们有量化好的 int8 Chat 模型可供使用：[Baichuan-13B-Chat-int8](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat-int8)：
```python
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat-int8", torch_dtype=torch.float16, trust_remote_code=True).cuda()
```

量化前后占用显存情况如下：
| Precision   | GPU Mem (GB) |
|-------------|:------------:|
| bf16 / fp16 | 26.0         |
| int8        | 15.8         |
| int4        | 9.7          |

量化后在各个 benchmark 上的结果和原始版本对比如下：

| Model 5-shot           | C-Eval | MMLU | CMMLU |
|------------------------|:------:|:----:|:-----:|
| Baichuan-13B-Base      | 52.4   | 51.6 | 55.3  |
| Baichuan-13B-Base-int8 | 51.2   | 49.9 | 54.5  |
| Baichuan-13B-Base-int4 | 47.6   | 46.0 | 51.0  |


## CPU 部署
Baichuan-13B 支持 CPU 推理，但需要强调的是，CPU 的推理速度相对较慢。需按如下方式修改模型加载的方式：
```python
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", torch_dtype=torch.float32, trust_remote_code=True)
```
使用CPU进行推理大概需要 60GB 内存。

# 对模型进行微调
开发者可以对 Baichuan-13B-Base 或 Baichuan-13B-Chat 进行微调使用。在此我们测试了与 Baichuan-13B 兼容的微调工具 [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning)，并给出`全量微调`和 `LoRA微调`的两种示范。

在开始之前，开发者需下载 LLaMA Efficient Tuning 项目并按其要求[安装依赖](https://github.com/hiyouga/LLaMA-Efficient-Tuning#getting-started)。

输入数据为放置在项目`data`目录下的 json 文件，用`--dataset`选项指定（参考下面示例），多个输入文件用`,`分隔。json 文件示例格式和字段说明如下：
```
[
    {
        "instruction": "What are the three primary colors?",
        "input": "",
        "output": "The three primary colors are red, blue, and yellow."
    },
    ....
]
```
json 文件中存储一个列表，列表的每个元素是一个 sample。其中`instruction`代表用户输入，`input`是可选项，如果开发者同时指定了`instruction`和`input`，会把二者用`\n`连接起来代表用户输入；`output`代表期望的模型输出。

下面我们给出两种微调场景下测试跑通的示范脚本。

## 全量微调
我们在 8 * Nvidia A100 80 GB + deepspeed 的环境下进行了全量微调测试。

训练启动脚本示例：
```shell
deepspeed --num_gpus=8 src/train_bash.py \
    --stage sft \
    --model_name_or_path baichuan-inc/Baichuan-13B-Base \
    --do_train \
    --dataset alpaca_gpt4_en,alpaca_gpt4_zh \
    --finetuning_type full \
    --output_dir path_to_your_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \ 
    --per_device_eval_batch_size 4 \ 
    --gradient_accumulation_steps 8 \ 
    --preprocessing_num_workers 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2.0 \
    --dev_ratio 0.01 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16 \
    --deepspeed deepspeed.json
```

deep_speed.json 配置示例：
```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "initial_scale_power": 16, 
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },  
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients" : true
  }
}
```

## LoRA微调
我们在单张 Nvidia A100 80G 显卡上进行了 LoRA 微调测试。

训练启动脚本示例：

```shell
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path baichuan-inc/Baichuan-13B-Base \
    --do_train \
    --dataset alpaca_gpt4_en,alpaca_gpt4_zh \
    --finetuning_type lora \
    --lora_rank 8 \ 
    --lora_target W_pack \
    --output_dir path_to_your_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \ 
    --per_device_eval_batch_size 4 \ 
    --gradient_accumulation_steps 8 \ 
    --preprocessing_num_workers 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2.0 \
    --dev_ratio 0.01 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16
```

关于使用 LLaMA Efficient Tuning 的更详细的用法，请参阅其项目主页说明。


# 声明

我们在此声明，我们的开发团队并未基于 Baichuan-13B 模型开发任何应用，无论是在 iOS、Android、网页或任何其他平台。我们强烈呼吁所有使用者，不要利用 Baichuan-13B 模型进行任何危害国家社会安全或违法的活动。另外，我们也要求使用者不要将 Baichuan-13B 模型用于未经适当安全审查和备案的互联网服务。我们希望所有的使用者都能遵守这个原则，确保科技的发展能在规范和合法的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。因此，如果由于使用 Baichuan-13B 开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。

# 协议
对本仓库源码的使用遵循开源许可协议 [Apache 2.0](https://github.com/baichuan-inc/Baichuan-13B/blob/main/LICENSE)。对 Baichuan-13B 模型的社区使用见[《Baichuan-13B 模型社区许可协议》](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/Baichuan-13B%20%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf)。Baichuan-13B 支持商用。如果将 Baichuan-13B 模型或其衍生品用作商业用途，请您按照如下方式联系许可方，以进行登记并向许可方申请书面授权：联系邮箱 <opensource@baichuan-inc.com>。
