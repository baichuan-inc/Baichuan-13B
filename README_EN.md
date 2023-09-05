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
        <b>English</b> |
        <a href="https://github.com/baichuan-inc/Baichuan-13B/blob/main/README.md">中文</a>
    <p>
</h4>
</div>

# Update
[2023.08.01] Updated weights of the aligned model [Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat), optimizing the effects in some scenarios.

# Table of Contents

- [Introduction](#Introduction)
- [Benchmark Results](#Benchmark-Results)
- [Model Details](#Model-Details)
- [Inference and Deployment](#Inference-and-Deployment)
- [Fine-tuning](#Fine-tuning)
- [Disclaimer](#Disclaimer)
- [Licenses](#Licenses)

# Introduction

Baichuan-13B is an open-source, commercially available large-scale language model developed by Baichuan Intelligent Technology following [Baichuan-7B](https://github.com/baichuan-inc/baichuan-7B), containing 13 billion parameters. It achieves the best results of the same size on both authoritative Chinese and English benchmarks. This release includes two versions: pre-training ([Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base)) and alignment ([Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)). Baichuan-13B has the following features:

  1. **Larger size, more data**: Baichuan-13B further expands the number of parameters to 13 billion based on [Baichuan-7B](https://github.com/baichuan-inc/baichuan-7B), and has trained 1.4 trillion tokens on high-quality corpora, exceeding LLaMA-13B by 40%. It is currently the model with the most training data under the open source 13B size. It supports both Chinese and English, uses ALiBi positional encoding, and has a context window length of 4096.
  2. **Pre-training and alignment models**: The pre-training model is a "base" suitable for developers, while the general users have a stronger demand for the aligned model with dialogue functions. Therefore, this open-source release also includes the aligned model (Baichuan-13B-Chat), which has strong dialogue capabilities, is ready-to-use, and can be simply deployed with a few lines of code.
  3. **More efficient inference**: To support wider user use, we are also open int8 and int4 quantized versions. Compared with the non-quantized version, they greatly lower the machine resource threshold for deployment with almost no loss of performance, and can be deployed on consumer-grade graphics cards like the Nvidia 3090.
  4. **Open source, free and available for commercial use**: Baichuan-13B is not only fully open to academic research, but developers can also use it commercially for free, just by applying for and obtaining an official commercial license via email.

# Benchmark Results

We conducted a `5-shot` evaluation on various authoritative Chinese and English benchmarks. The results are as follows:

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
> Note: We adopted the offical [Evaluation Scheme](https://github.com/hendrycks/test) from MMLU.

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
> Note: CMMLU is a comprehensive benchmark specifically designed to assess the knowledge and reasoning capabilities of language models in a Chinese context. We adopted the offical [Evaluation Scheme](https://github.com/haonan-li/CMMLU).

# Model Details

| Model Name   | Hidden dim| Layers | Attention Heads | Vocabulary | Total Params       | Training Tokens| Position Embedding                         | Max Length |
|--------------|:---------:|:------:|:---------------:|:----------:|:------------------:|:--------------:|:------------------------------------------:|:----------:|
| Baichuan-7B  | 4,096     | 32     | 32              | 64,000     | 7,000,559,616      | 1.2 Trillion   | [RoPE](https://arxiv.org/abs/2104.09864)   | 4,096      |
| Baichuan-13B | 5,120     | 40     | 40              | 64,000     | 13,264,901,120     | 1.4 Trillion   | [ALiBi](https://arxiv.org/abs/2108.12409)  | 4,096      |

# Inference and Deployment

The model weights, source code, and configuration needed for inference have been published on Hugging Face: [Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base) and [Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat). The following will use Baichuan-13B-Chat as an example to demonstrate multiple methods of inference. The program will automatically download necessary resources from Hugging Face.

Install requirements before doing inference：
```shell
pip install -r requirements.txt
```

## Python Code Inference

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from transformers.generation.utils import GenerationConfig
>>> tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-13B-Chat", use_fast=False, trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
>>> model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan-13B-Chat")
>>> messages = []
>>> messages.append({"role": "user", "content": "What is the second highest mountain in the world?"})
>>> response = model.chat(tokenizer, messages)
>>> print(response)
K2. The tallest one on earth is mount everest, and k2 which located at china-pakistan border ranks as number 2 k2 is also known by its pinyin name "chhogori"
```

> In the above code, the model loading specifies `device_map='auto'`, which will use all available GPUs. If you need to specify the device(s) to use, you can control it in a way similar to `export CUDA_VISIBLE_DEVICES=0,1` (using the 0 and 1 graphics cards).


## Command Line Tool Inference

```shell
python cli_demo.py
```

Demo result as follows:

<p align="center">
    <img src="media/cn-cli.png" width="70%"/>
</p>

## Web Demo Inference

Run the following command using streamlit. It will start a web service locally. You can access it by entering the address provided in the console into your browser.

```shell
streamlit run web_demo.py
```

Demo result as follows:

<p align="center">
    <img src="media/cn-web.gif" width="70%"/>
</p>

## Example outputs of Baichuan-13B-Chat

<details><summary><b>Content Creation</b></summary>

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

<details><summary><b>Ads Copywriting</b></summary>
  
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

<details><summary><b>Question Answering</b></summary>

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

</details>

<details><summary><b>Language Understanding</b></summary>

```
用户：
小张给领导买了一个礼物说，说“意思一下”，领导说“什么意思”，小张回答说：“没什么意思，就是随便意思意思”。请依次解读一下小张说的这几个“意思”的含义。

Baichun-13B-Chat：
1. 第一个“意思”：指的是小张送给领导的那个礼物，表示他已经买了这个礼物并准备送给他。
2. 第二个“意思”：指的是小张对领导的一种礼貌和尊重，表示自己并没有忘记他的身份和地位，在购买礼物时也考虑到了这一点。
3. 第三个“意思”：指的是小张对于这次赠送礼物的行为本身的态度，即不希望这件事情引起过多的关注或争议，只是想表达一种心意而已。
```

</details>

## Performance of Inference
Baichuan-13B uses ALiBi linear bias technology, which has a smaller computation load compared to Rotary Embedding, significantly improving the inference performance. Compared with standard LLaMA-13B, the average inference speed (tokens/s) has been empirically improved by 31.6%:

| Model       | tokens/s |
|-------------|:--------:|
| LLaMA-13B   | 19.4     |
| Baichuan-13B| 25.4     |

> Enviroment and params used in testing: GPU A100-SXM4-80G, PyTorch 2.0.0+cu117, transformers 4.29.1, batch size = 1, generation length = 2048, Precision fp16, using model Baichuan-13B-Base


## Quantization Deployment

Baichuan-13B supports int8 and int4 quantization, users only need to make a simple two-line change in the inference code to implement it. Please note, if quantization is done to save GPU memory, the original precision model should be loaded onto the CPU before starting quantization. Avoid adding parameters such as `device_map='auto'` or others that could cause the original precision model to be loaded directly onto the GPU when executing `from_pretrained`.

To use int8 quantization:
```python
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", torch_dtype=torch.float16, trust_remote_code=True)
model = model.quantize(8).cuda() 
```

Similarly, to use int4 quantization:
```python
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", torch_dtype=torch.float16, trust_remote_code=True)
model = model.quantize(4).cuda()
```

Besides, if you don't want to do quantize on the fly, we have [Baichuan-13B-Chat-int8](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat-int8) aviable for int8 quantization of the Chat version:
```python
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat-int8", torch_dtype=torch.float16, trust_remote_code=True).cuda()
```

The GPU memory usage before and after quantization is as follows:
| Precision   | GPU Mem (GB) |
|-------------|:------------:|
| bf16 / fp16 | 26.0         |
| int8        | 15.8         |
| int4        | 9.7          |

The results on various benchmarks after quantization compared to the original version are as follows:
| Model 5-shot           | C-Eval | MMLU | CMMLU |
|------------------------|:------:|:----:|:-----:|
| Baichuan-13B-Base      | 52.4   | 51.6 | 55.3  |
| Baichuan-13B-Base-int8 | 51.2   | 49.9 | 54.5  |
| Baichuan-13B-Base-int4 | 47.6   | 46.0 | 51.0  |


## CPU Deployment
Baichuan-13B supports CPU inference, but it should be emphasized that the inference speed on CPU will be very slow. Modify the model loading logic as follows:
```python
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", torch_dtype=torch.float32, trust_remote_code=True)
```
Loading the entire model approximately requires 60GB of memory.

# Fine-tuning

Developers can fine-tune both Baichuan-13B-Base and Baichuan-13B-Chat for use. Here we have tested the fine-tuning tool [LLaMA Efficient Tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning) which is compatible with our model, providing demonstrations for both `Full Params Fine-Tuning` and `LoRA Fine-Tuning`.

Before we start, developers should download the project LLaMA Efficient Tuning and [install it's requirements](https://github.com/hiyouga/LLaMA-Efficient-Tuning#getting-started).

The input data are in json files under `data` directory. Use option `--dataset` to specify. For multiple input files, seperate using `,`. The example format and field descriptions of a .json file are as follows:：
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
The .json file stores a list, each element of which is a sample. `instruction` represents the user's prompt, `input` is optional. If the developer specifies both `instruction` and `input`, they will be connected with `\n` to represent the user's prompt; `output` represents the expected model output.

Below, we provide demonstration scripts that have been successfully tested in two fine-tuning scenarios.

## Full Params Fine-tuning
We tested under 8 * Nvidia A100 80 GB + deepspeed for full params fine-tuning.

Example of script to start fine-tuning:
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

Example of deep_speed.json:
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

## LoRA Fine-tuning
We tested LoRA fine-tuning on a single Nvidia A100 80G GPU.

Example of script to start fine-tuning:
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

For more detailed usage of LLaMA Efficient Tuning, please refer to the instructions on its project homepage.

# Disclaimer

We hereby declare that our team has not developed any applications based on Baichuan-13B model, not on iOS, Android, the web, or any other platform. We strongly call on all users not to use Baichuan-13B model for any activities that harm national / social security or violate the law. Also, we ask users not to use Baichuan-13B model for Internet services that have not undergone appropriate security reviews and filings. We hope that all users can abide by this principle and ensure that the development of technology proceeds in a regulated and legal environment.

We have done our best to ensure the compliance of the data used in the model training process. However, despite our considerable efforts, there may still be some unforeseeable issues due to the complexity of the model and data. Therefore, if any problems arise due to the use of Baichuan-13B open-source model, including but not limited to data security issues, public opinion risks, or any risks and problems brought about by the model being misled, abused, spread or improperly exploited, we will not assume any responsibility.

# Licenses
The use of the source code in this repository complies with the [Apache 2.0](https://github.com/baichuan-inc/Baichuan-13B/blob/main/LICENSE) License. For community use of Baichuan-13B model, see [Community License for Baichuan-13B Model](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/resolve/main/Community%20License%20for%20Baichuan-13B%20Model.pdf). Baichuan-13B supports commercial use. If you use Baichuan-13B model or its derivatives for commercial purposes, please contact the licensor in the following way to register and apply for written authorization: Contact Email <opensource@baichuan-inc.com>.
