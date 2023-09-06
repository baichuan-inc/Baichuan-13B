import gradio as gr
#pip install gradio==3.28.3
import mdtex2html

import torch
# torch.cuda.set_device(2) 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

MODEL_PATH='./'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
print(model.generation_config)
# model = model.quantize(8).cuda()


"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

stream = True

def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    model.generation_config.temperature = temperature
    model.generation_config.top_p = top_p
    model.generation_config.max_new_tokens = max_length
    chatbot.append((parse_text(input), ""))

    history.append({"role": "user", "content": parse_text(input)})
    if stream:
        position = 0
        for response in model.chat(tokenizer, history, stream=True):
            chatbot[-1] = (parse_text(input), parse_text(response))
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            yield chatbot, history, past_key_values
        print(response)
        history.append({"role": "assistant", "content": response})
    else:
        response = model.chat(tokenizer, history)
        print(response)
        chatbot[-1] = (parse_text(input), parse_text(response))

    yield chatbot, history, past_key_values


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Baichuan-13B-Chat</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 2048, value=1024, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.85, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.9, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

demo.queue().launch(share=True, inbrowser=True,server_name="0.0.0.0", server_port=8891)
