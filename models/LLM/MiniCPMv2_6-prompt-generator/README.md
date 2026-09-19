---
frameworks:
- Pytorch
license: other
tasks:
- visual-question-answering
---


## prompt-generation-V1 
This is a prompt generation model fine-tuning based on int4 quantized version of [MiniCPM-V 2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6).   The fine-tuned model is trained on a midjourney prompt dataset and is trained with 2x 4090 24GB GPUs.
this model is trained with more than 3000 samples which contain images and prompts source from Midjourney.The model can generate short prompts and long prompts for images with natural language style. It can be used for making image labels when lora training.
Running with int4 version would use lower GPU memory (about 7GB).


## Usage
Inference using Huggingface transformers on NVIDIA GPUs. Requirements tested on python 3.10：
```
Pillow==10.1.0
torch==2.1.2
torchvision==0.16.2
transformers==4.40.0
sentencepiece==0.1.99
accelerate==0.30.1
bitsandbytes==0.43.1
peft==0.9.0
```

```python
# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('pzc163/prompt-generation-V1', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('pzc163/prompt-generation-V1', trust_remote_code=True)
model.eval()

image = Image.open('xx.jpg').convert('RGB')
question = 'Provide a detailed description of the details and content contained in the image, and generate a short prompt that can be used for image generation tasks in Stable Diffusion,remind you only need respons prompt itself and no other information.'
msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(res)

## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
    stream=True
)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
```
