from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
import json
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("/dfs/comicai/chenyu.liu/Models/Salesforce_blip2-opt-6.7b-coco")
model = Blip2ForConditionalGeneration.from_pretrained(
    "/dfs/comicai/chenyu.liu/Models/Salesforce_blip2-opt-6.7b-coco", torch_dtype=torch.float16
)
model.to(device)

# root_path = "/nas40/chenyu.liu/fastcomposer_release_danbooru/fastcomposer-main/data/train_fastcomposer_data_336k_pre_release_danbooru_/00000"
# root_path = "/nas40/chenyu.liu/Datasets/Genshin/Albedo/orig"
root_path = "/dfs/comicai/chenyu.liu/Datasets/MGC/00000/"
captions={}
dir_or_files = os.listdir(root_path)
dir_or_files = [x for x in dir_or_files if x not in [".ipynb_checkpoints", ".DS_Store"] and '.png' in x] # 删除所有特定元素
dir_or_files = sorted(dir_or_files)
for dir_file in tqdm(dir_or_files):
    file_name=dir_file.split('.')[0]
    dir_file_path = os.path.join(root_path,dir_file)
    raw_image = Image.open(dir_file_path)
    inputs = processor(images=raw_image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    captions[file_name]= [generated_text]
    # print(dir_file, generated_text)
    
json_str = json.dumps(captions, indent=4)
with open('/dfs/comicai/chenyu.liu/Datasets/MGC/blip2_captions_MGC_1916_batch_.json', 'w') as f:
    # f.write(json_str)
    lines = json_str.split('\n')
    for line in lines:
        f.write(line)
        f.write('\n')