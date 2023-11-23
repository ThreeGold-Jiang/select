import mimetypes
import os
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
import sys
import time
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
sys.path.append("../../src")
# make sure you can properly access the otter folder
from otter_ai import OtterForConditionalGeneration

# Disable warnings
requests.packages.urllib3.disable_warnings()

# ------------------- Utility Functions -------------------


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------


def extract_frames(video_path, num_frames=16):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * frame_step)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame).convert("RGB")
            frames.append(frame)

    video.release()
    return frames


def get_image(url: str) -> Union[Image.Image, list]:
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    elif "video" in content_type:
        video_path = "temp_video.mp4"
        if "://" not in url:  # Local file
            video_path = url
        else:  # Remote URL
            with open(video_path, "wb") as f:
                f.write(requests.get(url, stream=True, verify=False).content)
        frames = extract_frames(video_path)
        if "://" in url:  # Only remove the temporary video file if it was downloaded
            os.remove(video_path)
        return frames
    else:
        raise ValueError("Invalid content type. Expected image or video.")


# ------------------- OTTER Prompt and Response Functions -------------------


def get_formatted_prompt(prompt: str) -> str:
    return f"{prompt}"


def get_response(input_data, prompt: str, model=None, image_processor=None, tensor_dtype=None) -> str:
    if isinstance(input_data, Image.Image):
        vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    elif isinstance(input_data, list):  # list of video frames
        vision_x = image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"].unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt),
        ],
        return_tensors="pt",
    )

    # Get the data type from model's parameters
    model_dtype = next(model.parameters()).dtype

    # Convert tensors to the model's data type
    vision_x = vision_x.to(dtype=model_dtype)
    lang_x_input_ids = lang_x["input_ids"]
    lang_x_attention_mask = lang_x["attention_mask"]

    bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x_input_ids.to(model.device),
        attention_mask=lang_x_attention_mask.to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_id,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output


# ------------------- Main Function -------------------
load_bit = "fp32"
if load_bit == "fp16":
    precision = {"torch_dtype": torch.float16}
elif load_bit == "bf16":
    precision = {"torch_dtype": torch.bfloat16}
elif load_bit == "fp32":
    precision = {"torch_dtype": torch.float32}

# This model version is trained on MIMIC-IT DC dataset.
model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-9B-DenseCaption", device_map="auto", **precision)
tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]

model.text_tokenizer.padding_side = "left"
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()
model.eval()




new_data=[]

with open("/home/sxjiang/dataset/AGQA_balanced/ann/mist/agqa/agqa_val.jsonl","r") as f:
    question_data=f.readline()
    question_data=eval(question_data)
    for i in range(len(question_data)):
        new_line={}
        new_line["video_id"]=question_data[i]["video_id"]
        new_line["question_id"]=question_data[i]["question_id"]
        new_line["question"]=question_data[i]["question"]
        new_line["answer"]=question_data[i]["answer"]
        new_data.append(new_line)
        # print(question_data[i])
with open("/home/sxjiang/dataset/AGQA_balanced/ann/mist/agqa/agqa_val_candidates.json","r") as f:
    answer_data=f.readline()
    answer_data=eval(answer_data)
    for i in range(len(new_data)):
        new_data[i]["answer_wl"]=answer_data[new_data[i]["question_id"]]

def get_message(line):
    message="<image>User: This is a multi-choice question. Only output the choice without other text.\n"
    message=message+"User: "+line["question"]+"\n"
    message=message+"User: "
    choices=", ".join(line["answer_wl"])+"\n"
    message=message+choices+"GPT:<answer>"
    return message



few_frames=[]

def get_fewshot(shot_num=2):
    shot_message=""
    for i in range(shot_num):
        video_url="/home/sxjiang/dataset/AGQA_balanced/Charades_v1"+"/"+new_data[i]["video_id"]+".mp4"
        frames_list = get_image(video_url)
        for j in range(len(frames_list)):
            few_frames.append(frames_list[j])
        message=get_message(new_data[i])
        message=message+new_data[i]["answer"]+"<|endofchunk|>"+"\n"
        shot_message=shot_message+message
    return shot_message


shot_num=2
count=shot_num
print(len(new_data))

for i in range(shot_num,len(new_data)):
    t1=time.time()
    count+=1
    if count>8000+shot_num:
        break
    video_url="/home/sxjiang/dataset/AGQA_balanced/Charades_v1"+"/"+new_data[i]["video_id"]+".mp4"
    # while True:
        # video_url = input("Enter video path: ")  # Replace with the path to your video file, could be any common format.
    prompt_frame = few_frames
    
    frames_list = get_image(video_url)
    for j in range(len(frames_list)):
        prompt_frame.append(frames_list[j])
    # while True:
    prompts_input =get_fewshot(2)+get_message(new_data[i])
    # prompts_input = input("Enter prompts: ")

    if prompts_input.lower() == "quit":
        break

    logging.info(f"\nPrompt: {prompts_input}")
    response = get_response(frames_list, prompts_input, model, image_processor, tensor_dtype)
    logging.info("{}/8000".format(count))
    logging.info(f"Response: {response}")
    logging.info("Golden answer: {}".format(new_data[i]["answer"]))
    logging.info(f"video path:  {video_url}")
    # print("Golden answer: {}".format(new_data[i]["answer"]))
    # print(f"video path:  {video_url}")
    new_data[i]["pred_answer"]=response
    t2=time.time()
    logging.info("cost time: {}s".format(t2-t1))

with open("./result.json","w") as f:
    f.write(str(new_data))