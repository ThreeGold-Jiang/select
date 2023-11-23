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
import argparse
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)


sys.path.append("../../src")
# make sure you can properly access the otter folder
from otter_ai import OtterForConditionalGeneration,MySelector3

# Disable warnings
requests.packages.urllib3.disable_warnings()




# ------------------- 命令行参数 ---------------------------
parser = argparse.ArgumentParser(description="Main training script for the model")
parser.add_argument(
        "--out_path",
        type=str,
        default="result.json",
        help="set to save model to external path",
    )
parser.add_argument(
        "--is_select",
        type=bool,
        default=False,
        help="set to save model to external path",
    )
parser.add_argument(
        "--len_test",
        type=int,
        default=-1,
        help="set to save model to external path",
    )

args=parser.parse_args()




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
    return f"<image>User: {prompt} GPT:<answer>"


def get_response(input_data, prompt: str, model=None, image_processor=None, tensor_dtype=None,text=None) -> str:
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
        selector=selector,
        question_txt=text,
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
if args.selector==None:
    selector = MySelector3(device="cuda:0",numbeams=3)
    checkpoint = torch.load("/home/sxjiang/myproject/mistotter/Otter/result/otter-9b-selector-b8/final_weights.pt", map_location="cpu")
    print(checkpoint.keys())
    selector.load_state_dict(checkpoint, False)
    selector = selector.to("cuda:0")
    selector = selector.eval()
else:
    selector=None
# selector=None/
new_data=[]
len_test=args.len_test


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
        if i==len_test:
            break
        print(question_data[i])
with open("/home/sxjiang/dataset/AGQA_balanced/ann/mist/agqa/agqa_val_candidates.json","r") as f:
    answer_data=f.readline()
    answer_data=eval(answer_data)
    for i in range(len(new_data)):
        new_data[i]["answer_wl"]=answer_data[new_data[i]["question_id"]]
def get_message(line):
    message="This is a multi-choice question. Only output the choice without other text.\n"
    message=message+line["question"]+"\n"
    choices=", ".join(line["answer_wl"])+"\n"
    message=message+choices
    return message
# while True:
#     # video_url = input("Enter video path: ")  # Replace with the path to your video file, could be any common format.
#     video_url="/home/sxjiang/dataset/AGQA_balanced/Charades_v1/YNJ86.mp4"
#     frames_list = get_image(video_url)
#     while True:
#         # prompts_input = input("Enter prompts: ")
#         prompts_input="what is the woman doing?"

#         if prompts_input.lower() == "quit":
#             break
#         print(f"\nPrompt: {prompts_input}")
#         response = get_response(frames_list, prompts_input, model, image_processor, tensor_dtype)
#         print(f"Response: {response}")
shot_num=0
count=0
for i in range(shot_num,len(new_data)):
    t1=time.time()
    count+=1
    if count>8000+shot_num:
        break
    video_url="/home/sxjiang/dataset/AGQA_balanced/Charades_v1"+"/"+new_data[i]["video_id"]+".mp4"
    # while True:
        # video_url = input("Enter video path: ")  # Replace with the path to your video file, could be any common format.
    # prompt_frame = few_frames
    
    frames_list = get_image(video_url)
    # for j in range(len(frames_list)):
    #     prompt_frame.append(frames_list[j])
    # while True:
    prompts_input =get_message(new_data[i])
    # prompts_input = input("Enter prompts: ")

    if prompts_input.lower() == "quit":
        break

    logging.info(f"\nPrompt: {prompts_input}")
    response = get_response(frames_list, prompts_input, model, image_processor, tensor_dtype,text=new_data[i]["question"])
    logging.info("{}/8000".format(count))
    logging.info(f"Response: {response}")
    logging.info("Golden answer: {}".format(new_data[i]["answer"]))
    logging.info(f"video path:  {video_url}")
    # print("Golden answer: {}".format(new_data[i]["answer"]))
    # print(f"video path:  {video_url}")
    new_data[i]["pred_answer"]=response
    t2=time.time()
    logging.info("cost time: {}s".format(t2-t1))
out_path="./"+args.out_path
with open(out_path,"w") as f:
    f.write(str(new_data))