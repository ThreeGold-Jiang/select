""" Main training script """

import argparse
import glob
import os
import random
import time

import numpy as np
import gc
import torch
import torch.nn
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import sys
import wandb
sys.path.append("../../src")
sys.path.append("/home/sxjiang/myproject/mistotter/Otter/")
# make sure you can properly access the otter folder
from otter_ai import OtterForConditionalGeneration,MySelector3
from otter_ai import FlamingoForConditionalGeneration
from pipeline  import get_data
from pipeline  import world_info_from_env
from pipeline  import AverageMeter, get_checkpoint, get_image_attention_mask
from transformers import AutoProcessor

import deepspeed
import time
import logging
timestamp = time.time() # 
formatted_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(timestamp)) # 将时间戳转换为格式化字符串
print(formatted_time) # 
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)
#TODO:这个设置有用吗
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.info("import packages completed")
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Try importing IdeficsForVisionText2Text, and if it's not available, define a dummy class
try:
    from transformers import IdeficsForVisionText2Text
except ImportError:
    print("IdeficsForVisionText2Text does not exist")
    IdeficsForVisionText2Text = type(None)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def train_one_epoch(args,generate_model, model, epoch, mimicit_loader, tokenizer, optimizer, lr_scheduler, device_id, accelerator, wandb):
    num_batches_per_epoch = len(mimicit_loader)
    total_training_steps = num_batches_per_epoch * args.num_epochs
    # special design for Idefics Model's prompt strategy
    # fake_token_image_exists = True if "<fake_token_around_image>" in tokenizer.special_tokens_map["additional_special_tokens"] else False
    # fake_token_image_token_id = tokenizer("<fake_token_around_image>", add_special_tokens=False)["input_ids"][-1]

    # normal prompt strategy
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_text = (
        "<|endofchunk|>" if "<|endofchunk|>" in tokenizer.special_tokens_map["additional_special_tokens"] else "<end_of_utterance>"
    )  # for different tokenizer
    endofchunk_token_id = tokenizer(endofchunk_text, add_special_tokens=False)["input_ids"][-1]
    answer_token_id = tokenizer("<answer>", add_special_tokens=False)["input_ids"][-1]
    ens_token_id = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"][-1]

    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = AverageMeter()  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()
    autocast_type = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32

    # loop through dataloader
    #THREEGOLDCHANGE:把原来的多数据集训练给改成了单数据集
    for num_steps, batch in tqdm(
        enumerate(mimicit_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch
        #### MIMIC-IT FORWARD PASS ####

        images = batch["video_embedds"].to(device_id, non_blocking=True)
        question_txt = batch["question_txt"]
        prompts=batch["prompts"]
        # answers=batch["answer_txt"].to(device_id,non_blocking=True)
        '''
        input
        '''
        input_ids=tokenizer(prompts,return_tensors="pt",padding=True)["input_ids"].to(device_id, non_blocking=True)
        lang_x_attention_mask = tokenizer(prompts,return_tensors="pt",padding=True)["attention_mask"].to(device_id, non_blocking=True)
        #TODO:attention_masks可能需要改一改
        # attention_mask = batch_mimicit["net_input"]["attention_masks"].to(device_id, non_blocking=True)
        labels=input_ids.clone()
        # labels =tokenizer(answers)["input_ids"].to(device_id,non_blocking=True) 
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        for i in range(labels.shape[0]):
            # get index of all endofchunk/media tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            media_idxs = torch.where(labels[i] == media_token_id)[0]

            # remove loss for any token the before the first <answer>
            token_idx = 0
            while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
                labels[i][token_idx] = -100
                token_idx += 1
            #这里应该是为了in-context-learning时使用的
            #TODO:in-context-instruct
            # remove loss for any token between <|endofchunk|> and <answer>, except <image>
            for endofchunk_idx in endofchunk_idxs[:-1]:
                token_idx = endofchunk_idx + 1
                while token_idx < labels.shape[1] and labels[i][token_idx] != answer_token_id:
                    if labels[i][token_idx] == media_token_id:
                        pass
                    else:
                        labels[i][token_idx] = -100
                    token_idx += 1
        #为什么这里没有变成-100
        labels[labels == answer_token_id] = -100
        labels[labels == media_token_id] = -100

        with accelerator.autocast():
            unwrapped_model = accelerator.unwrap_model(model)
            if num_steps == 0:
                # info check
                accelerator.print(f"input_ids: {input_ids.shape}")
                accelerator.print(f"images: {images.shape}")
                accelerator.print(f"labels: {labels.shape}")
                accelerator.print(f"model: {unwrapped_model.__class__.__name__}")
                # accelerator.print(f"model dtype: {unwrapped_model.dtype}")
            loss_mimicit = generate_model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=lang_x_attention_mask,
                selector=model,
                question_txt=question_txt,
                labels=labels
            )[0]
        if accelerator.mixed_precision == "fp16":
            accelerator.backward(loss_mimicit.to(device_id))
        else:
            accelerator.backward(loss_mimicit)

        if num_steps==args.stop_steps:
            break
        #### BACKWARD PASS ####
        # accelerator.backward(total_loss_sum.to(device_id))

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # step time and reset end outside of rank 0
        step_time_m.update(time.time() - end)
        end = time.time()
        logging.info(f"global_step={global_step}")
        #TODO:change the global_step
        if args.rank == 0 and global_step != 0 and (args.save_steps_interval != -1) and global_step%args.save_steps_interval==0:
            if not os.path.exists(args.external_save_dir):                
                os.makedirs(args.external_save_dir)
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_dict = {
                "steps": global_step,
                "model_state_dict": get_checkpoint(unwrapped_model),
            }
            logging.info(f"Saving checkpoint to {args.external_save_dir}/checkpoint_steps_{global_step}_{epoch}.pt")
            accelerator.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_steps_{global_step}_{epoch}.pt")
            if os.path.exists(f"{args.external_save_dir}/checkpoint_step_{global_step-args.save_steps_interval}.pt"):
                os.remove(f"{args.external_save_dir}/checkpoint_step_{global_step-args.save_steps_interval}.pt")
        if ((num_steps + 1) % args.logging_steps == 0) :
            logging.info(f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss MIMIC-IT: {loss_mimicit.item():.3f}")
        




def parse_args():
    """
    Parse the command line arguments and perform the initial setup.
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Main training script for the model")

    # Add arguments to the parser
    # TODO: Add help messages to clarify the purpose of each argument

    # Model configuration arguments
    parser.add_argument(
        "--external_save_dir",
        type=str,
        default="/home/sxjiang/myproject/mistotter/Otter/result/",
        help="set to save model to external path",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="otter-9b-selector-"+formatted_time,
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="otter",
        choices=["otter", "flamingo", "idefics"],
        help="otters or flamingo",
    )
    parser.add_argument(
        "--inst_format",
        type=str,
        default="simple",
        choices=["simple", "llama2", "idefics"],
        help="simple is for mpt/llama1, rest are in different instruction templates.",
    )
    # Prepare the arguments for different types of data sources.
    # Arguments are grouped by data types and whether the data is from past or new sources.
    # Arguments for image-text data, including multi-run conversations.
    parser.add_argument(
        "--past_mimicit_path",
        type=str,
        default="",
        help="Path to the past image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--past_images_path",
        type=str,
        default="",
        help="Path to the past images dataset (including base64 format images). Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--past_train_config_path",
        type=str,
        default="",
        help="Path to the past images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
    )

    parser.add_argument(
        "--mimicit_path",
        type=str,
        default="",
        help="Path to the new image-text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        default="",
        help="Path to the new images dataset (including base64 format images). Should be in format /path/to/xx.json",
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="",
        help="Path to the new images dataset (including current ids and related in-context ids). Should be in format /path/to/xx_train.json",
    )

    # Arguments for image-text in-context data.
    parser.add_argument(
        "--past_mimicit_ic_path",
        type=str,
        default="",
        help="Path to the past in-context image-text dataset. Should be in format /path/to/xx_instruction.json",
    )

    # Arguments for text data, including multi-run conversations.
    parser.add_argument(
        "--mimicit_text_path",
        type=str,
        default="",
        help="Path to the new text dataset (including multi-run conversations). Should be in format /path/to/xx_instruction.json",
    )

    # Arguments for video-text data.
    parser.add_argument(
        "--training_data_yaml",
        type=str,
        default="",
        help="Path to the training data yaml file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/sxjiang/dataset/AGQA_balanced/ann/mist/agqa/agqa_train.jsonl",
        help="set to save model to external path",
    )
    #THREEGOLD:命令行参数 for aqga
    parser.add_argument(
            "--answer_path",
            type=str,
            default="/home/sxjiang/dataset/AGQA_balanced/ann/mist/agqa/agqa_train_candidates.json",
            help="set to save model to external path",
        )
    parser.add_argument(
            "--video_path",
            type=str,
            default="/home/sxjiang/dataset/AGQA_balanced/Charades_v1",
            help="set to save model to external path",
        )

    # optimizer args
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--save_ckpt_each_epoch", default=True,action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=1, help="log loss every n steps")
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_num_samples", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps_interval", type=int, default=5)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="path to huggingface model or model identifier from local path or huggingface.co",
        default="luodian/OTTER-9B-DenseCaption",
    )
    parser.add_argument(
        "--trained_ckpt",
        type=str,
        help="path to trained_ckpt",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--warmup_steps", default=60, type=int)
    parser.add_argument("--stop_steps", default=1600, type=int)
    parser.add_argument("--warmup_steps_ratio", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    # YH: Training detail
    parser.add_argument("--mask_lm_head", action="store_true")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="the maximum src sequence length",
    )
    parser.add_argument("--patch-image-size", type=int, default=224)
    # this could potentially save 33GB of all model parameters for otter-9b, including the language and vision model.
    parser.add_argument("--save_hf_model", default=False, action="store_true")
    parser.add_argument("--task_name", default="", type=str, help="task name, used to decide different function to load dataset.")
    # wandb args
    #TODO:learn to use wandb
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        action="store_true",
        help="resume from checkpoint (original openflamingo pt format, not hf format)",
    )
    # TODO: remove additional data args, all args would be processed in above parser
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    # parser = add_data_args(parser)
    args = parser.parse_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    random_seed(args.seed)

    return args


def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="bf16")
    device_id = accelerator.device
    #TODO:适配不同的模型
    if args.pretrained_model_name_or_path is not None:
        accelerator.print(f"Loading pretrained model from {args.pretrained_model_name_or_path}")
        device_map = {"": device_id} if accelerator.distributed_type == "MULTI_GPU" or accelerator.distributed_type == "DEEPSPEED" else "auto"
        kwargs = {"local_files_only": args.offline, "device_map": device_map}
        if "otter" in args.model_name.lower():
            load_bit = "fp32"
            if load_bit == "fp16":
                precision = {"torch_dtype": torch.float16}
            elif load_bit == "bf16":
                precision = {"torch_dtype": torch.bfloat16}
            elif load_bit == "fp32":
                precision = {"torch_dtype": torch.float32}

            # This model version is trained on MIMIC-IT DC dataset.
            generate_model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-9B-DenseCaption", device_map="auto", **precision)
            tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]
            generate_model.text_tokenizer.padding_side = "left"
            tokenizer = generate_model.text_tokenizer
            image_processor = CLIPImageProcessor()
            model = MySelector3(args=args,vision_dim=1024,qdim=512,hdim=1024,numc=4,nump=256,numf=16,device=device_id)


    if args.trained_ckpt is not None:
        train_ckpt = torch.load(args.trained_ckpt, map_location="cpu")
        if train_ckpt.get("model_state_dict", None) is not None:
            train_ckpt = train_ckpt["model_state_dict"]
        _ = model.load_state_dict(train_ckpt, strict=False)
        print(_[1])
    #TODO:确定这里在干什么
    # if hasattr(model, "lang_encoder") and "LlamaForCausalLM" in model.lang_encoder.__class__.__name__:
    #     model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))
    accelerator.wait_for_everyone()
    args.distributed_type = accelerator.distributed_type
    random_seed(args.seed, 0)
    print(f"Start running training on rank {args.rank}.")

    mimicit_loader = get_data(args, image_processor, tokenizer, "agqa")
    #TODO:make clear wd
    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return "gated_cross_attn_layer" in x and "ff_gate" not in x and "attn_gate" not in x and "norm" not in x and "bias" not in x

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    total_training_steps = len(mimicit_loader) * args.num_epochs

    resume_from_epoch = 0
    #TODO:实现继续训练
    # check if a checkpoint exists for this run
    args.external_save_dir = os.path.join(args.external_save_dir, args.run_name) if args.external_save_dir else args.run_name
    if os.path.exists(f"{args.external_save_dir}") and args.resume_from_checkpoint is True:
        checkpoint_list = glob.glob(f"{args.external_save_dir}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.external_save_dir}.")
        else:
            resume_from_checkpoint_path = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(f"Found checkpoint {resume_from_checkpoint_path} for run {args.external_save_dir}.")

            print(f"Loading checkpoint from {resume_from_checkpoint_path}")
        checkpoint = torch.load(resume_from_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], False)
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1

    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.learning_rate)

    logging.info(f">>>>>>>>>>>>>>>\nTotal training steps: {total_training_steps}\n<<<<<<<<<<<<<<<<<<<<\n")

    args.warmup_steps = args.stop_steps * args.warmup_steps_ratio if args.warmup_steps_ratio is not None else args.warmup_steps

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    if accelerator.distributed_type == "DEEPSPEED" or accelerator.distributed_type == "MULTI_GPU":
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        model, optimizer, lr_scheduler, mimicit_loader = accelerator.prepare(model, optimizer, lr_scheduler, mimicit_loader)

    # model, optimizer, lr_scheduler, mimicit_loader = accelerator.prepare(model, optimizer, lr_scheduler, mimicit_loader)

    model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):

        train_one_epoch(
            args=args,
            model=model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            mimicit_loader=mimicit_loader,
            accelerator=accelerator,
            device_id=device_id,
            wandb=wandb,
            generate_model=generate_model
        )
        if args.save_ckpt_each_epoch:
            if args.rank == 0:
                if not os.path.exists(args.external_save_dir):
                    os.makedirs(args.external_save_dir)

                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": get_checkpoint(unwrapped_model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                }
                # checkpoint_dict = {
                #     "model_state_dict": get_checkpoint(unwrapped_model),
                # }
                print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_{epoch}.pt")
                accelerator.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_{epoch}.pt")
                if args.delete_previous_checkpoint:
                    if epoch > 0:
                        os.remove(f"{args.external_save_dir}/checkpoint_{epoch-1}.pt")
            accelerator.wait_for_everyone()
    accelerator.wait_for_everyone()
    if not os.path.exists(args.external_save_dir) and  accelerator.is_local_main_process:
        os.makedirs(args.external_save_dir)
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint_dict = get_checkpoint(model=unwrapped_model)

        trainable_params_name = [name for name, p in unwrapped_model.named_parameters() if p.requires_grad]
        for name in list(checkpoint_dict.keys()):
            if name not in trainable_params_name:
                del checkpoint_dict[name]

        accelerator.save(
            checkpoint_dict,
            f"{args.external_save_dir}/final_weights.pt",
        )
        # save the config
        # unwrapped_model.config.save_pretrained(args.external_save_dir)
    # accelerator.wait_for_everyone()


if __name__ == "__main__":
    # agqaloader=get_data("agqa")
    main()
