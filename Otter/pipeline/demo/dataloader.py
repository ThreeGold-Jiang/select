import argparse
import sys
sys.path.append("/home/sxjiang/myproject/mistotter/Otter/")
from pipeline import get_data,AGQADataset
from torch.utils.data.dataloader import DataLoader
import transformers
parser = argparse.ArgumentParser(description="Main training script for the model")
parser.add_argument(
        "--data_path",
        type=str,
        default="/home/sxjiang/dataset/AGQA_balanced/ann/mist/agqa/agqa_train.jsonl",
        help="set to save model to external path",
    )
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
parser.add_argument(
        "--batch_size",
        type=int,
        default="2",
        help="set to save model to external path",
    )

args=parser.parse_args()
dataset=AGQADataset(args=args)
# for index,data in enumerate(dataset):
#     print(data)
#     count=0
image_processor = transformers.CLIPImageProcessor()
print(len(dataset))
dataloader=DataLoader(dataset=dataset,batch_size=2)
dataloader=get_data(args, image_processor=image_processor, tokenizer=None, dataset_type="agqa", epoch=0)
for index,batch in enumerate(dataloader):
    print(batch)
    print(batch["video_embedds"])


