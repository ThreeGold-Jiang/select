import json



with open("/home/sxjiang/project/Otter/pipeline/demo/result.json","r") as f:
    data=f.read()
    data=eval(data)


acc_count=0

for i in range(8000):
    gt_answer=data[i]["answer"]
    pred_answer=data[i]["pred_answer"]
    if pred_answer.lower().find(gt_answer.lower())!=-1:
        acc_count+=1
        print("acc!")
    else:
        print("not acc!")
    print("question:{}".format(data[i]["question"]))
    print("video:{}".format(data[i]["video_id"]))
    print("gt_answer:{}".format(gt_answer))
    print("pr_answer:{}".format(data[i]["pred_answer"]))
    print("\n")

print("acc: {}".format(acc_count/8000))