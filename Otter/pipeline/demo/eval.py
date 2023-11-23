import json

with open("/home/sxjiang/myproject/mistotter/Otter/pipeline/demo/result_origin.json","r") as f:
    origin_data=f.readline()
    origin_data=eval(origin_data)
with open("/home/sxjiang/myproject/mistotter/Otter/pipeline/demo/result_select.json","r") as f:
    select_data=f.readline()
    select_data=eval(select_data)
origin_num=0
select_num=0
for i in range(len(select_data)):
    origin_line=origin_data[i]
    select_line=select_data[i]
    
    
    if select_line['answer'].lower() in origin_line["pred_answer"].lower():
        origin_num+=1
        print("\n")
        print(select_line["question"])
        print(f"correct answer: {select_line['answer']}")
        print("origin: "+origin_line["pred_answer"])
        print("select: "+select_line["pred_answer"])
    if select_line['answer'].lower() in select_line["pred_answer"].lower():
        select_num+=1
print(origin_num)
print(select_num)