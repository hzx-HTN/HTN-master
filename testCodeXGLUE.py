import json


vul = 0
no_vul = 0
with open("E:/1_code/CodeXGLUE/function.json", "r") as f:
    content = json.load(f)
    print(type(content))
    for item in content:
        if item['target'] == 0:
            no_vul += 1
        else:
            vul += 1
    print(vul)
    print(no_vul)
    print("缺陷比：{}".format(vul/(vul+no_vul)))

