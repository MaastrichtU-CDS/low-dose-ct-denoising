import os
CUR_PATH = r'./checkpoints/20210501-1317/'
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

del_file(CUR_PATH)
print("Helloworld!")