import os
import random
class_map={"daisy":0,"dandelion":1,"rose":2,"sunflower":3,"tulip":4}

root_path="/m/liyz/flower_classification/raw_data/train/"
file_list="/m/liyz/flower_classification/raw_data/train_filelist.txt"
all_sample=[]

def get_iamges_paths(root_dir_path):
    if  not os.path.exists(root_dir_path):
        raise FileExistsError
    file_list=os.listdir(root_dir_path)
    # print(root_dir_path,len(file_list))
    for f in file_list:
        file_name = os.path.join(root_dir_path,f)
        if os.path.isdir(file_name):
            get_iamges_paths(file_name)
        else:
            class_name=file_name.split("/")[-2]
            one_sample=file_name+" "+str(class_map[class_name])
            all_sample.append(one_sample)


def write_file_list(file_path,data):
    with open(file_path,"w") as f:
        for line in data:
            f.writelines(line+"\n")

if __name__ == "__main__":
    get_iamges_paths(root_path)
    random.shuffle(all_sample)
    write_file_list(file_list,all_sample)
    print("总共有训练样本：",len(all_sample))