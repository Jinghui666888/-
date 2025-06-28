import pickle


def inspect_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 打印数据类型和前几个样本的结构
    print(f"Loaded from {file_path}")
    print(f"Type of data: {type(data)}")

    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:5]}")
        for k in list(data.keys())[:1]:  # 打印第一个 key 的数据
            print(f"Sample key: {k}")
            print(f"Value type: {type(data[k])}")
            print(f"Sample value: {str(data[k])[:500]}")  # 防止输出过长
    elif isinstance(data, list):
        print(f"Length: {len(data)}")
        print(f"First item type: {type(data[0])}")
        print(f"First item: {str(data[0])[:500]}")
    else:
        print(f"Data: {str(data)[:500]}")


# 改成你要查看的文件路径
inspect_pkl('/Volumes/halis没有报销/dataset/twitter15/dev.pkl')
inspect_pkl('/Volumes/halis没有报销/dataset/twitter15/relations.pkl')
inspect_pkl('/Volumes/halis没有报销/dataset/twitter15/test.pkl')
inspect_pkl('/Volumes/halis没有报销/dataset/twitter15/train.pkl')