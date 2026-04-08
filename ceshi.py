import yaml

# 打开 YAML 文件
with open('data.yaml', 'r') as file:
    # 加载 YAML 数据
    data = yaml.safe_load(file)

# 打印读取的数据
print(data)