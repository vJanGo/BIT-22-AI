import json
import os

def convert_json_to_yolo(json_file, output_txt_file, class_mapping):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    yolo_labels = []

    # 遍历shapes，提取每个标注的类别和位置信息
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        shape_type = shape['shape_type']

        # 获取类别ID，"weed" 为 0，其他类别为 1
        class_id = class_mapping.get(label, 1)

        # 处理圆形标注，计算中心和半径
        if shape_type == 'circle':
            x_center = points[0][0]
            y_center = points[0][1]
            radius = ((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)**0.5
            width = height = 2 * radius



        
        
        # 将中心坐标和宽高归一化
        x_center/= image_width
        y_center /= image_height
        width = width / image_width 
        height = height / image_height
        # 创建YOLO格式的标签行
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # 将YOLO格式的标签写入txt文件
    with open(output_txt_file, 'w') as f:
        f.writelines(yolo_labels)

def convert_all_jsons_in_folder(input_folder, output_folder, class_mapping):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有JSON文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(input_folder, file_name)
            output_txt_file = os.path.join(output_folder, file_name.replace('.json', '.txt'))
            
            # 转换当前的JSON文件
            convert_json_to_yolo(json_file_path, output_txt_file, class_mapping)
            print(f"Converted {file_name} to YOLO format.")

# 示例类别映射，将 "weed" 设置为0，其他类别默认为1
class_mapping = {
    "weed": 0
}

# 设置输入JSON文件夹路径和输出TXT文件夹路径
input_folder = 'CV_project/weeddetection/train/labels'  # 替换为你的JSON文件夹路径
output_folder = 'CV_project/weeddetection/train/yolo_labels'    # 替换为输出TXT文件夹路径

# 转换整个文件夹中的所有JSON文件
convert_all_jsons_in_folder(input_folder, output_folder, class_mapping)
