import json
import os
import argparse
from pathlib import Path

JSON_DIR = "/mnt/gaia/datasets/project_data/ScanQA_map/data_processed/qa_output/"
IMAGE_DIR = "/mnt/gaia/datasets/project_data/ScanQA_map/data_processed/map_output/bev/"
SENS_DIR =  '/mnt/gaia/datasets/project_data/ScanNet/scans/'
SCANQA_PATH = '/mnt/gaia/datasets/project_data/ScanQA/'

SCENE_ID_MAP_FILE = SCANQA_PATH + "scene_id_map.json"
OUTPUT_DIR = "scanqa_map.json"


def determine_split_from_scene_id(scene_id, scene_id_map):
    """根据场景ID确定数据集分割"""
    if scene_id in scene_id_map["train"]:
        return "train"
    elif scene_id in scene_id_map["val"]:
        return "val"
    elif scene_id in scene_id_map["test"]:
        return "test"
    elif scene_id in scene_id_map["test_w_obj"]:
        return "test_w_obj"
    elif scene_id in scene_id_map["test_wo_obj"]:
        return "test_wo_obj"
    else:
        return None
    
def convert_dataset(input_dir, output_file, image_dir, sens_dir):
    new_dataset = []
    
    count = 0
    with open(SCENE_ID_MAP_FILE, "r") as f:
        scene_id_map = json.load(f)
    for json_file in os.listdir(input_dir):
        if not json_file.endswith('.json'):
            continue
        
        file_name = json_file.split('.')[0] 
        scene_name = file_name.split('with_coordinates_')[-1]
        input_file = os.path.join(input_dir, json_file)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
            
        for i, item in enumerate(original_data):
            question = item.get("question", "")
            question_id = item.get("question_id", "")
            scene_id = item.get("scene_id", "")
            
            image_file = f"{scene_id}_bev_{question_id}.png"
            sens_file = f"{scene_id}/{scene_id}.sens" 
            

            image_path = os.path.join(image_dir, image_file)
            sens_path = os.path.join(sens_dir, sens_file)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue
            
            if not os.path.exists(sens_path):
                print(f"Warning: Sens file not found: {sens_path}")
                continue
            
            try:
                coordinates = item.get("answer_coordinates", [{}])[0]
                min_x = coordinates.get("min_x", 0)
                min_y = coordinates.get("min_y", 0)
                max_x = coordinates.get("max_x", 0)
                max_y = coordinates.get("max_y", 0)

                answer = item.get("answers", [""])[0]
                if not answer:
                    continue
                
                new_item = {
                    "sample_id": count, # TODO
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image><video>\nYou are given one bird view map with the camera path and specific objects tags, and an ego-view video of a house. Answer the following question and give the coordinates of the target object on the map.{question}"
                        },
                        {
                            "from": "gpt",
                            "value": f"{answer}\ncoordinates: min_x: {min_x:.2f} min_y: {min_y:.2f} max_x: {max_x:.2f} max_y: {max_y:.2f}"
                        }
                    ],
                    "image": image_file,
                    "video": sens_file,
                    "metadata": {
                        "dataset": "ScanQA_map",
                        "split": determine_split_from_scene_id(scene_id, scene_id_map),
                        "num_samples": len(original_data),
                        "question_type": "space-understanding",
                    }

                }
                
                new_dataset.append(new_item)
                count += 1
            except:
                answer = item.get("answers", [""])[0]
                if not answer:
                    continue
                
                new_item = {
                    "sample_id": count, # TODO
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image><video>\nYou are given one bird view map with the camera path and specific objects tags, and an ego-view video of a house. Answer the following question and give the coordinates of the target object on the map.{question}"
                        },
                        {
                            "from": "gpt",
                            "value": f"{answer}"
                        }
                    ],
                    "image": image_file,
                    "video": sens_file,
                    "metadata": {
                        "dataset": "ScanQA_map",
                        "split": determine_split_from_scene_id(scene_id, scene_id_map),
                        "num_samples": len(original_data),
                        "question_type": "space-understanding",
                    }

                }
                
                new_dataset.append(new_item)
                count += 1


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion completed! Processed {len(original_data)} records, generated {len(new_dataset)} new records.")
    print(f"Output file: {output_file}")

def main():
    output_dir = os.path.dirname(OUTPUT_DIR)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    convert_dataset(JSON_DIR, OUTPUT_DIR, IMAGE_DIR, SENS_DIR)

if __name__ == "__main__":
    main()