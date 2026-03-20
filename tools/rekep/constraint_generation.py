import base64
import os
import cv2
import json
import parse
import numpy as np
import time
from datetime import datetime
from vlm_client import request_chat_completion

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ConstraintGenerator:
    def __init__(self, config):
        self.config = config
        self.client_config = None
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')
        with open(os.path.join(self.base_dir, 'prompt_template.txt'), 'r') as f:
            self.prompt_template = f.read()

    def _build_prompt(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        prompt_text = self.prompt_template.format(instruction=instruction)
        # save prompt
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_template.format(instruction=instruction)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages, prompt_text

    def _trace_path(self):
        return os.path.join(self.task_dir, 'vlm_trace.json')

    def _write_vlm_trace(self, *, instruction, prompt_text, output):
        payload = {
            'instruction': instruction,
            'prompt_text': prompt_text,
            'response_text': output,
            'model': self.client_config['model'] if self.client_config else self.config['model'],
            'base_url': self.client_config['base_url'] if self.client_config else '',
            'api_key_env': self.client_config['api_key_env'] if self.client_config else '',
            'prompt_path': os.path.join(self.task_dir, 'prompt.txt'),
            'response_path': os.path.join(self.task_dir, 'output_raw.txt'),
        }
        trace_path = self._trace_path()
        with open(trace_path, 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return trace_path

    def _parse_and_save_constraints(self, output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        for i, line in enumerate(lines):
            if line.startswith("def "):
                start = i
                name = line.split("(")[0].split("def ")[1]
            if line.startswith("    return "):
                end = i
                functions[name] = lines[start:end+1]
        # organize them based on hierarchy in function names
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]  # last one is the constraint idx
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")
    
    def _parse_other_metadata(self, output):
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])
        # find grasp_keypoints
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}"
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        # convert into list of ints
        grasp_keypoints = grasp_keypoints['grasp_keypoints'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints = [int(x.strip()) for x in grasp_keypoints]
        data_dict['grasp_keypoints'] = grasp_keypoints
        # find release_keypoints
        release_keypoints_template = "release_keypoints = {release_keypoints}"
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        # convert into list of ints
        release_keypoints = release_keypoints['release_keypoints'].replace("[", "").replace("]", "").split(",")
        release_keypoints = [int(x.strip()) for x in release_keypoints]
        data_dict['release_keypoints'] = release_keypoints
        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate(self, img, instruction, metadata):
        """
        Args:
            img (np.ndarray): image of the scene (H, W, 3) uint8
            instruction (str): instruction for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        # create a directory for the task
        fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
        self.task_dir = os.path.join(self.base_dir, fname)
        os.makedirs(self.task_dir, exist_ok=True)
        # save query image
        image_path = os.path.join(self.task_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])
        # build prompt
        messages, prompt_text = self._build_prompt(image_path, instruction)
        start = time.time()
        print(f'[vlm] model: {self.config["model"]}')
        print(f'[vlm] instruction: {instruction}')
        print('[vlm] user_prompt_begin')
        print(prompt_text)
        print('[vlm] user_prompt_end')
        print(f'[{time.time()-start:.2f}s] Querying VLM API...', end='\r')
        response, self.client_config = request_chat_completion(messages=messages,
                                                               default_model=self.config['model'],
                                                               temperature=self.config['temperature'],
                                                               max_tokens=self.config['max_tokens'],
                                                               timeout=600)
        output = response['choices'][0]['message']['content']
        print(f'[{time.time()-start:.2f}s] Querying VLM API...Done')
        print(f'[vlm] resolved_model: {self.client_config["model"]}')
        print('[vlm] assistant_response_begin')
        print(output)
        print('[vlm] assistant_response_end')
        # save raw output
        with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
            f.write(output)
        trace_path = self._write_vlm_trace(
            instruction=instruction,
            prompt_text=prompt_text,
            output=output,
        )
        print(f'[vlm] trace_path: {trace_path}')
        # parse and save constraints
        self._parse_and_save_constraints(output, self.task_dir)
        # save metadata
        metadata.update(self._parse_other_metadata(output))
        self._save_metadata(metadata)
        return self.task_dir
