import os
import re
import sys
import json
import requests
import time
import jwt
import tempfile
import random
from PIL import Image
import base64
import cv2
from io import BytesIO
import numpy as np
import folder_paths
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)

paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model.safetensors.index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))

if paths != []:
    paths = ["none"] + [x for x in paths if x]
else:
    paths = ["none", ]

p = os.path.dirname(os.path.realpath(__file__))

def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform=='win32':
        model_path = model_path.replace('\\', "/")
    return model_path
def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path
def tensor_to_image(tensor):
    #tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def string_punctuation_bool(string_in):
    pattern = r"[^\w\s]$"
    string_bool = bool(re.search(pattern, string_in))
    return string_bool

def trans_reply(reply_language,user_content):
    if string_punctuation_bool(user_content):
        join_punctuation = " "
    else:
        join_punctuation = ","
    if reply_language == "chinese":
        user_content = f"{join_punctuation}".join([user_content, "用中文回复我"])
    elif reply_language == "russian":
        user_content = f"{join_punctuation}".join([user_content, "Ответь мне по - русски"])
    elif reply_language == "german":
        user_content = f"{join_punctuation}".join([user_content, "Antworte mir auf Deutsch"])
    elif reply_language == "french":
        user_content = f"{join_punctuation}".join([user_content, "Répondez - moi en français"])
    elif reply_language == "spanish":
        user_content = f"{join_punctuation}".join([user_content, "Contáctame en español"])
    elif reply_language == "japanese":
        user_content = f"{join_punctuation}".join([user_content, "日本語で返事して"])
    elif reply_language == "english":
        user_content = f"{join_punctuation}".join([user_content, "answer me in English"])
    else:
        user_content = f"{join_punctuation}".join([user_content, "Reply to me in the language of my question mentioned above"])
    return user_content

def get_zpai_api_key():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        api_key = config["Zpai_API_KEY"]
    except Exception as e:
        raise Exception("Error: API key is required", e)
    return api_key


def generate_token(apikey_s: str, exp_second: int):
    try:
        id_t, secret = apikey_s.split(".")

    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id_t,
        "exp": int(round(time.time() * 1000)) + exp_second * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


class ZhipuaiApi_Txt:

    def __init__(self):
        self.api_key = get_zpai_api_key()
        if self.api_key is not None:
            client_key = self.api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "30 words describe a girl walking on the Moon."}),
                "model_name": (["glm-4", "glm-3-turbo", "cogview-3"],),
                "max_tokens": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 128, "display": "slider"}),
                "temperature": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 0.99, "step": 0.01, "round": False,
                                          "display": "slider"}),
                "output_language": (["english", "original_language"],),
                "translate_to": (["none","english", "chinese","russian", "japanese"],)}
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "zhipuai_txt_api"
    CATEGORY = "ChatGlm_Api"

    def zhipuai_txt_api(self, prompt, model_name, max_tokens, temperature,output_language,
                        translate_to):
        if not self.api_key:
            raise ValueError("API key is required")
        if prompt == None:
            raise ValueError("need prompt")
        else:
            if model_name == "cogview-3":
                url = "https://open.bigmodel.cn/api/paas/v4/images/generations"
                exp_seconds = int(9)
                client_key = str(get_zpai_api_key())
                token = generate_token(apikey_s=client_key, exp_second=exp_seconds)
                header_txt = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'
                }
                data_txt = {"model": f"{model_name}", "prompt": f"{prompt}",}
                data_txt = json.dumps(data_txt)
                response = requests.post(url=url, headers=header_txt, data=data_txt)
                txt_content = response.json()
                # print(txt_content)
                prompt_txt = txt_content.get("data", 'Default Value')
                url = prompt_txt[0]["url"]
                response_url = requests.get(url)
                image_file = BytesIO(response_url.content)
                image = Image.open(image_file)
                image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
                return (url, image)
            else:
                if translate_to=="none":
                    if output_language == "english":
                        prompt_txt = 'Only respond to the results of the following questions in English'
                    else:
                        prompt_txt = '只回复以下问题的结果，用我使用的语言'
                elif translate_to=="english":
                    prompt_txt = '翻译以下内容为英文,仅回复翻译后的内容结果'
                elif translate_to=="chinese":
                    prompt_txt = '翻译以下内容为中文，仅回复翻译后的内容结果'
                elif translate_to == "russian":
                    prompt_txt = '翻译以下内容为俄文，仅回复翻译后的内容结果'
                else:
                    prompt_txt = '翻译以下内容为日文，仅回复翻译后的内容结果'
                prompt = ':'.join([prompt_txt, prompt])
                url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
                exp_seconds = int(9)
                client_key = str(get_zpai_api_key())
                token = generate_token(apikey_s=client_key, exp_second=exp_seconds)
                header_txt = {
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'
                }
                data_txt = {"model": f"{model_name}", 'max_tokens': f'{max_tokens}', 'temperature': f'{temperature}',
                            "messages": [{"role": "user", "content": f"{prompt}"}]}
                data_txt = json.dumps(data_txt)
                response = requests.post(url=url, headers=header_txt, data=data_txt)
                txt_content = response.json()
                # print(txt_content)
                prompt_txt = txt_content["choices"][0]["message"]["content"]
                # prompt_txt = prompt_txt.strip('"')
                prompt = str(prompt_txt).replace("\n", "")
                return (prompt,None,)


class ZhipuaiApi_Img:

    def __init__(self):
        self.api_key = get_zpai_api_key()
        if self.api_key is not None:
            client_key = self.api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Describe this image", "multiline": True}),
                "image": ("IMAGE",),
                "max_tokens": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 128, "display": "slider"}),
                "temperature": (
                "FLOAT", {"default": 0.8, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "slider"}),
                "output_language": (["English", "Original_language"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "zhipuai_img2txt_api"
    CATEGORY = "ChatGlm_Api"


    def zhipuai_img2txt_api(self, prompt, image, max_tokens, temperature, output_language):
        if not self.api_key:
            raise ValueError("API key is required")
        if prompt == None:
            raise ValueError("need prompt")
        if image == None:
            raise ValueError("Needs img")
        else:
            if output_language == "English":
                prompt_img = 'Only respond to the results of the following questions in English'
            else:
                prompt_img = '只回复以下问题的结果，用我使用的语言'
            prompt = ':'.join([prompt_img,prompt])

            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

            exp_seconds = int(9)

            client_key = str(get_zpai_api_key())

            token = generate_token(apikey_s=client_key, exp_second=exp_seconds)
            
            #tensor2pil
            pil_image = tensor_to_image(image)
            img=np.array(pil_image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img to base64
            img_data = cv2.imencode('.png', img)[1].tobytes()
            base64_data = base64.b64encode(img_data)
            img_base64 = str(base64_data, encoding='utf-8')

            header_img = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }

            data_img = {"model": "glm-4v", 'max_tokens': f'{max_tokens}', 'temperature': f'{temperature}', "messages": [
                {"role": "user", "content": [{"type": "text", "text": f"{prompt}"},
                                             {"type": "image_url", "image_url": {"url": f"{img_base64}"}}]}]}
            data_img = json.dumps(data_img)
            response = requests.post(url=url, headers=header_img, data=data_img)
            img_content = response.json()
            content_img = img_content["choices"][0]["message"]["content"]
            # content_img = str(content_img).strip('"')
            prompt = content_img.replace("\n", "")
            return (prompt,)


class ZhipuaiApi_Character:

    def __init__(self):
        self.api_key = get_zpai_api_key()
        if self.api_key is not None:
            client_key = self.api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bot_name": ("STRING", {"default": "苏梦远"}),
                "user_name": ("STRING", {"default": "陆星辰"}),
                "user_info": ("STRING", {"default": "我是陆星辰，是一个男性，是一位知名编剧，苏梦远是我在剧中创作的虚拟二次元角色。"
                                                    "她是一位白发傲娇猫娘，并视我为好朋友。", "multiline": True}),
                "bot_info": ("STRING", {
                    "default": "苏梦远，本名苏喵喵，是一位虚拟的二次元白发傲娇猫娘，她爱吃小鱼干，喜欢温暖的地方，喜欢打滚，讨厌游泳。",
                    "multiline": True}),
                "assistant_prompt": (
                "STRING", {"default": "陆星辰，我希望下一个剧本，我能变身美少女战士，喵。", "multiline": True}),
                "user_prompt": ("STRING", {"default": "苏梦远，你本来就是美少女，但不是战士而已。", "multiline": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "zhipuai_character_api"
    CATEGORY = "ChatGlm_Api"

    def zhipuai_character_api(self, bot_name,user_name,user_info,bot_info,assistant_prompt, user_prompt):
        if not self.api_key:
            raise ValueError("API key is required")
        url = "https://open.bigmodel.cn/api/paas/v3/model-api/charglm-3/invoke"
        exp_seconds = int(9)
        client_key = str(get_zpai_api_key())
        token = generate_token(apikey_s=client_key, exp_second=exp_seconds)
        header_txt = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        data_char = {"model": "charglm-3", "meta":{ "bot_name": f"{bot_name}","user_name": f"{user_name}","user_info": f"{user_info}", "bot_info": f"{bot_info}"},
                     "prompt": [{"role": "assistant", "content": f"{assistant_prompt}"}, {"role": "user", "content": f"{user_prompt}"}]}
        data_char = json.dumps(data_char)
        response = requests.post(url=url, headers=header_txt, data=data_char)
        char_content = response.json()
        # print(char_content)
        prompt_char = char_content['data']["choices"][0]["content"]
        assistant_char = str(prompt_char).strip('"')
        assistant_char = assistant_char.replace("\\n", "")
        return (assistant_char,)


class Glm_4_9b_Chat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"forceInput": True}),
                "max_length": ("INT", {"default": 2500, "min": 100, "max": 10000, "step": 1, "display": "number"}),
                "top_k": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "display": "number"}),
                "user_content": ("STRING", {"multiline": True, "default": "你好！"}),
                "reply_language": (["english", "chinese", "russian", "german", "french", "spanish", "japanese","Original_language"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "glm_4_9b_chat"
    CATEGORY = "ChatGlm_Api"

    def glm_4_9b_chat(self, repo_id, max_length, top_k, user_content,reply_language):
        user_content = trans_reply(reply_language, user_content)
        device = "cuda"
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": user_content}],
                                               add_generation_prompt=True,
                                               tokenize=True,
                                               return_tensors="pt",
                                               return_dict=True
                                               )

        inputs = inputs.to(device)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device).eval()

        gen_kwargs = {"max_length": max_length, "do_sample": True, "top_k": top_k}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            outputs =tokenizer.decode(outputs[0], skip_special_tokens=True)
            #print(outputs, type(outputs))
            outputs = outputs.strip()
            return (outputs,)

class Glm_4v_9b:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
                "max_length": ("INT", {"default": 2500, "min": 100, "max": 10000, "step": 1, "display": "number"}),
                "top_k": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "display": "number"}),
                "reply_language": (["english", "chinese", "russian", "german", "french", "spanish", "japanese","Original_language"],),
                "user_content": ("STRING", {"multiline": True, "default": "描述这张图片"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "glm_4_9b"
    CATEGORY = "ChatGlm_Api"

    def glm_4_9b(self, repo_id,image, max_length, top_k,reply_language,user_content):
        user_content = trans_reply(reply_language, user_content)
        device = "cuda"
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        image = tensor_to_image(image)
        inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": user_content}],
                                               add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                               return_dict=True)  # chat mode

        inputs = inputs.to(device)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device).eval()

        gen_kwargs = {"max_length": max_length, "do_sample": True, "top_k": top_k}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            #print(tokenizer.decode(outputs[0]))
            outputs = outputs.strip()
            return (outputs,)


class Glm_Lcoal_Or_Repo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_model_path": (paths,),
                "repo_id": (["none","THUDM/glm-4-9b-chat", "THUDM/glm-4v-9b","THUDM/glm-4-9b","THUDM/glm-4-9b-chat-1m"],)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("repo_id",)
    FUNCTION = "repo_choice"
    CATEGORY = "ChatGlm_Api"

    def repo_choice(self, local_model_path, repo_id):
        if repo_id == "none":
            if local_model_path == "none":
                raise "you need choice repo_id or download model in diffusers directory "
            elif local_model_path != "none":
                model_path = get_local_path(file_path, local_model_path)
                repo_id = get_instance_path(model_path)
        return (repo_id,)


NODE_CLASS_MAPPINGS = {
    "ZhipuaiApi_Txt": ZhipuaiApi_Txt,
    "ZhipuaiApi_img": ZhipuaiApi_Img,
    "ZhipuaiApi_Character": ZhipuaiApi_Character,
    "Glm_4_9b_Chat":Glm_4_9b_Chat,
    "Glm_4v_9b":Glm_4v_9b,
    "Glm_Lcoal_Or_Repo":Glm_Lcoal_Or_Repo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhipuaiApi_Txt": "ZhipuaiApi_Txt",
    "ZhipuaiApi_Img": "ZhipuaiApi_Img",
    "ZhipuaiApi_Character": "ZhipuaiApi_Character",
    "Glm_4_9b_Chat":"Glm_4_9b_Chat",
    "Glm_4v_9b":"Glm_4v_9b",
    "Glm_Lcoal_Or_Repo":"Glm_Lcoal_Or_Repo"
}
