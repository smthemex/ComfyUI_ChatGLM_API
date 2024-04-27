import os
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
import torch

p = os.path.dirname(os.path.realpath(__file__))


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
                "output_language": (["English", "Original_language"],),
            },
            "optional": {"prompt_tran2english_only": ("BOOLEAN", {"default": False},), }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "zhipuai_txt_api"
    CATEGORY = "ChatGlm_Api"

    def zhipuai_txt_api(self, prompt, model_name, max_tokens, temperature,output_language,
                        prompt_tran2english_only):
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
                if prompt_tran2english_only:
                    prompt_txt = '翻译以下内容为英文'
                else:
                    if output_language == "English":
                        prompt_txt = 'Only respond to the results of the following questions in English'
                    else:
                        prompt_txt = '只回复以下问题的结果，用我使用的语言'
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

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

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

            pil_image = self.tensor_to_image(image)

            temp_directory = tempfile.gettempdir()
            unique_suffix = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
            filename = f"image{unique_suffix}.png"
            img_path = os.path.join(temp_directory, filename)

            pil_image.save(img_path)

            # img to base64
            img = cv2.imread(img_path)
            img_data = cv2.imencode('test.PNG', img)[1].tobytes()
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
            # print(content_img)
            # content_img = str(content_img).strip('"')
            prompt = content_img.replace("\n", "")
            os.remove(img_path)
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


NODE_CLASS_MAPPINGS = {
    "ZhipuaiApi_Txt": ZhipuaiApi_Txt,
    "ZhipuaiApi_img": ZhipuaiApi_Img,
    "ZhipuaiApi_Character": ZhipuaiApi_Character
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhipuaiApi_Txt": "ZhipuaiApi_Txt",
    "ZhipuaiApi_Img": "ZhipuaiApi_Img",
    "ZhipuaiApi_Character": "ZhipuaiApi_Character"
}
