import os
import json
import requests
import time
import jwt
import re

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
                "model_name": (["glm-4", "glm-3-turbo"],),
            },
        }

    RETURN_TYPES = ("STRING",)  
    RETURN_NAMES = ("text",)  
    FUNCTION = "zhipuai_txt_api"  
    CATEGORY = "ChatGlm_Api"  


    def zhipuai_txt_api(self, prompt, model_name):
        if not self.api_key:
            raise ValueError("API key is required")
        if prompt == None:
            raise ValueError("need prompt")
        else:
            prompt_txt = 'in English'
            prompt = ''.join([prompt, prompt_txt])
            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            exp_seconds = int(9)
            client_key = str(get_zpai_api_key())
            token = generate_token(apikey_s=client_key, exp_second=exp_seconds)
            header_txt = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            data_txt = {"model": f"{model_name}", "messages": [{"role": "user", "content": f"{prompt}"}]}
            data_txt = json.dumps(data_txt)
            response = requests.post(url=url, headers=header_txt, data=data_txt)
            txt_content = response.json()
            prompt_txt = txt_content.get("choices", 'Default Value')[0]
            prompt_txt = prompt_txt.get("message", 'Default Value')
            prompt_txt = prompt_txt.get("content", 'Default Value')
            prompt = re.sub(r'\n+', '', prompt_txt)
        return (prompt,)


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
                "img_url": ("STRING", {"default": "https://", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "zhipuai_img2txt_api"
    CATEGORY = "ChatGlm_Api"

    def zhipuai_img2txt_api(self, prompt, img_url):
        if not self.api_key:
            raise ValueError("API key is required")
        if prompt == None:
            raise ValueError("need prompt")
        if img_url == None:
            raise ValueError("Needs a Internet img url")
        else:
            prompt_img = 'in English'
            prompt = ''.join([prompt, prompt_img])

            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

            exp_seconds = int(9)

            client_key = str(get_zpai_api_key())

            token = generate_token(apikey_s=client_key, exp_second=exp_seconds)
            url_img = f"{img_url}"
            header_img = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }

            data_img = {"model": "glm-4v", "messages": [{"role": "user",
                                                         "content": [{"type": "text", "text": f"{prompt}"},
                                                                     {"type": "image_url",
                                                                      "image_url": {"url": f"{img_url}"}}]}]}
            data_img = json.dumps(data_img)

            response = requests.post(url=url, headers=header_img, data=data_img)

            img_content = response.json()

            content_img = img_content.get("choices", 'Default Value')[0]
            content_img = content_img.get("message", 'Default Value')
            content_img = content_img.get("content", 'Default Value')
            prompt = re.sub(r'\n+', '', content_img)
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "ZhipuaiApi_Txt": ZhipuaiApi_Txt,
    "ZhipuaiApi_img": ZhipuaiApi_Img
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhipuaiApi_Txt": "ZhipuaiApi_Txt",
    "ZhipuaiApi_Img": "ZhipuaiApi_Img"
}
