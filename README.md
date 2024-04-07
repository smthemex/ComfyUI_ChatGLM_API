# ComfyUI_ChatGLM_API

**You can call Chatglm's API in comfyUI to translate and describe pictures, and the API similar to OPENAI is suitable for users in Chinese Mainland。
你可以在comfyUI里调用Chatglm的api，用来翻译、描述图片等等，类似OPenAI 的API，适合中国大陆用户**

**A  使用方法：**
1、在..\ComfyUI\custom_nodes目录下，打开git，使用git clone https://github.com/smthemex/ComfyUI_ChatGLM_API.git 安装插件；
2、打开ComfyUI\custom_nodes\ComfyUI-CHATGLM4-API下的config文件，把api_key里替换成你申请到的api_key，并保存；
3、直接使用；
4、网络鉴权或许会有延时……


**B 注意事项:**
这是一个调用ChATGLM-4，GLM-3-Turbo,CHATGLM-4V的ComfyUI节点，在使用此节点之前，你需要去智谱AI的官网 https://open.bigmodel.cn，注册并申请API_key,新用户送300万token(这个活动可能已经没了)，有效期1个月。
如果是付费使用，费用如下，推荐用GLM-3-Turbo，主打便宜
GLM-4：  0.1元 / 千tokens，
GLM-4V： 0.1元 / 千tokens，
GLM-3-Turbo：  0.005元 / 千tokens
制作chatGLM4 和ChatGLM3的智谱其实做了ChatGLM_API接口的SDK(https://github.com/zhipuai/zhipuai-sdk-python)，但是迁移到ComfyUI时，我发现SDK在ComfyUI的插件节点里好像不大管用，所以这个节点不是使用智谱的SDK，而是用HTTP调用API接口。
2个节点，一个是可选ChATGLM-4，GLM-3-Turbo模型，节点为txt你可以直接输入中文，或者然GLM4帮你写提示词，其实也就是类似CHATGPT那种，只是这个是适用于中国大陆而已，另一个节点是img2txt，用的是CHATGLM-4V，类似WD14一类的图片反推，只是用CHATGLM-4V帮你反推而已。
不过，GLM-4V本来是使用提示词和图片的模式，但是官方的HTTP调用API是用url和base64编码传参，在comfyUI节点里整互联网url比较麻烦，要上传到其他服务器再获取，而base64编码一大串数据，也是占用token的，所以我在img2txt节点就偷懒了，直接要求输入“描述限定词”和“图片网络链接”，

**C 其他**
后续如果有空，会继续补全其他的模型的。


----------
注意！！本插件的代码部分，参考了-ZHO-的基于通义千问API开发的comfyUI节点（https://github.com/ZHO-ZHO-ZHO/ComfyUI-Qwen-VL-API）
