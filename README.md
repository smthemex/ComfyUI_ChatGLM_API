# ComfyUI_ChatGLM_API

You can call Chatglm's API in comfyUI to translate and describe pictures, and the API similar to OPENAI is suitable for users in Chinese Mainland.  
-----------

你可以在ComfyUI里调用Chatglm的api，用来翻译、描述图片等等，类似OPenAI 的API，适合中国大陆用户。
-----------    


**A  使用方法：**  
1、在..\ComfyUI\custom_nodes目录下，打开git，使用git clone https://github.com/smthemex/ComfyUI_ChatGLM_API.git 安装插件；  
2、打开ComfyUI\custom_nodes\ComfyUI-CHATGLM4-API下的config文件，把api_key里替换成你申请到的api_key，并保存；  
3、直接使用；  
4、网络鉴权或许会有延时……  


**B 注意事项:**
这是一个调用ChATGLM-4，GLM-3-Turbo,CHATGLM-4V的ComfyUI节点，在使用此节点之前，你需要去智谱AI的官网 https://open.bigmodel.cn/， 注册并申请API_key,新用户送300万token(这个活动可能已经没了,未调查)，有效期1个月。  
如果是付费使用，费用如下:(推荐用GLM-3-Turbo，主打便宜。)    

GLM-4：  0.1元 / 千tokens，    
GLM-4V： 0.1元 / 千tokens，    
GLM-3-Turbo：  0.005元 / 千tokens    
------

智谱其实做了ChatGLM_API接口的SDK https://github.com/zhipuai/zhipuai-sdk-python/， 但是似乎废弃了一样。    
迁移到ComfyUI时，我发现SDK在ComfyUI的节点里好像不大管用，所以这个节点未使用智谱的SDK，而是用常规的HTTP调用API接口。    
  
**C 示例：**    
   节点一: txt2txt,文本翻译,故事描述等...    
可选ChATGLM-4，GLM-3-Turbo模型，prompt你可以直接输入中文或任何其他文字，让然GLM4帮你写提示词，中国大陆用户方便点，外网用好像也可以，比较麻烦而已。    

<span style="color:#333333"><img src="https://github.com/smthemex/ComfyUI_ChatGLM_API/blob/main/workflow/txt2txt.png" width="50%"></span>

  节点二: img2txt,图片描述,图片分析等    
用的是CHATGLM-4V模型，类似WD14一类的图片反推，只是用CHATGLM-4V帮你反推,    

注意图片载入前,最好使用"图片剪裁或缩放节点",减少图片编码为Base64数据后的长度,官方只接受小于5M的图片.      
----  
<span style="color:#333333"><img src="https://github.com/smthemex/ComfyUI_ChatGLM_API/blob/main/workflow/img2txt.png" width="50%"></span>

**D 其他:**    
如果有空，会继续补全其他的模型。        


----------
致谢: 本节点参考了-ZHO-的基于"通义千问API"开发的ComfyUI节点部分代码 https://github.com/ZHO-ZHO-ZHO/ComfyUI-Qwen-VL-API/  
-----
