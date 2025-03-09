from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import *
from astrbot.api.all import *
import os
import subprocess

global flash_attention_2, quantize, input_text
input_text = ''

def is_video_file(filename):
    # 常见视频文件扩展名
    video_extensions = {
        '.avi', '.mp4', '.mov', '.wmv', '.mkv', '.flv', '.webm', '.vob',
        '.ogv', '.ogg', '.drc', '.mts', '.m2ts', '.ts', '.mxf', '.rm',
        '.asf', '.amv', '.m4v', '.mpg', '.mpeg', '.3gp', '.f4v', '.f4p',
        '.f4a', '.f4b'
    }
    
    # 提取文件的扩展名并转换为小写
    ext = '.' + filename.split('.')[-1].lower()
    
    # 检查扩展名是否在视频扩展名集合中  
    return ext in video_extensions

def quantize_model(model, device):
    from bitsandbytes import nn as bnn
    import torch
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 创建量化层实例并移动到目标设备
            quantized_layer = bnn.Linear8bitLt(
                module.in_features, 
                module.out_features,
                module.bias is not None
            ).to(device)
            
            # 确保原始模块的权重不是元张量，并复制权重和偏置
            if not module.weight.is_meta:
                quantized_layer.weight.data.copy_(module.weight.data.to(device))
                if module.bias is not None:
                    quantized_layer.bias.data.copy_(module.bias.data.to(device))
            else:
                raise ValueError("Weight tensor is a meta tensor and cannot be copied directly.")
            
            # 替换原来的模块
            if '.' in name:  # 如果是嵌套模块
                parent_name, attr_name = name.rsplit('.', 1)
                setattr(model.get_submodule(parent_name), attr_name, quantized_layer)
            else:  # 如果是顶层模块
                setattr(model, name, quantized_layer)
    return model

async def describe(file_path, file_type, if_flash_attention_2,if_quantize):
    from qwen_vl_utils import process_vision_info  # 假设这是正确的导入路径
    from modelscope import snapshot_download
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch
    
    model_dir = snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', 
                                  local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'Qwen2.5-VL-3B-Instruct'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if if_flash_attention_2:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        ).to(device)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype="auto",
            local_files_only=True,
        ).to(device)
    if if_quantize:
        model = quantize_model(model,device)
    else:
        pass
    model.eval()

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

    if file_type == "image":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": file_path},
                    {"type": "text", "text": "详细描述一下这个图片里面有什么，尽量详细到方位、人名、物品名称、形状、文字等"},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    elif file_type == "video":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": file_path, "max_pixels": 360 * 420, "fps": 1.0},
                    {"type": "text", "text": "详细描述一下这个视频里面有什么，尽量详细到方位、人名、物品名称、形状等"},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=1.0, padding=True, return_tensors="pt", **video_kwargs)

    # 将所有输入张量移动到同一设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    # 确保从inputs字典中正确获取input_ids
    input_ids = inputs.get('input_ids')
    if input_ids is None:
        raise ValueError("Expected 'input_ids' in the processed inputs but found none.")
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    string = ''.join(output_text)
    return string

@register("astrbot_plugin_image_video_understanding_Qwen2.5_VL", "xiewoc", "为本地模型提供的视频/图片理解补充，使用Qwen2.5-VL-3B-Instruct", "1.0.1", "https://github.com/xiewoc/astrbot_plugin_image_video_understanding_Qwen2.5_VL")
class astrbot_plugin_image_video_understanding_Qwen2_5_VL(Star):
    def __init__(self, context: Context,config: dict):
        super().__init__(context)
        self.config = config
        
        global flash_attention_2,quantize
        flash_attention_2 = self.config['enable_using_flash_attention_2']
        quantize = self.config['enable_using_quantize']
        
    @event_message_type(EventMessageType.PRIVATE_MESSAGE)
    async def on_message(self, event: AstrMessageEvent):
        
        #print(event.message_obj.message) # AstrBot 解析出来的消息链内容
        global flash_attention_2, quantize, input_text
        opt = None
        for item in event.message_obj.message:
            if isinstance(item, Image):#图像解析
                if event.get_platform_name() == "aiocqhttp":
                    # qq
                    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                    assert isinstance(event, AiocqhttpMessageEvent)
                    client = event.bot # 得到 client
                    payloads = {
                        "file_id": item.file,
                    }
                    ret = await client.api.call_action('get_file', **payloads) # 调用协议端  API
                    path = ret['file']
                    
                    #save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'temp',ret['file']) 
                    # 使用 curl
                    #subprocess.run(["curl", "-o", save_path, url])
                opt = await describe(path,'image',flash_attention_2,quantize)
            elif isinstance(item, Video):
                if event.get_platform_name() == "aiocqhttp":
                    # qq
                    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                    assert isinstance(event, AiocqhttpMessageEvent)
                    client = event.bot # 得到 client
                    payloads = {
                        "file_id": item.file,
                    }
                    ret = await client.api.call_action('get_file', **payloads) # 调用协议端  API
                    url = ret['url']
                    
                    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'temp','tmp_vid.mp4')  # 替换为你想保存的路径和文件名
                    # 使用 curl
                    subprocess.run(["curl", "-o", save_path, url])
                #print(item.file)
                opt = await describe(save_path,'video',flash_attention_2,quantize)
            elif isinstance(item, File):#有时会返回为文件
                if is_video_file(item.name):
                    if event.get_platform_name() == "aiocqhttp":
                        # qq
                        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                        assert isinstance(event, AiocqhttpMessageEvent)
                        client = event.bot # 得到 client
                        payloads = {
                            "file_id": item.name,
                        }
                        ret = await client.api.call_action('get_file', **payloads) # 调用 协议端  API
                        path = ret['url']
                        
                    #print(item.file)
                    opt = await describe(path,'video',flash_attention_2,quantize)
                else:
                    pass
            else:
                opt = None
        for item in event.message_obj.message:
            if opt:
                if isinstance(item, Plain):
                    input_text = item.text + '对方发了个视频/图片，内容是：' + opt
                else:
                    input_text = '对方发了个视频/图片，内容是：' + opt
            else:
                pass
        
    from astrbot.api.provider import ProviderRequest

    @filter.on_llm_request()
    async def on_call_llm(self, event: AstrMessageEvent, req: ProviderRequest): # 请注意有三个参数
        global input_text
        if input_text != '':
            req.system_prompt += input_text
        else:
            pass
        
        
