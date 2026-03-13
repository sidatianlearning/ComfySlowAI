import math

class ImageSizeTransformer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 1280, "min": 1, "max": 8192}),
                "target_height": ("INT", {"default": 720, "min": 1, "max": 8192}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompt", "aspect_ratio",)
    FUNCTION = "transform"
    CATEGORY = "SlowAI"

    def transform(self, image, target_width, target_height):
        # 1. 获取原图尺寸
        _, h, w, _ = image.shape
        
        # 2. 计算最接近的比例
        ratios = {
            "1:1": 1.0, "2:3": 2/3, "3:2": 3/2, "3:4": 3/4, 
            "4:3": 4/3, "4:5": 4/5, "5:4": 5/4, "9:16": 9/16, 
            "16:9": 16/9, "21:9": 21/9
        }
        target_ratio = target_width / target_height
        closest_ratio = min(ratios.keys(), key=lambda k: abs(ratios[k] - target_ratio))

        # 3. 生成中文提示词逻辑
        is_orientation_change = (w > h and target_width < target_height) or (w < h and target_width > target_height)
        
        if is_orientation_change:
            instruction = (f"将原图从 {w}x{h} 调整为 {target_width}x{target_height}。"
                           f"这是一次横竖版式转换，请在改变尺寸的同时，严格保持原图的视觉风格、主体内容和构图逻辑一致，避免画面拉伸或内容丢失。")
        else:
            instruction = (f"将原图从 {w}x{h} 调整为 {target_width}x{target_height}。"
                           f"这是一次缩放处理，请在改变尺寸的同时，确保画面清晰，并完美保留原图的风格、细节和主体内容。")

        return (instruction, closest_ratio)



# 节点注册
NODE_CLASS_MAPPINGS = {
    "ImageSizeTransformer": ImageSizeTransformer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSizeTransformer": "ImageSizeTransformer",
}