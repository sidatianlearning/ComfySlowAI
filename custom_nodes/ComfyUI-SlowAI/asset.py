import os, io, json, time, tempfile
import numpy as np
from PIL import Image
import torch
import oss2   # pip install oss2

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".avi")

# ---------- OSS 连接（从环境变量读） ----------
OSS_AK = os.environ.get("OSS_AK")
OSS_SK = os.environ.get("OSS_SK")
OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT")
OSS_BUCKET = os.environ.get("OSS_BUCKET")
_PREFIX = os.environ.get("OSS_PREFIX", "asset_library").strip("/")
_auth = oss2.Auth(OSS_AK, OSS_SK)
_bucket = oss2.Bucket(_auth, OSS_ENDPOINT, OSS_BUCKET)

def _key(rel):
    rel = rel.lstrip("/")
    return f"{_PREFIX}/{rel}" if _PREFIX else rel

def _read_meta(asset_key):
    mk = os.path.splitext(asset_key)[0] + ".json"
    try:
        if _bucket.object_exists(mk):
            return json.loads(_bucket.get_object(mk).read().decode("utf-8"))
    except Exception as e:
        print(f"[Asset] 读元数据失败: {e}")
    return {}

def _put_meta(base, meta):
    _bucket.put_object(base + ".json",
                       json.dumps(meta, ensure_ascii=False, indent=2).encode())

def _tags(s):
    return [t for t in s.replace(",", " ").split() if t]

def _load_image_tensor(key):
    """从 OSS 读一张图 → [H,W,3] float32 tensor"""
    data = _bucket.get_object(key).read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)

def _resize_to(t, size):
    """把 [H,W,3] tensor 缩放到 size×size（缩略图用），保持等比+居中裁剪"""
    pil = Image.fromarray((t.numpy() * 255).astype(np.uint8))
    pil.thumbnail((size, size))
    # padding 到正方形，方便拼 batch
    canvas = Image.new("RGB", (size, size), (30, 30, 30))
    x = (size - pil.width) // 2
    y = (size - pil.height) // 2
    canvas.paste(pil, (x, y))
    arr = np.array(canvas).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


# ============ 保存图片 ============
class AssetSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "business": ("STRING", {"default": "general"}),
            "tag":      ("STRING", {"default": ""}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("files",)          # keys → files
    FUNCTION = "save"
    CATEGORY = "SlowAI/Asset"
    OUTPUT_NODE = True

    def save(self, images, business, tag):
        date = time.strftime("%Y%m%d")
        tags = _tags(tag)
        n = images.shape[0]
        files = []
        for i, img in enumerate(images):
            arr = (img.cpu().numpy() * 255).astype(np.uint8)
            buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG")
            # 单图不加 _i 后缀，多图才加
            suffix = f"_{i}" if n > 1 else ""
            base = _key(f"{business}/{date}/{int(time.time()*1000)}{suffix}")
            _bucket.put_object(base + ".png", buf.getvalue())
            _put_meta(base, {
                "type": "image", "business": business, "tags": tags,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "key": base + ".png",
            })
            files.append(base + ".png")
            print(f"[Asset] ✅ 上传图片 {i+1}/{n}: {base}.png")

        # ComfyUI 前端弹出提示
        result = "\n".join(files)
        summary = f"[Asset] 共上传 {n} 张图片\n" + result
        print(summary)
        return {
            "ui": {"text": [summary]},   # 前端可显示
            "result": (result,),
        }


# ============ 检索（带缩略图预览） ============
class AssetList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "business":   ("STRING", {"default": ""}),
            "kind":       (["all", "image", "video"], {"default": "all"}),
            "tag_filter": ("STRING", {"default": ""}),
            "limit":      ("INT", {"default": 20, "min": 0, "max": 500}),
            "thumb_size": ("INT", {"default": 256, "min": 64, "max": 1024}),
        }}
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("files", "details", "preview")
    FUNCTION = "run"
    CATEGORY = "SlowAI/Asset"
    OUTPUT_NODE = True

    def run(self, business, kind, tag_filter, limit, thumb_size):
        prefix = _key(business) if business else _PREFIX
        want_tags = [t.lower() for t in _tags(tag_filter)]

        matched, keys = [], []
        for obj in oss2.ObjectIterator(_bucket, prefix=prefix):
            k = obj.key
            ext = os.path.splitext(k)[1].lower()
            if ext == ".json": continue
            if kind == "image" and ext not in IMAGE_EXTS: continue
            if kind == "video" and ext not in VIDEO_EXTS: continue

            m = _read_meta(k)
            mt = [t.lower() for t in m.get("tags", [])]
            if want_tags and not all(t in mt for t in want_tags): continue
            matched.append((k, m)); keys.append(k)
            if limit and len(matched) >= limit: break

        lines = [f"共 {len(matched)} 个素材\n" + "=" * 40]
        thumbs = []
        for k, m in matched:
            lines.append(
                f"[{m.get('type','?')}] {m.get('business','')} "
                f"标签:{' '.join(m.get('tags',[]))}\n"
                f"  时间:{m.get('created','')}\n"
                f"  key:{k}\n" + "-" * 40)
            # 只有图片才生成缩略图
            ext = os.path.splitext(k)[1].lower()
            if ext in IMAGE_EXTS:
                try:
                    thumbs.append(_resize_to(_load_image_tensor(k), thumb_size))
                except Exception as e:
                    print(f"[Asset] 缩略图失败 {k}: {e}")

        details = "\n".join(lines)
        print("[Asset]\n" + details)

        # 拼 batch 预览；没有图片时给一个占位黑图，避免报错
        if thumbs:
            preview = torch.stack(thumbs)
        else:
            preview = torch.zeros((1, thumb_size, thumb_size, 3))

        return {
            "ui": {"text": [details]},
            "result": ("\n".join(keys), details, preview),
        }


# ============ 加载图片（支持多行 key + 索引 / 全部） ============
class AssetLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "files": ("STRING", {"default": "", "multiline": True}),
            "index": ("INT", {"default": 0, "min": -1}),  # -1 = 全部加载成 batch
        }}
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "tags", "key")
    FUNCTION = "run"
    CATEGORY = "SlowAI/Asset"

    def run(self, files, index):
        keys = [x.strip() for x in files.splitlines() if x.strip()]
        if not keys:
            raise ValueError("[Asset] files 为空")

        if index == -1:
            # 加载全部，尺寸不一时统一到第一张的尺寸
            imgs, first = [], None
            for k in keys:
                t = _load_image_tensor(k)
                if first is None:
                    first = (t.shape[0], t.shape[1])
                if (t.shape[0], t.shape[1]) != first:
                    t = _resize_to(t, max(first))  # 简单处理，尺寸不一致就缩到方图
                imgs.append(t)
            batch = torch.stack(imgs)
            m = _read_meta(keys[0])
            print(f"[Asset] 加载 {len(keys)} 张图片(batch)")
            return (batch, " ".join(m.get("tags", [])), "\n".join(keys))
        else:
            idx = min(index, len(keys) - 1)
            key = keys[idx]
            t = _load_image_tensor(key)
            m = _read_meta(key)
            print(f"[Asset] 加载图片[{idx}]: {key}")
            return (t[None,], " ".join(m.get("tags", [])), key)