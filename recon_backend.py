"""recon_backend.py — 窗口重建后端切换: lingbot-map / vggt-omega。

统一接口:
    backend = load_backend(name, model_path, device)
    c2w = backend.window_c2w(window, raw_images=raw_images)
    # window: 旧检索/lingbot 图像张量; raw_images: 同顺序、方向已校正的 PIL 原图。
    # lingbot 使用 window; vggt_omega 使用 raw_images。均返回 (N,4,4) c2w。

约定差异(实测): lingbot 的 pose_enc 解码出来就是 c2w;vggt-omega 输出 w2c,
本模块统一对外返回 c2w(2026-10-14 probes4 对照实测:w2c 解释下中位 0.025m,
c2w 解释下 22° 全错)。
"""

import numpy as np


def preprocess_omega_images(raw_images, image_resolution=512):
    """在内存中复现 Omega 官方 balanced 预处理，避免原图经过旧尺寸中转。"""
    import torch
    from PIL import Image
    from vggt_omega.utils.load_fn import (
        _balanced_target_shape, _crop_to_supported_aspect_ratio,
        _pad_images_to_common_size,
    )

    if not raw_images:
        raise ValueError("VGGT-Omega requires a nonempty list of original PIL images")
    images, shapes = [], set()
    for image in raw_images:
        if not isinstance(image, Image.Image):
            raise TypeError("VGGT-Omega raw_images must contain PIL images")
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = _crop_to_supported_aspect_ratio(image.convert("RGB"))
        width, height = image.size
        th, tw = _balanced_target_shape(height / width, image_resolution, 16)
        image = image.resize((tw, th), Image.Resampling.BICUBIC)
        # Match torchvision ToTensor's float32 division used by the official loader.
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255)
        images.append(tensor)
        shapes.add((th, tw))
    if len(shapes) > 1:
        images = _pad_images_to_common_size(images, shapes)
    return torch.stack(images)


def load_backend(name, model_path, device):
    if name == "vggt_omega":
        return VggtOmegaBackend(model_path, device)
    if name == "lingbot":
        return LingbotBackend(model_path, device)
    raise ValueError(f"未知重建后端: {name}(可选 lingbot / vggt_omega)")


class LingbotBackend:
    """现有 lingbot-map 流式窗口重建(默认, 不变)。"""
    name = "lingbot"

    def __init__(self, model_path, device):
        import torch
        from types import SimpleNamespace
        import relocalize as R
        self._R = R
        self._torch = torch
        self.device = device
        self.model = R.load_lingbot_model(
            SimpleNamespace(model_path=model_path), device)

    def window_c2w(self, window, *, raw_images=None):
        return self._R.run_window_c2w(self.model, window, self.device,
                                      self._torch.bfloat16)


class VggtOmegaBackend:
    """vggt-omega 全量注意力重建，原图独立预处理后返回 c2w。

    注意两点差异(均已在本类内处理):
      - 预处理: vggt-omega 用 balanced 模式 512/patch16, 与 lingbot 的 518/patch14 不同
      - 输出约定: vggt-omega 输出 w2c, 统一转成 c2w 再返回
    """
    name = "vggt_omega"

    def __init__(self, model_path, device):
        import torch
        from vggt_omega.models.vggt_omega import VGGTOmega
        self._torch = torch
        self.device = device
        model = VGGTOmega().eval()
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        self.model = model.to(device)
        from vggt_omega.utils.pose_enc import encoding_to_camera
        self._encoding_to_camera = encoding_to_camera

    def _preprocess(self, raw_images):
        return preprocess_omega_images(raw_images).to(self.device)

    def window_c2w(self, window, *, raw_images=None):
        torch = self._torch
        if raw_images is None or len(raw_images) != len(window):
            raise ValueError("VGGT-Omega requires one original PIL image per window frame")
        imgs = self._preprocess(raw_images)
        with torch.inference_mode():
            preds = self.model(imgs)
        ext, _ = self._encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
        w2c = np.tile(np.eye(4), (ext.shape[1], 1, 1))
        w2c[:, :3, :4] = ext[0].detach().float().cpu().numpy()
        return np.linalg.inv(w2c)                    # w2c → c2w
