"""recon_backend.py — 窗口重建后端切换: lingbot-map / vggt-omega。

统一接口:
    backend = load_backend(name, model_path, device)
    c2w = backend.window_c2w(window)   # window: (N,3,H,W) [0,1] torch tensor
                                       # 返回 (N,4,4) c2w numpy(窗口自身坐标系)

约定差异(实测): lingbot 的 pose_enc 解码出来就是 c2w;vggt-omega 输出 w2c,
本模块统一对外返回 c2w(2026-10-14 probes4 对照实测:w2c 解释下中位 0.025m,
c2w 解释下 22° 全错)。
"""

import numpy as np


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

    def window_c2w(self, window):
        return self._R.run_window_c2w(self.model, window, self.device,
                                      self._torch.bfloat16)


class VggtOmegaBackend:
    """vggt-omega 全量注意力重建(probes4 实测中位 0.025m/0.13°, 优于 lingbot 3.4x)。

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
        from vggt_omega.utils.load_fn import _balanced_target_shape
        self._encoding_to_camera = encoding_to_camera
        self._balanced_target_shape = _balanced_target_shape

    def _preprocess(self, window):
        """(N,3,H,W) [0,1] tensor → vggt-omega 输入(balanced 512, patch16 对齐)。"""
        from PIL import Image
        imgs = []
        for i in range(window.shape[0]):
            arr = window[i].float().cpu().numpy().transpose(1, 2, 0)
            pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
            w, h = pil.size
            th, tw = self._balanced_target_shape(h / w, 512, 16)
            pil = pil.resize((tw, th), Image.Resampling.BICUBIC)
            imgs.append(np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return self._torch.from_numpy(np.stack(imgs)).to(self.device)

    def window_c2w(self, window):
        torch = self._torch
        imgs = self._preprocess(window)
        with torch.inference_mode():
            preds = self.model(imgs)
        ext, _ = self._encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
        w2c = np.tile(np.eye(4), (ext.shape[1], 1, 1))
        w2c[:, :3, :4] = ext[0].detach().float().cpu().numpy()
        return np.linalg.inv(w2c)                    # w2c → c2w
