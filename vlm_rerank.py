"""vlm_rerank.py — VLM 图片对验证,用于首帧定位检索重排

用 VLM (默认 qwen3.8-27b-fp8,兼容 OpenAI Vision API 的端点均可) 做
"这两张图是否同一位置"的语义判断,过滤 DINOv2 检索中的视觉混淆候选。
注意:推理型模型默认关闭思考(disable_thinking=True),否则回答会被思考过程
吃光且单次调用 45s+;回答落在 reasoning 字段的情况已做兼容。

两个使用模式:
1) 远程服务模式(推荐): 调用已在服务器上运行的 Qwen3-VL 服务
2) 直接 API 模式: 调用任意兼容 OpenAI Vision API 的端点

集成方式:
    from vlm_rerank import VLMVerifier, RELOC_PROMPT
    verifier = VLMVerifier(base_url="http://<server>:10003")
    ok = verifier.verify_pair(query_img, candidate_img)  # → bool

在 eval_heldout.py 管线中的位置:
    DINOv2 检索 top-16 → VLM 顺序验证 → 过滤后候选 → LingBot-Map 窗口推理

用法:
    python vlm_rerank.py --selftest   # 自测:用 6 个已知失败案例验证 VLM 判别力
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import requests
from PIL import Image


# =============================================================================
# 室内重定位专用 Prompt
# =============================================================================
# 针对售楼处/公寓/室内走廊场景设计。与景区 prompt (ROAMII_PROMPT_V4) 的区别:
#   - 关注房间结构和独特细节,而非"景点主体"
#   - 明确"装修风格相似不代表同处"——这对售楼处同户型房间至关重要
#   - 文字/海报/标语内容不同是强否决信号

RELOC_PROMPT_SYSTEM = (
    "You are an indoor visual localization expert. "
    "Your task is to determine whether two images show the same physical location."
)

RELOC_PROMPT_USER = (
    "Do these two images show the same physical location (same room, same area)?\n\n"
    "Rules:\n"
    "1. Judge by structural layout, distinctive details, and permanent fixtures.\n"
    "2. Similar decoration style or furniture does NOT mean the same location — "
    "different rooms in the same building often share the same style.\n"
    "3. If visible text, posters, signs, or labels differ between the images, "
    "they are DEFINITELY different locations.\n"
    "4. Different viewpoints or lighting of the same location count as a match.\n\n"
    "Answer only 'Yes' or 'No'."
)


# =============================================================================
# 工具函数
# =============================================================================

def np_image_to_data_url(img: np.ndarray, max_size: int = 768) -> str:
    """numpy 图片 → base64 data URL (JPEG 压缩,节省 HTTP 传输带宽)。

    img: (H,W,3) uint8 或 (3,H,W) float32/float16 [0,1]
    max_size: 长边最大像素数
    """
    # 标准化为 (H,W,3) uint8
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)  # CHW → HWC
    if img.dtype in (np.float32, np.float16):
        img = (img * 255).clip(0, 255)
    img = img.astype(np.uint8)

    pil = Image.fromarray(img)
    # 限制尺寸,减少 VLM token 消耗
    w, h = pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def parse_yes_no(text: str) -> bool | None:
    """从 VLM 回答中解析 Yes/No 判断。

    支持: "Yes"/"No", "yes"/"no", "是"/"否", 以及其他语言变体。
    返回 None 表示无法解析(视为不确定,保守拒绝)。
    """
    lowered = text.strip().lower()
    # 优先精确匹配
    match = re.search(r"\b(yes|no|是|否)\b", lowered)
    if match:
        word = match.group(1)
        return word in ("yes", "是")

    # fallback: 找最后一次出现的位置
    yes_pos = lowered.rfind("yes")
    no_pos = lowered.rfind("no")
    if yes_pos >= 0 or no_pos >= 0:
        return yes_pos > no_pos

    shi_pos = lowered.rfind("是")
    fou_pos = lowered.rfind("否")
    if shi_pos >= 0 or fou_pos >= 0:
        return shi_pos > fou_pos

    return None


# =============================================================================
# VLM 验证器
# =============================================================================

@dataclass
class VerificationResult:
    """单次验证结果"""
    is_match: bool
    raw_response: str
    latency_ms: float


@dataclass
class VLMVerifier:
    """VLM 图片对验证器。

    调用兼容 OpenAI Vision API 的端点,发两图一问,判断是否同一位置。

    参数:
      base_url: 服务地址,如 "http://127.0.0.1:8000"
      model: 模型名,用于 /v1/chat/completions 的 model 字段
      api_key: Bearer 鉴权令牌(代理要求时填写)
      timeout: HTTP 超时秒数
      max_image_size: 图片长边最大像素
      system_prompt / user_prompt: 可自定义
    """

    base_url: str = field(default_factory=lambda: os.environ.get("VLM_URL", "http://127.0.0.1:8000"))
    model: str = field(default_factory=lambda: os.environ.get("VLM_MODEL", "qwen3.8-27b-fp8"))
    api_key: str = field(default_factory=lambda: os.environ.get("VLM_API_KEY", ""))
    timeout: float = 120.0
    max_image_size: int = 768
    max_tokens: int = 8
    temperature: float = 0.0
    # 推理型模型(如 qwen3.8-27b-fp8)必须关思考:思考会把 max_tokens 吃光导致
    # content 为空且单次 45s+(2026-09-23 实测);关闭后 0.3~0.5s 且判断正确。
    # 对非推理模型该参数被模板忽略,无害。
    disable_thinking: bool = True

    system_prompt: str = RELOC_PROMPT_SYSTEM
    user_prompt: str = RELOC_PROMPT_USER

    # 统计
    total_calls: int = field(default=0, init=False)
    total_latency_ms: float = field(default=0.0, init=False)

    def verify_pair(self, query_img: np.ndarray, candidate_img: np.ndarray) -> VerificationResult:
        """判断两张图是否同一位置。

        query_img / candidate_img: numpy 图片,支持 (H,W,3) uint8 或 (3,H,W) float32
        返回 VerificationResult.
        """
        t0 = time.perf_counter()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": np_image_to_data_url(query_img, self.max_image_size)}},
                        {"type": "image_url", "image_url": {"url": np_image_to_data_url(candidate_img, self.max_image_size)}},
                        {"type": "text", "text": self.user_prompt},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()

        # 推理型模型回答可能落在 reasoning 字段而 content 为 null,做兜底
        msg = body["choices"][0]["message"]
        raw_text = msg.get("content") or msg.get("reasoning") or ""
        is_match = parse_yes_no(raw_text) or False

        latency = (time.perf_counter() - t0) * 1000
        self.total_calls += 1
        self.total_latency_ms += latency

        return VerificationResult(is_match=is_match, raw_response=raw_text, latency_ms=latency)

    # ------------------------------------------------------------------
    # 高层 API
    # ------------------------------------------------------------------

    def rerank_candidates(
        self,
        query_img: np.ndarray,
        candidate_imgs: Sequence[np.ndarray],
        top_k: int = 8,
        early_stop: bool = True,
        verbose: bool = True,
    ) -> tuple[list[int], list[VerificationResult]]:
        """从候选图片中筛选出与查询图同一位置的帧。

        按 DINOv2 相似度顺序逐个验证,找到 top_k 个匹配或遍历完所有候选。

        参数:
          query_img: 查询帧
          candidate_imgs: 候选帧列表(按 DINOv2 检索顺序排列)
          top_k: 最多取几个匹配候选
          early_stop: True=找到第 k 个匹配就停止; False=全部验证
          verbose: 打印进度

        返回: (通过的候选索引列表, 所有验证结果列表)
        """
        passed: list[int] = []
        all_results: list[VerificationResult] = []

        for i, cand_img in enumerate(candidate_imgs):
            result = self.verify_pair(query_img, cand_img)
            all_results.append(result)

            if verbose:
                status = "✓" if result.is_match else "✗"
                print(f"  VLM [{i}] → {result.raw_response.strip()!r} {status} | {result.latency_ms:.0f}ms")

            if result.is_match:
                passed.append(i)
                if early_stop and len(passed) >= top_k:
                    if verbose:
                        print(f"  找到 {len(passed)} 个匹配,提前停止")
                    # 填充剩余结果为 None (表示未验证)
                    break

        return passed, all_results

    def filter_candidates(
        self,
        query_img: np.ndarray,
        candidate_imgs: Sequence[np.ndarray],
        verbose: bool = True,
    ) -> list[int]:
        """顺序验证,返回第一个匹配的候选索引(用于 quick-filter 模式)。

        如果全部不匹配,返回空列表。
        """
        passed, _ = self.rerank_candidates(
            query_img, candidate_imgs, top_k=1, early_stop=True, verbose=verbose
        )
        return passed


# =============================================================================
# 自测:用已知失败案例验证 VLM 判别力
# =============================================================================

# short.mp4 上当前管线的 6 个失败案例(query, 错误top1, 真邻居)
FAILURE_CASES = [
    (609, 393, 610),
    (94, 99, 95),
    (992, 995, 993),
    (836, 832, 837),
    (389, 592, 390),
    (879, 791, 880),
]


def selftest(
    npz_path: str = "output/short_stream_full.npz",
    base_url: str = "http://127.0.0.1:8000",
    model: str = "qwen3.8-27b-fp8",
):
    """用 6 个已知失败案例测试 VLM 判别能力。

    每个案例测两对:
      - query vs 错误 top-1 → 期望 No (VLM 应该能识破视觉混淆)
      - query vs 真邻居 → 期望 Yes (VLM 应该能认出同一位置)
    """
    print("=" * 60)
    print("VLM 自测: 6 个首帧定位失败案例的图片对验证")
    print(f"服务地址: {base_url}")
    print(f"模型: {model}")
    print("=" * 60)

    data = np.load(npz_path)
    imgs = data["images"][0]  # (N,3,H,W) fp16

    verifier = VLMVerifier(base_url=base_url, model=model)

    correct, total = 0, 0
    total_ms = 0.0

    for q, wrong, right in FAILURE_CASES:
        for cand, expect in [(wrong, "No"), (right, "Yes")]:
            desc = f"Q{q} vs {cand}"
            print(f"\n{desc} (期望 {expect}):")
            try:
                result = verifier.verify_pair(imgs[q], imgs[cand])
                ok = (result.is_match and expect == "Yes") or (not result.is_match and expect == "No")
                correct += int(ok)
                total += 1
                total_ms += result.latency_ms
                mark = "✓" if ok else "✗"
                print(f"  → {result.raw_response.strip()!r} {mark} | {result.latency_ms:.0f}ms")
            except Exception as e:
                print(f"  ✗ 调用失败: {e}")
                total += 1

    print(f"\n{'=' * 60}")
    print(f"合计: {correct}/{total} 正确")
    if correct == total:
        print("✓ VLM 完美判别所有失败案例 → 可以作为检索过滤器部署")
    else:
        print(f"⚠ 有 {total - correct} 个判断错误,需要分析原因")
    print(f"平均延迟: {total_ms / total:.0f}ms/对 (共 {verifier.total_calls} 次调用)")
    print(f"{'=' * 60}")
    return correct, total


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM 图片对验证自测(首帧定位)")
    parser.add_argument("--selftest", action="store_true", help="用 6 个失败案例自测")
    parser.add_argument("--npz", default="output/short_stream_full.npz", help="建图 npz 路径")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000", help="VLM 服务地址")
    parser.add_argument("--model", default="qwen3.8-27b-fp8", help="模型名")
    args = parser.parse_args()

    if args.selftest:
        selftest(npz_path=args.npz, base_url=args.base_url, model=args.model)
    else:
        parser.print_help()
