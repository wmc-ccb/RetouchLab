"""ZhaJinWenQianCake HTTP API.

Endpoints
---------
GET  /                     the editor UI
GET  /api/params           parameter metadata: groups, ranges, defaults
GET  /api/presets          named presets
POST /api/upload           multipart image -> session id + preview
POST /api/render           {session, params} -> rendered JPEG (base64)
GET  /api/faces/{session}  detected face boxes for the current image
POST /api/save             {session, params, path} -> writes a full-res file

Sessions are kept in memory with a bounded LRU, so the UI can re-render
interactively without re-uploading the original.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import string
import sys
import time
import uuid
import threading
from collections import OrderedDict
from typing import Dict, Any, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.imgutil import load_image, save_image, encode_jpeg, fit_long_edge
from engine.pipeline import RetouchPipeline, DEFAULTS, PRESETS, HSL_BANDS
from engine.face import FaceDetector
from engine import rawio
from engine.rawio import RAW_EXTENSIONS
from engine import formula as formula_engine
from engine.paths import ui_dir as _ui_dir, data_dir as _data_dir
from engine.paths import register_cuda_dlls as _register_cuda

# 早于任何 torch 使用：把 torch/lib 加进 DLL 搜索路径。
# 直接 `uvicorn server.app:app` 启动时不会经过 tools/launch.py，
# 这一行是打包版 GPU 能否生效的兜底。
try:
    _register_cuda()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
# UI_DIR：打包后 ui/ 在 exe 旁边；源码运行时在项目根。走 paths 统一解析。
UI_DIR = _ui_dir()
# 可写的状态文件统一放 data/：装到 Program Files 时 exe 目录不可写，
# 写在 exe 旁边会直接崩在"保存预设"这种小操作上。
DATA_DIR = _data_dir()
MAX_SESSIONS = 8
PREVIEW_LONG_EDGE = 1400

# ---- 版本标识 -----------------------------------------------------------
# 发布流程（tools/assemble_dist.py 与 README「发布」章节）以这里为准。
# VERSION 每次发版手动改；BUILD 优先级：环境变量 > exe 旁的 build-stamp.txt
# （打包时写入，形如 2026-10-24.1）> dev（源码运行）。
APP_VERSION = "1.1.0"


def _read_build() -> str:
    if os.environ.get("RETOUCHLAB_BUILD"):
        return os.environ["RETOUCHLAB_BUILD"]
    try:
        # build-stamp.txt 由 assemble_dist 写在 exe 旁（源码运行时在
        # server/ 旁边），不向上跳：assemble 放的就是同级目录。
        base = (os.path.dirname(sys.executable)
                if getattr(sys, "frozen", False)
                else os.path.dirname(os.path.dirname(__file__)))
        stamp = os.path.join(base, "build-stamp.txt")
        if os.path.isfile(stamp):
            with open(stamp, encoding="utf-8") as f:
                return f.read().strip()[:40]
    except Exception:
        pass
    return "dev"


APP_BUILD = _read_build()

app = FastAPI(title="ZhaJinWenQianCake", version=APP_VERSION)

# ---- 访问口令（公网隧道必须开）
# 环境变量 RETOUCHLAB_PASSWD 设置后，所有请求都要带凭证
# （用户名随意，密码=该值）。局域网使用不设此变量即无认证。
# 401 带 WWW-Authenticate 让浏览器弹原生登录框，手机/平板同样适用。
_PASSWD = os.environ.get("RETOUCHLAB_PASSWD", "").strip()


_PASSWD_HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>炸金文谦蛋糕 · 需要访问口令</title>
<style>body{margin:0;background:#111;color:#ddd;font:15px/1.7 system-ui,"Microsoft YaHei";display:flex;align-items:center;justify-content:center;min-height:100vh}.card{max-width:460px;padding:28px 30px;background:#1b1b1b;border:1px solid #2c2c2c;border-radius:12px}h1{font-size:18px;margin:0 0 14px;color:#e8b341}p{margin:8px 0}code{background:#000;padding:1px 6px;border-radius:4px;color:#7fd1a8}.tip{color:#8b8b8b;font-size:13px;margin-top:16px;border-top:1px solid #2c2c2c;padding-top:14px}</style>
</head><body><div class="card"><h1>需要访问口令</h1>
<p>这个地址开启了访问保护。浏览器通常会弹出登录框：</p>
<p>• 用户名：<code>随便填</code><br>• 口令：<code>向管理员索取</code></p>
<p class="tip">如果没弹登录框（部分手机浏览器或微信内置浏览器会这样）：
先在地址栏用 <code>用户名:口令@</code> 前缀打开，或换 Chrome / Safari 试试。<br>
另外确认协议是 <code>http://</code> 不是 <code>https://</code>——本服务没有开 TLS。</p>
</div></body></html>"""


@app.middleware("http")
async def _passwd_gate(request, call_next):
    if _PASSWD:
        # 三种认证券，任一通过即可：
        #   1) Cookie rl_auth —— 用 ?k=口令 打开一次后自动写入，手机最省事
        #   2) Basic 认证 —— 桌面浏览器原生登录框
        #   3) ?k=口令 —— 微信/App 内置浏览器不弹登录框，用这个进
        # 本机回环免密：同机浏览器/进程就是本人；公网与局域网设备
        # 的源 IP 不是回环，仍然必须带凭证。
        client = getattr(request, "client", None)
        host = getattr(client, "host", "") if client else ""
        if host in ("127.0.0.1", "::1", "localhost"):
            return await call_next(request)

        ticket = hashlib.sha256(
            ("retouchlab-v1:" + _PASSWD).encode()).hexdigest()[:32]
        ok = False
        # 1) cookie
        if request.cookies.get("rl_auth", "") == ticket:
            ok = True
        # 2) Basic
        auth = request.headers.get("authorization", "")
        if not ok and auth.startswith("Basic "):
            try:
                userpwd = base64.b64decode(auth[6:]).decode("utf-8")
                ok = userpwd.split(":", 1)[1] == _PASSWD
            except Exception:
                ok = False
        # 3) ?k= 口令 -> 发 cookie 并重定向到干净 URL
        if not ok and request.query_params.get("k", "") == _PASSWD:
            from starlette.responses import RedirectResponse

            rest = str(request.url.replace(query="", fragment=""))
            keep = [(a, b) for a, b in request.query_params.multi_items()
                    if a != "k"]
            if keep:
                from urllib.parse import urlencode

                rest = rest + "?" + urlencode(keep)
            resp = RedirectResponse(rest, status_code=302)
            resp.set_cookie("rl_auth", ticket, max_age=30 * 86400,
                            httponly=True, samesite="lax")
            return resp

        if not ok:
            # 浏览器导航请求返回提示页（裸 JSON 在手机上看起来像
            # "报错"）；API 调用仍回 JSON，程序好判断。
            headers = {"WWW-Authenticate": 'Basic realm="ZhaJinWenQianCake"'}
            wants_html = (
                "text/html" in request.headers.get("accept", "")
                or request.url.path == "/")
            if wants_html:
                from starlette.responses import HTMLResponse

                return HTMLResponse(_PASSWD_HTML, status_code=401,
                                    headers=headers)
            return JSONResponse({"detail": "需要访问口令"},
                                status_code=401, headers=headers)
    return await call_next(request)


@app.exception_handler(Exception)
async def _unhandled(request, exc):
    """未捕获异常：把完整堆栈写进 out/error.log，再回 500。

    打包发行版里用户看不到终端输出，如果没有这份日志，"接口 500"就是
    一个无法诊断的黑洞。这里同时把请求路径和堆栈落盘。
    """
    import traceback

    try:
        from engine.paths import out_dir

        os.makedirs(out_dir(), exist_ok=True)
        path = os.path.join(out_dir(), "error.log")
        # 轮转：长跑的实例（联机拍摄一挂几天）会让日志无限涨，
        # /api/errors 要全文解析，超大会拖慢排障端点。超 2MB 砍掉前半。
        try:
            if os.path.exists(path) and os.path.getsize(path) > 2 * 1024 * 1024:
                with open(path, "rb") as f:
                    tail = f.read()[1024 * 1024:]
                cut = tail.find(b"\n==== ")
                with open(path, "wb") as f:
                    f.write(tail[cut + 1:] if cut >= 0 else tail)
        except Exception:
            pass
        with open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write("\n==== %s  %s %s\n"
                    % (time.strftime("%Y-%m-%d %H:%M:%S"),
                       request.method, request.url.path))
            # 校验类异常会把整个请求体（可能是几 MB 的图片二进制）嵌进
            # traceback，写日志时既爆 UnicodeDecodeError 又把文件灌成几十 MB。
            # 截断 + 替换不可解码字节，排障要的是栈，不是载荷。
            text = traceback.format_exc()
            f.write(text[:8000])
            if len(text) > 8000:
                f.write("\n...（已截断，完整堆栈 %d 字符）\n" % len(text))
    except Exception:
        pass
    traceback.print_exc()
    return JSONResponse({"detail": "服务器内部错误: %s" % str(exc)[:200]},
                        status_code=500)


@app.on_event("startup")
def _warm_models():
    """Load heavy models off the request path.

    CLIP takes 32 s to load (weights + CUDA context) and `/api/cull` needs it.
    Loading it lazily meant the **first** pick-the-best call hung for 300 s and
    timed out, which reads as "the feature is broken". Warming it in a daemon
    thread at startup makes the first real call fast while leaving the server
    responsive immediately.
    """
    def work():
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from engine import match as MT
            if MT.clip_available():
                MT._clip()
        except Exception:
            pass
        # ComfyUI 空闲自动停止的看护线程（拉起过的 15 分钟没用就停，
        # 还回 ~8GB 内存）
        try:
            from engine import comfy_mgr
            comfy_mgr.install_watcher()
        except Exception:
            pass
    threading.Thread(target=work, daemon=True).start()

_sessions: "OrderedDict[str, dict]" = OrderedDict()
_lock = threading.Lock()
_pipe = RetouchPipeline()
# 渲染缓存（/api/render）：会话+参数哈希+源版本 → 渲染结果。
# 32 条 × ~300KB ≈ 10MB 内存，换滑杆回拖的瞬时响应。
_RENDER_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_RENDER_CACHE_MAX = 32


# --------------------------------------------------------------------------
# parameter metadata, driving the UI
# --------------------------------------------------------------------------

def _p(key, label, lo, hi, default, group, hint=""):
    return {"key": key, "label": label, "min": lo, "max": hi,
            "default": default, "group": group, "hint": hint}


PARAM_SPEC = [
    # ---- 光影 tone -----------------------------------------------------
    _p("exposure", "曝光", -100, 100, 0, "光影",
       "（EV 显示 -5.00..+5.00，与 PS/像素蛋糕一致）：EV = 滑杆 / 20"),
    _p("contrast", "对比度", -100, 100, 0, "光影"),
    _p("brightness", "亮度", -100, 100, 0, "光影"),
    _p("highlights", "高光", -100, 100, 0, "光影"),
    _p("shadows", "阴影", -100, 100, 0, "光影"),
    _p("whites", "白色", -100, 100, 0, "光影"),
    _p("blacks", "黑色", -100, 100, 0, "光影"),
    # ---- 存在愁presence ------------------------------------------------
    _p("texture_tone", "纹理", -100, 100, 0, "存在感"),
    _p("clarity", "清晰度", -100, 100, 0, "存在感"),
    _p("dehaze", "去薄雾", -100, 100, 0, "存在感"),
    _p("vibrance", "鲜艳度", -100, 100, 0, "存在感"),
    _p("saturation", " 饱和", -100, 100, 0, "存在感"),
    # ---- 白平衁---------------------------------------------------------
    _p("temp", "色温", -100, 100, 0, "白平衡"),
    _p("tint", "色调", -100, 100, 0, "白平衡"),
    # ---- 美肤 skin ------------------------------------------------------
    _p("spotless_acne", "祛痘", 0, 100, 0, "美肤"),
    _p("spotless_spot", "祛斑", 0, 100, 0, "美肤"),
    _p("spotless_texture", "质感保留", 0, 100, 0, "美肤"),
    _p("even_tone", "肤色均匀", 0, 100, 0, "美肤"),
    _p("smooth", "磨皮", 0, 100, 0, "美肤"),
    _p("texture", "肤质细节", 0, 100, 50, "美肤"),
    _p("freq_low", "低频平整", 0, 100, 0, "美肤"),
    _p("freq_high", "高频增强", 0, 100, 0, "美肤"),
    # ---- 五官 facial ----------------------------------------------------
    _p("removeEyeBag", "祛眼袋", 0, 100, 0, "五官"),
    _p("keepSilkworm", "保留卧蚕", 0, 100, 0, "五官"),
    _p("eyeHighlight", "眼神光", 0, 100, 0, "五官"),
    _p("teethWhitening", "牙齿美白", 0, 100, 0, "五官"),
    _p("teethDeYellow", "牙齿祛黄", 0, 100, 0, "五官"),
    _p("eyelid_tape", "祛双眼皮贴", 0, 100, 0, "五官",
       "软化眼睑上方的贴纸线，保留自然褶皱"),
    # ---- 美型 reshape ---------------------------------------------------
    _p("slimFace", "瘦脸", 0, 100, 0, "美型"),
    _p("jaw", "下颌", 0, 100, 0, "美型"),
    _p("cheekbone", "颧骨", 0, 100, 0, "美型"),
    _p("enlargeEye", "大眼", 0, 100, 0, "美型"),
    _p("eyeHeight", "眼高", -100, 100, 0, "美型"),
    _p("eyeDistance", "眼距", -100, 100, 0, "美型"),
    _p("noseWidth", "鼻翼", 0, 100, 0, "美型"),
    _p("mouthWidth", "嘴宽", 0, 100, 0, "美型"),
    _p("chinLength", "下巴", -100, 100, 0, "美型"),
    _p("foreheadHeight", "额头", -100, 100, 0, "美型"),
    # ---- 身形 body ------------------------------------------------------
    _p("legLengthen", "拉长腿", 0, 100, 0, "身形"),
    _p("slimLeg", "瘦腿", 0, 100, 0, "身形"),
    _p("slimWaist", "瘦腰", 0, 100, 0, "身形"),
    _p("slimArm", "瘦手臂", 0, 100, 0, "身形"),
    _p("neckLength", "天鹅颈", 0, 100, 0, "身形"),
    _p("shoulderWidth", "肩宽", -100, 100, 0, "身形"),
    _p("heightGain", "增高", 0, 100, 0, "身形"),
    _p("bodySmooth", "身型流畅", 0, 100, 0, "身形"),
    # ---- 手足（检测命中才生效，没有手/脚的画面安静跳过）---------------
    _p("hand_smooth", "手部平滑", 0, 100, 0, "手足"),
    _p("hand_tone", "手部匀肤", 0, 100, 0, "手足"),
    _p("foot_smooth", "脚部平滑", 0, 100, 0, "手足"),
    _p("foot_tone", "脚部匀肤", 0, 100, 0, "手足"),
    _p("shoe_clean", "鞋子祛瑕疵", 0, 100, 0, "手足"),
    # ---- 细节 detail ----------------------------------------------------
    # 锐化数量 0..150（PS/ACR 同款），半径 0.5..3.0 像素（ACR 同款！
    _p("sharpening", "锐化", 0, 150, 0, "细节"),
    _p("sharp_radius", "锐化半径", 0.5, 3.0, 1.0, "细节"),
    _p("sharp_detail", "锐化细节", 0, 100, 25, "细节"),
    _p("nr_luminance", "噪点消除", 0, 100, 0, "细节"),
    _p("nr_color", "减少杂色", 0, 100, 0, "细节"),
    _p("nr_detail", "降噪细节", 0, 100, 50, "细节"),
    _p("nr_smoothness", "平滑度", 0, 100, 50, "细节"),
    _p("ai_denoise", "AI 降噪", 0, 100, 0, "细节"),
    _p("ai_denoise_strength", "AI 强度", 0, 100, 50, "细节"),
    # ---- 镜头 optics ----------------------------------------------------
    _p("distortion", "畸变", -100, 100, 0, "镜头"),
    _p("vignetting", "暗角", -100, 100, 0, "镜头"),
    _p("vignetting_midpoint", "暗角中点", 0, 100, 50, "镜头"),
    _p("ca_amount", "色差", -100, 100, 0, "镜头", "正=消红边，负=消蓝边"),
    # ---- 场景（沙滩）----------------------------------------------------
    _p("beach_debris", "沙滩·祛杂物", 0, 100, 0, "场景"),
    _p("beach_flatten", "沙滩·沙面平整", 0, 100, 0, "场景"),
    _p("beach_warm", "沙滩·暖沙调", 0, 100, 0, "场景"),
    # ---- 颗粒 grain -----------------------------------------------------
    _p("grain_amount", "颗粒效果", 0, 100, 0, "颗粒"),
    _p("grain_size", "颗粒大小", 0, 100, 25, "颗粒"),
    _p("grain_roughness", "粗糙度", 0, 100, 50, "颗粒"),
    _p("grain_shadows", "阴影", 0, 100, 0, "颗粒"),
    _p("grain_midtones", "中间调", 0, 100, 0, "颗粒"),
    _p("grain_highlights", "高光", 0, 100, 0, "颗粒"),
    # ---- 颜色分级 grade -------------------------------------------------
    _p("grade_global_hue", "全局 色相", -180, 180, 0, "颜色分级"),
    _p("grade_global_sat", "全局 饱和", -100, 100, 0, "颜色分级"),
    _p("grade_global_lum", "全局 明度", -100, 100, 0, "颜色分级"),
    _p("grade_mid_hue", "中间谁色相", -180, 180, 0, "颜色分级"),
    _p("grade_mid_sat", "中间谁饱和", -100, 100, 0, "颜色分级"),
    _p("grade_mid_lum", "中间谁明度", -100, 100, 0, "颜色分级"),
    _p("grade_high_hue", "高光 色相", -180, 180, 0, "颜色分级"),
    _p("grade_high_sat", "高光 饱和", -100, 100, 0, "颜色分级"),
    _p("grade_high_lum", "高光 明度", -100, 100, 0, "颜色分级"),
    _p("grade_balance", "区域平衡", -100, 100, 0, "颜色分级",
       "负值把高光处理压向中间调，正值推高"),
    _p("grade_blending", "区域重叠", 0, 100, 50, "颜色分级",
       "越大过渡越平滑"),
    # ---- 背景背景 (AI) -------------------------------------------
    _p("matte_enabled", "智能抠图", 0, 1, 0, "背景", "开启后启用主体分离"),
    _p("bg_blur", "背景虚化", 0, 60, 0, "背景"),
    _p("bg_darken", "背景压暗", 0, 100, 0, "背景"),
    _p("bg_desat", "背景去色", 0, 100, 0, "背景"),
    _p("bg_mode", "纯色背景", 0, 1, 0, "背景", "1 = 使用下方颜色"),
    _p("bg_color_r", "背景 R", 0, 1, 1.0, "背景"),
    _p("bg_color_g", "背景 G", 0, 1, 1.0, "背景"),
    _p("bg_color_b", "背景 B", 0, 1, 1.0, "背景"),
    _p("matte_feather", "边缘羽化", 0, 6, 1.0, "背景"),
    _p("matte_refine", "边缘对齐", 0, 1, 1, "背景", "贴合真实轮廓"),
    # ---- 主体保护 --------------------------------------------------------
    _p("protect_background", "背景保护", 0, 1, 1.0, "背景",
       "调色时保护背景不被误改"),
]

# HSL bands are generated so the UI can render them in a loop.
# 色相单位丁PS/ACR 一致：-100..+100（引擎内部换算成 ±30°）、
for _b, _label in [("red", "红色"), ("orange", "橙色"), ("yellow", "黄色"),
                   ("green", "绿色"), ("aqua", "浅绿"), ("blue", "蓝色"),
                   ("purple", "紫色"), ("magenta", "洋红")]:
    PARAM_SPEC.append(_p(f"hsl_{_b}_hue", f"{_label} 色相", -100, 100, 0,
                         "HSL", "丁PS 一致：±100 = ±30° 色相旋转"))
    PARAM_SPEC.append(_p(f"hsl_{_b}_sat", f"{_label} 饱和", -100, 100, 0, "HSL"))
    PARAM_SPEC.append(_p(f"hsl_{_b}_lum", f"{_label} 明度", -100, 100, 0, "HSL"))


class RenderRequest(BaseModel):
    session: str
    params: Dict[str, Any] = {}
    quality: int = 88
    # Render at a smaller long edge. The preset strip asks for every preset's
    # thumbnail on a 220 px image; running the full-resolution pipeline seven
    # times to fill 100 px cards is pure waste, and the result is identical
    # because every stage is defined in relative coordinates.
    long_edge: int = 0
    # Ask for the histogram of the rendered result. `/api/render` is a pure
    # preview: it deliberately does not write back to `s["preview"]`, because
    # every slider tick would otherwise become an undo step. That also means
    # `/api/histogram` —which reads the session's stored preview —describes
    # the *uploaded* frame, not what the user is looking at. Returning the
    # histogram from the rendered pixels is the only way the panel can match.
    with_histogram: bool = False


def _histogram_of(img: np.ndarray, nb: int = 64) -> dict:
    """RGB + luminance histograms for one image."""
    nb = int(max(16, min(256, nb)))
    # Clip before binning: `np.histogram(range=(0,1))` puts the last bin at
    # [1-1/nb, 1), so a pixel exactly at 1.0 falls outside and is dropped.
    src = np.clip(img, 0.0, 1.0)
    out = {}
    for i, name in enumerate(("r", "g", "b")):
        h, _ = np.histogram(src[..., i], bins=nb, range=(0.0, 1.0))
        out[name] = h.astype(int).tolist()
    lum = np.clip((src * np.array([0.2126, 0.7152, 0.0722], np.float32))
                  .sum(axis=2), 0.0, 1.0)
    hl, _ = np.histogram(lum, bins=nb, range=(0.0, 1.0))
    out["lum"] = hl.astype(int).tolist()
    out["bins"] = nb
    out["clipped_low"] = int((lum <= 1e-4).sum())
    out["clipped_high"] = int((lum >= 1.0 - 1e-4).sum())
    out["total"] = int(lum.size)
    return out


class SaveRequest(BaseModel):
    session: str
    params: Dict[str, Any] = {}
    path: str


# --------------------------------------------------------------------------
# session store
# --------------------------------------------------------------------------

def _put_session(sid: str, data: dict) -> None:
    with _lock:
        _sessions[sid] = data
        _sessions.move_to_end(sid)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


# --------------------------------------------------------------------------
# undo history
# --------------------------------------------------------------------------

HISTORY_DEPTH = 12


def _new_history(preview: np.ndarray):
    """Create the undo stack for a session, seeded with the uploaded frame."""
    from engine.history import History

    return History(preview, depth=HISTORY_DEPTH)


def _commit(s: dict, out: np.ndarray, label: str) -> None:
    """Make `out` the session's working image and record an undo step.

    Every destructive operation goes through here rather than assigning
    `s["preview"]` directly, so the history cannot silently fall out of sync
    with the image —a step that mutates without recording would make undo
    skip past it and restore the wrong frame.
    """
    s["preview"] = out
    # 源图变了：作废该会话的渲染缓存（见 /api/render 的 _RENDER_CACHE）
    s["render_ver"] = int(s.get("render_ver", 0)) + 1
    hist = s.get("history")
    if hist is None:
        s["history"] = _new_history(out)
    else:
        hist.push(out, label)


def _get_session(sid: str) -> dict:
    with _lock:
        s = _sessions.get(sid)
        if s is None:
            raise HTTPException(404, "session not found (re-upload the image)")
        _sessions.move_to_end(sid)
        return s


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join(UI_DIR, "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>ZhaJinWenQianCake</h1><p>web/index.html missing</p>", 500)
    with open(path, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/api/params")
def params():
    # Derive the group order from PARAM_SPEC rather than listing it by hand:
    # a hard-coded list silently drops any group added later, and the UI builds
    # its tabs from this, so the controls become unreachable.
    groups: List[str] = []
    for spec in PARAM_SPEC:
        if spec["group"] not in groups:
            groups.append(spec["group"])
    return JSONResponse({"params": PARAM_SPEC, "groups": groups})


@app.get("/api/presets")
def presets():
    # 「我的、 用户保存的自定义预设，存在磁盘上（JSON），隁named presets
    # 一起返回，前端不需要第二套数据流、
    return JSONResponse({"presets": {**PRESETS, **_user_presets()}})


class SavePresetRequest(BaseModel):
    name: str
    params: Dict[str, Any]
    overwrite: bool = False


_USER_PRESET_FILE = os.path.join(DATA_DIR, "_user_presets.json")


def _user_presets() -> Dict[str, Dict[str, Any]]:
    try:
        with open(_USER_PRESET_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@app.get("/api/presets/user")
def user_presets_list():
    return JSONResponse({"presets": _user_presets()})


@app.post("/api/presets/save")
def presets_save(req: SavePresetRequest):
    """把当前参数存成「我的预设、

    名字会规范化为 key（去空格、转小写），内置预设名不允许覆盖——用户改皁
    应该是自己的预设，不动出厂值。`overwrite` 显式允许更新同名自定义预设、
    """
    key = "".join(c for c in req.name.strip().lower() if not c.isspace())
    if not key:
        raise HTTPException(400, "预设名不能为空")
    if key in PRESETS and not req.overwrite:
        raise HTTPException(409, "「%s」是内置预设，不能覆盖"% req.name)
    if not req.params:
        raise HTTPException(400, "参数为空（先把滑杆调到想要的值）")
    # 只保畁pipeline 认识的参数，避免存进一堆无效键
    from engine.pipeline import DEFAULTS

    clean = {k: v for k, v in req.params.items() if k in DEFAULTS}
    if not clean:
        raise HTTPException(400, "没有可识别的参数（全部不在管线定义里）")
    mine = _user_presets()
    if key in mine and not req.overwrite:
        raise HTTPException(409, "「%s」已存在，勾选覆盖再保存" % req.name)
    mine[key] = clean
    tmp = _USER_PRESET_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(mine, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, _USER_PRESET_FILE)
    return JSONResponse({"ok": True, "key": key, "params": clean})


@app.post("/api/presets/delete")
def presets_delete(req: dict = Body(...)):
    key = str(req.get("key") or "")
    mine = _user_presets()
    if key not in mine:
        raise HTTPException(404, "没有这个自定义预设：" + key)
    del mine[key]
    tmp = _USER_PRESET_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(mine, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, _USER_PRESET_FILE)
    return JSONResponse({"ok": True})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")

    tmp = os.path.join(DATA_DIR, "_upload_tmp")
    os.makedirs(tmp, exist_ok=True)
    ext = os.path.splitext(file.filename or "img.jpg")[1].lower() or ".jpg"
    is_camera_raw = ext in RAW_EXTENSIONS
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff") \
            and not is_camera_raw:
        ext = ".jpg"
    src = os.path.join(tmp, f"{int(time.time()*1000)}{ext}")
    with open(src, "wb") as fh:
        fh.write(raw)

    try:
        if is_camera_raw:
            # A camera RAW file is decoded through LibRaw with exposure
            # normalisation, not through the image loader: feeding a CFA
            # stream to cv2.imdecode is what produced "cannot decode image"
            # on the .jpg rename. The working image is the half-size decode —
            # a 4000px frame is 4x the preview cost for no retouching gain —
            # and the full decode happens at export time.
            full, decode_ms = rawio.timed_decode(src, half_size=True)
            decode_info = {"raw": True, "decode_ms": round(decode_ms, 1)}
        else:
            full = load_image(src)
            decode_info = {"raw": False}
    except Exception as exc:
        raise HTTPException(400, f"cannot decode image: {exc}")

    sid = f"s{int(time.time()*1000)}"
    preview = fit_long_edge(full, PREVIEW_LONG_EDGE)

    # detect once and reuse, so sliders do not re-run the detector
    faces_info = []
    try:
        det = _pipe.detector()
        if det is not None:
            for f in det.detect(preview):
                faces_info.append({"box": [round(v, 1) for v in f.box],
                                   "score": round(f.score, 3)})
    except Exception:
        pass

    _put_session(sid, {
        "full": full,
        "preview": preview,
        "src": src,
        "name": file.filename or os.path.basename(src),
        "faces": faces_info,
        "decode": decode_info,
        "created": time.time(),
        # Undo history is seeded with the uploaded frame, so the original is a
        # reachable state and undo can never walk past it.
        "history": _new_history(preview),
    })

    return JSONResponse({
        "session": sid,
        "width": int(full.shape[1]),
        "height": int(full.shape[0]),
        "preview_width": int(preview.shape[1]),
        "preview_height": int(preview.shape[0]),
        "faces": faces_info,
        "decode": decode_info,
        "preview": base64.b64encode(encode_jpeg(preview, 88)).decode("ascii"),
    })


@app.post("/api/render")
def render(req: RenderRequest):
    s = _get_session(req.session)
    src = s["preview"]

    # ---- 渲染缓存：滑杆回拖/参数对比是最高频操作，全管线重跑 1-4s/次。
    # key 含会话版本（_commit 递增），源图一变就自然失效。
    cache_key = None
    if req.long_edge and req.params is not None:
        cache_key = (req.session, int(req.long_edge), int(req.quality),
                     int(s.get("render_ver", 0)),
                     hashlib.sha256(json.dumps(
                         req.params, sort_keys=True,
                         ensure_ascii=False).encode()).hexdigest()[:20])
        hit = _RENDER_CACHE.get(cache_key)
        if hit is not None:
            _RENDER_CACHE.move_to_end(cache_key)
            payload = dict(hit)
            payload["cached"] = True
            return JSONResponse(payload)

    if req.long_edge and max(src.shape[:2]) > req.long_edge:
        # 缩略图用的小尺寸渲染：所有阶段都以相对坐标定义，缩过的图和全尺寸
        # 结果视觉上一致，但速度快一个数量级
        src = fit_long_edge(src, int(req.long_edge))
    t0 = time.perf_counter()
    out, info = _pipe.run(src, req.params, detect=True)
    ms = (time.perf_counter() - t0) * 1000
    payload = {
        "image": base64.b64encode(encode_jpeg(out, req.quality)).decode("ascii"),
        "ms": round(ms, 1),
        "applied": info["applied"],
        "stages": info["stages"],
        "faces": info["faces"],
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
    }
    if req.with_histogram:
        hg = _histogram_of(out)
        # 同一仁meta，让 UI 的说明行在预览渲染时也能显示尺寸
        hg["meta"] = {"width": int(out.shape[1]),
                      "height": int(out.shape[0]),
                      "raw": False}
        payload["histogram"] = hg
    if cache_key is not None:
        _RENDER_CACHE[cache_key] = payload
        while len(_RENDER_CACHE) > _RENDER_CACHE_MAX:
            _RENDER_CACHE.popitem(last=False)
    return JSONResponse(payload)


@app.get("/api/faces/{sid}")
def faces(sid: str):
    s = _get_session(sid)
    return JSONResponse({"faces": s["faces"],
                         "width": int(s["preview"].shape[1]),
                         "height": int(s["preview"].shape[0])})


def _write_exif_fields(dst: str, rx: dict) -> None:
    """抁RAW 读出的曝光字段写进导凁JPEG 皁EXIF.

    用Pillow 皁Exif 对象重建（不是透传）：Make/Model/Artist/DateTime
    进主 IFD，ISO/FNumber/ExposureTime/FocalLength/LensModel/
    DateTimeOriginal 迁Exif IFD——与相机原布局一致、
    """
    from PIL import Image
    from PIL.ExifTags import TAGS

    im = Image.open(dst)
    try:
        ex = im.getexif()
        t = TAGS
        put = {}
        if rx.get("camera"):
            parts = str(rx["camera"]).split(" ", 1)
            put[t.get("Make", 0x010F)] = parts[0]
            if len(parts) > 1:
                put[t.get("Model", 0x0110)] = parts[1]
        if rx.get("artist"):
            put[t.get("Artist", 0x013B)] = str(rx["artist"])
        if rx.get("taken"):
            put[t.get("DateTime", 0x0132)] = str(rx["taken"])
        for k, v in put.items():
            if k:
                ex[k] = v
        sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else None
        if sub is not None:
            sp = {}
            if rx.get("iso"):
                sp[t.get("ISOSpeedRatings", 0x8827)] = int(rx["iso"])
            if rx.get("aperture"):
                sp[t.get("FNumber", 0x829D)] = (
                    int(round(float(rx["aperture"]) * 10)), 10)
            if rx.get("shutter"):
                denom = max(1, int(round(1.0 / float(rx["shutter"]))))
                sp[t.get("ExposureTime", 0x829A)] = (1, denom)
            if rx.get("focal"):
                sp[t.get("FocalLength", 0x920A)] = (
                    int(round(float(rx["focal"]) * 10)), 10)
            if rx.get("lens"):
                sp[t.get("LensModel", 0xA434)] = str(rx["lens"])
            if rx.get("taken"):
                sp[t.get("DateTimeOriginal", 0x9003)] = str(rx["taken"])
            for k, v in sp.items():
                if k:
                    sub[k] = v
        exif_bytes = ex.tobytes()
        im.save(dst, "jpeg", quality=95, exif=exif_bytes)
    finally:
        im.close()


def _copy_exif(src: str, dst: str) -> dict:
    """把源文件皁EXIF 透传到导出的 JPEG.

    摄影师需要导出文件保留拍摄信息（ISO/光圈/快门/镜头/版权）——
    `save_image` 用OpenCV 编码，天生不带 EXIF。用 Pillow 皁Exif 对象
    原样搬运：RAW（NEF 等）无法直接嵌进 JPEG，这类源只透传 JPEG→JPEG！
    返回里说明做了什么，不静默假装成功、
    """
    note = {"copied": False}
    if not src or not os.path.isfile(src):
        note["reason"] = "no source file"
        return note
    src_ext = os.path.splitext(src)[1].lower()
    if src_ext not in (".jpg", ".jpeg"):
        note["reason"] = "source is %s; EXIF passthrough only jpg->jpg" % (
            src_ext or "?")
        return note
    try:
        from PIL import Image

        with Image.open(src) as im:
            exif = im.getexif()
        if not exif or not exif.tobytes():
            note["reason"] = "source has no exif"
            return note
        with Image.open(dst) as im2:
            im2.save(dst, "JPEG", quality=95, exif=exif.tobytes(),
                     subsampling=1)
        note = {"copied": True, "tags": len(exif)}
    except Exception as exc:
        note = {"copied": False, "reason": str(exc)[:100]}
    return note


@app.post("/api/save")
def save(req: SaveRequest):
    s = _get_session(req.session)
    # A RAW session's "full" image is the half-size decode used for previews.
    # Export re-decodes at full resolution so the file keeps everything the
    # sensor captured; a JPEG session's full image is already full-res.
    if s.get("decode", {}).get("raw") and os.path.exists(s.get("src", "")):
        try:
            full = rawio.decode(s["src"], half_size=False)
        except Exception as exc:
            raise HTTPException(500, f"full RAW decode failed: {exc}")
    else:
        full = s["full"]
    out, info = _pipe.run(full, req.params, detect=True)
    path = os.path.abspath(req.path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_image(path, out)
    exif_note = _copy_exif(s.get("src", ""), path)
    # RAW 源不透传 EXIF（_copy_exif 如实说明），佁RAW 的曝光字段恰恁
    # 是摄影师最看重的——用 rawexif 仁NEF/TIFF 头读出来写进 JPEG、
    if not exif_note.get("copied") and \
            s.get("src", "").lower().endswith((".nef", ".nrw")):
        try:
            from engine.rawexif import raw_exif

            rx = raw_exif(s["src"])
            if rx.get("camera"):
                _write_exif_fields(path, rx)
                exif_note = {"copied": True, "source": "raw",
                             "fields": [k for k in
                                        ("camera", "lens", "iso",
                                         "aperture", "shutter", "taken")
                                        if rx.get(k)]}
        except Exception:
            pass
    return JSONResponse({"path": path,
                         "width": int(out.shape[1]),
                         "height": int(out.shape[0]),
                         "applied": info["applied"],
                         "exif": exif_note})


class GreyPointRequest(BaseModel):
    session: str
    x: float
    y: float
    radius: int = 8
    strength: float = 1.0
    # The UI previews a grey-point pick once and applies it once, so the commit
    # flag exists mainly to keep the contract explicit: only a committed
    # operation becomes an undo step.
    commit: bool = True


@app.post("/api/greypoint")
def greypoint(req: GreyPointRequest):
    """Neutralise the picked patch and correct the whole frame."""
    from engine.wheel import grey_point

    s = _get_session(req.session)
    try:
        out, info = grey_point(s["preview"], req.x, req.y,
                               radius=int(req.radius),
                               strength=float(req.strength))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if req.commit:
        _commit(s, out, "灰点白平衡")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 90)).decode("ascii"),
        "info": info,
    })


class WheelRequest(BaseModel):
    session: str
    family: str
    hue: float = 0.0
    sat: float = 0.0
    lum: float = 0.0
    width: float = 0.5
    feather: float = 0.6
    # Dragging a wheel slider previews continuously; only "apply" commits. If
    # every preview became a history step, one drag would fill the stack and a
    # single undo would step back a fraction of a movement instead of an edit.
    commit: bool = False


@app.post("/api/wheel")
def wheel(req: WheelRequest):
    """One CMYK colour-family adjustment, for the colour-wheel UI."""
    from engine.wheel import select_color

    s = _get_session(req.session)
    try:
        out = select_color(s["preview"], req.family,
                           hue=req.hue, sat=req.sat, lum=req.lum,
                           width=req.width, feather=req.feather)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if req.commit:
        _commit(s, out, "色轮 " + req.family)
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 90)).decode("ascii"),
        "committed": bool(req.commit),
    })


@app.get("/api/health")
def health():
    """服务与模型的就绪状态。

    所有探测都走"只读已加载状态"的路径，**不主动加载模型**：
    `_pipe.detector()` / `_pipe.pose_estimator()` 那种调用会真的把模型
    读进显存，健康检查因此要 2~4 秒（而且每次都探）。健康检查是探活
    用的高频端点，必须 O(1)。
    """
    from engine import matting as _mt
    from engine import match as _match
    from engine import semantic as _sem
    from engine import nndenoise as _nn
    from engine import genfill as _gf
    from engine import superres as _sr
    from engine import lama as _lama

    return {"ok": True, "sessions": len(_sessions),
            "face_model": _pipe.detector_loaded(),
            "matting_model": _mt.available(),
            "semantic_model": _sem.available(),
            "denoise_model": _nn.available(),
            "superres_model": _sr.available(),
            "lama_model": _lama.available(),
            "clip": _match.clip_available(),
            "pose_model": _pipe.pose_loaded(),
            "generative_fill": _gf.available(),
            # 内存画像：模型常驻 + 会话上限（LRU=8）决定整体水位。
            # 诊断用——用户反馈"越用越卡"时先看这里。
            "memory_mb": _server_memory_mb(),
            "max_sessions": MAX_SESSIONS}


@app.get("/api/version")
def version():
    """版本与构建信息。UI 的「关于」和更新检查以此为本地基准。

    update_url 指向 GitHub Releases 的 latest 接口；仓库未公开时该地址
    返回 404，前端把它当「检查不可用」静默处理，绝不能弹错打扰用户。
    """
    return {
        "name": "炸金文谦蛋糕",
        "version": APP_VERSION,
        "build": APP_BUILD,
        "frozen": bool(getattr(sys, "frozen", False)),
        "python": sys.version.split()[0],
        # 公开仓库后更新检查才会真正生效；换分发渠道时改这里。
        "update_url": ("https://api.github.com/repos/wmc-ccb/ZhaJinWenQianCake"
                       "/releases/latest"),
    }


def _parse_error_log(limit: int = 20):
    """解析 out/error.log，按 `==== 时间 METHOD 路径` 分段返回最近几条。

    每条取：时间、请求、异常类型（堆栈最后一行）、首行堆栈上下文。
    崩溃上报的「复制诊断报告」直接用这份结构化数据，用户不用找文件。
    """
    from engine.paths import out_dir

    path = os.path.join(out_dir(), "error.log")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return []
    entries = []
    blocks = text.split("\n==== ")
    for block in reversed(blocks[1:]):  # 新的在后，倒序取
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        head = lines[0].strip()          # "2026-10-24 12:00:00  POST /api/x"
        exc_line = ""
        for l in reversed(lines):        # 最后一帧是异常类型与消息
            if l.startswith(("Traceback", "  ")) or l.startswith("\t"):
                continue
            exc_line = l.strip()
            break
        parts = head.split("  ", 1)
        entries.append({
            "time": parts[0],
            "request": parts[1] if len(parts) > 1 else "",
            "exception": exc_line[:300],
        })
        if len(entries) >= limit:
            break
    return entries


@app.get("/api/errors")
def errors(limit: int = 20):
    """最近的服务端未捕获异常（out/error.log 的结构化视图）。

    配合 /api/diagnose 组成「复制诊断报告」：发行版用户反馈问题时，
    点一下关于面板里的复制按钮即可，不用去翻安装目录。
    """
    return {"errors": _parse_error_log(min(limit, 100))}


@app.get("/api/diagnose")
def diagnose():
    """运行环境诊断：打包版排查 GPU / 依赖问题的入口。

    发行版里用户没法开终端跑 `python -c "import torch"`，出问题时
    "超清很慢"到底是"没 GPU"还是"模型太大"分不清。这个端点把关键
    事实一次列出来。
    """
    info = {"frozen": bool(getattr(sys, "frozen", False))}

    # Python / 进程
    info["python"] = sys.version.split()[0]
    info["executable"] = sys.executable
    try:
        from engine.paths import home
        info["home"] = home()
    except Exception as e:
        info["home_error"] = str(e)[:100]

    # torch / CUDA
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        info["cudnn"] = bool(getattr(torch.backends, "cudnn", None)
                             and torch.backends.cudnn.is_available())
    except Exception as e:
        info["torch_error"] = "%s: %s" % (type(e).__name__, str(e)[:120])

    # onnxruntime providers
    try:
        import onnxruntime as ort
        info["ort_version"] = ort.__version__
        info["ort_providers"] = list(ort.get_available_providers())
    except Exception as e:
        info["ort_error"] = str(e)[:120]

    # 关键模型是否就位
    try:
        from engine import (face, lama, matting, semantic, superres,
                            nndenoise, pose, cliponnx)
        info["models"] = {
            "face": os.path.exists(face.model_path()),
            "lama": os.path.exists(lama.model_path()),
            "matting": os.path.exists(matting.model_path()),
            "semantic": os.path.exists(semantic.model_path()),
            "superres": os.path.exists(superres.model_path()),
            "denoise": os.path.exists(nndenoise.model_path()),
            "pose": os.path.exists(pose.model_path()),
            "clip_onnx": cliponnx.available(),
        }
    except Exception as e:
        info["models_error"] = str(e)[:120]

    # **真正在 GPU 上跑一次**：is_available() 为 true 但初始化失败是
    # 打包环境最常见的假阳性（DLL 找到了、cublas 加载失败被吞掉）。
    try:
        import time as _t

        import torch
        if torch.cuda.is_available():
            a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            t0 = _t.time()
            for _ in range(10):
                b = a @ a
            torch.cuda.synchronize()
            ms = (_t.time() - t0) * 100
            info["gpu_matmul_ms"] = round(ms, 2)
            info["gpu_mem_allocated_mb"] = round(
                torch.cuda.memory_allocated() / 1024 ** 2, 1)
            info["gpu_works"] = True
            del a, b
            torch.cuda.empty_cache()
    except Exception as e:
        info["gpu_works"] = False
        info["gpu_error"] = "%s: %s" % (type(e).__name__, str(e)[:180])

    return info


def _server_memory_mb() -> float | None:
    """本进程 RSS（MB）。psutil 缺席时读 Windows API（psapi）。"""
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD),
                        ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        # K32GetProcessMemoryInfo（kernel32 直出）；不声昁argtypes 旁
        # 64 位句柄会被截成 32 佁int，调用静默失败返回0（实测）、
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wt.DWORD]
        fn.restype = wt.BOOL
        if fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return round(pmc.WorkingSetSize / 1048576, 0)
    except Exception:
        pass
    try:
        import resource          # POSIX 兜底
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                     1024, 0)
    except Exception:
        return None


# --------------------------------------------------------------------------
# AI endpoints
# --------------------------------------------------------------------------

class MatchRequest(BaseModel):
    session: str
    reference_session: str = ""
    reference_path: str = ""
    strength: float = 0.8
    method: str = "reinhard"
    preserve_luma: float = 0.35
    quality: int = 88
    # colour matching is a deliberate, one-shot edit, so it commits by default
    commit: bool = True


class CullRequest(BaseModel):
    sessions: List[str] = []
    paths: List[str] = []
    duplicates: bool = True
    # include a small thumbnail per result so the UI can show a contact sheet
    thumbs: bool = True
    max_thumbs: int = 48


@app.get("/api/matte/{sid}")
def matte(sid: str, mode: str = "hybrid"):
    """Return a subject matte for a session as a greyscale PNG.

    `mode` selects the source: "portrait" (MODNet, best hair detail),
    "semantic" (6-class segmentation, keeps accessories), or "hybrid"
    (both —the default, since it takes MODNet's hair and restores what a
    portrait matte drops).
    """
    s = _get_session(sid)
    from engine import matting as MT

    if not MT.available():
        raise HTTPException(503, "matting model not installed")
    if mode not in ("portrait", "semantic", "hybrid"):
        raise HTTPException(400, "mode must be portrait, semantic or hybrid")
    img = s["preview"]
    a = MT.subject_alpha(img, mode=mode)
    rgb = np.repeat(a[..., None], 3, axis=2)
    data = encode_jpeg(rgb, 88)
    return Response(content=data, media_type="image/jpeg",
                    headers={"X-Matte-Mode": mode,
                             "X-Matte-Coverage": "%.4f" % float((a > 0.5).mean())})


@app.get("/api/semantic/{sid}")
def semantic_regions(sid: str):
    """Per-class semantic regions for a session: coverage plus a colourised map."""
    s = _get_session(sid)
    from engine import semantic as SEM

    if not SEM.available():
        raise HTTPException(503, "semantic model not installed")
    reg = SEM.segment(s["preview"])
    if reg is None:
        raise HTTPException(503, "segmentation failed")

    colours = [(0.15, 0.15, 0.18), (0.95, 0.55, 0.15), (0.30, 0.75, 0.35),
               (0.95, 0.80, 0.65), (0.30, 0.55, 0.95), (0.90, 0.25, 0.55)]
    vis = np.zeros_like(s["preview"])
    cat = reg.category
    for c, col in enumerate(colours):
        vis[cat == c] = col
    return JSONResponse({
        "coverage": reg.coverage(),
        "dominant": reg.dominant(),
        "classes": list(SEM.CLASS_NAMES),
        "image": base64.b64encode(encode_jpeg(vis, 88)).decode("ascii"),
    })


@app.post("/api/match")
def match(req: MatchRequest):
    """Apply a reference image's colour character to a session."""
    from engine import match as MT
    from engine.imgutil import encode_jpeg

    s = _get_session(req.session)
    ref_img = None

    if req.reference_session:
        ref_img = _get_session(req.reference_session)["preview"]
    elif req.reference_path:
        if not os.path.exists(req.reference_path):
            raise HTTPException(404, "reference image not found")
        from engine.imgutil import load_image
        ref_img = load_image(req.reference_path)
    else:
        raise HTTPException(400, "provide reference_session or reference_path")

    if req.method not in ("reinhard", "histogram"):
        raise HTTPException(400, "method must be 'reinhard' or 'histogram'")

    out = MT.match_color(s["preview"], ref_img,
                         strength=float(req.strength),
                         preserve_luma=float(req.preserve_luma),
                         method=req.method)
    stats_src = MT.analyze_color(s["preview"])
    stats_ref = MT.analyze_color(ref_img)
    stats_out = MT.analyze_color(out)

    if req.commit:
        _commit(s, out, "追色")

    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, int(max(40, min(100, req.quality))))).decode(),
        "source": stats_src.as_dict(),
        "reference": stats_ref.as_dict(),
        "result": stats_out.as_dict(),
        "committed": bool(req.commit),
    })


# --------------------------------------------------------------------------
# undo / redo
# --------------------------------------------------------------------------

def _history_payload(s: dict, image: np.ndarray = None) -> dict:
    hist = s.get("history")
    out = {
        "can_undo": bool(hist and hist.can_undo()),
        "can_redo": bool(hist and hist.can_redo()),
        "label": hist.current_label() if hist else "原图",
        "steps": hist.list() if hist else [],
        "depth": HISTORY_DEPTH,
    }
    if image is not None:
        out["image"] = base64.b64encode(encode_jpeg(image, 90)).decode("ascii")
        out["width"] = int(image.shape[1])
        out["height"] = int(image.shape[0])
    return out


class HistoryRequest(BaseModel):
    session: str


@app.post("/api/undo")
def undo(req: HistoryRequest):
    """Step the session back one destructive edit."""
    s = _get_session(req.session)
    hist = s.get("history")
    if hist is None or not hist.can_undo():
        raise HTTPException(409, "nothing to undo")
    img, label = hist.undo()
    s["preview"] = img
    return JSONResponse(_history_payload(s, img) | {"restored": label})


@app.post("/api/redo")
def redo(req: HistoryRequest):
    """Step forward again after an undo."""
    s = _get_session(req.session)
    hist = s.get("history")
    if hist is None or not hist.can_redo():
        raise HTTPException(409, "nothing to redo")
    img, label = hist.redo()
    s["preview"] = img
    return JSONResponse(_history_payload(s, img) | {"restored": label})


@app.get("/api/history/{sid}")
def history_state(sid: str):
    """Current undo/redo state, so a reloaded page can rebuild its buttons."""
    s = _get_session(sid)
    return JSONResponse(_history_payload(s))


class UpscaleRequest(BaseModel):
    session: str
    tile: int = 256
    overlap: int = 32
    path: Optional[str] = None
    quality: int = 95
    commit: bool = False
    job: bool = False              # true = 后台跑, 用 job id 轮询进度


# 超清任务表：100-150 秒的重活必须有进度，否则用户以为死机。
_UPSCALE_JOBS: Dict[str, dict] = {}
_UPSCALE_JOBS_LOCK = threading.Lock()


def _upscale_job_status(j: dict) -> dict:
    return {k: j[k] for k in ("phase", "pct", "done", "total", "error")}


@app.get("/api/upscale/job/{job_id}")
def upscale_job(job_id: str):
    with _UPSCALE_JOBS_LOCK:
        j = _UPSCALE_JOBS.get(job_id)
        if j is None:
            raise HTTPException(404, "job not found")
        st = _upscale_job_status(j)
        if j["phase"] == "done":
            st.update(width=j["width"], height=j["height"], ms=j["ms"],
                      path=j["path"], scale=4)
        return JSONResponse(st)


@app.post("/api/upscale/job/{job_id}/cancel")
def upscale_job_cancel(job_id: str):
    with _UPSCALE_JOBS_LOCK:
        j = _UPSCALE_JOBS.get(job_id)
        if j is None:
            raise HTTPException(404, "job not found")
        j["cancel"] = True
    return JSONResponse({"ok": True})


@app.post("/api/upscale")
def upscale(req: UpscaleRequest):
    """4x super-resolution (Real-ESRGAN), tiled, saved to disk.

    `job=true` 时转后台任务并返回 job_id：超清是 100-150 秒的重活，
    同步 HTTP 会把前端吊在半空，进度用 /api/upscale/job/{id} 轮询。
    """
    from engine import superres as SR
    from engine.imgutil import save_image

    s = _get_session(req.session)
    if not SR.available():
        raise HTTPException(503, "super-resolution model not installed")

    if req.job:
        job_id = uuid.uuid4().hex[:12]
        with _UPSCALE_JOBS_LOCK:
            _UPSCALE_JOBS[job_id] = {"phase": "running", "pct": 0,
                                     "done": 0, "total": 0, "error": "",
                                     "width": 0, "height": 0, "ms": 0,
                                     "path": None, "cancel": False}
        # 只留最近 4 个任务（每个带 384MP 级结果的元数据，不大）
        with _UPSCALE_JOBS_LOCK:
            while len(_UPSCALE_JOBS) > 4:
                _UPSCALE_JOBS.popitem(last=False)

        def _work():
            with _UPSCALE_JOBS_LOCK:
                j = _UPSCALE_JOBS[job_id]
            try:
                def _progress(done, total):
                    with _UPSCALE_JOBS_LOCK:
                        j["done"], j["total"] = done, total
                        j["pct"] = int(done * 100 / max(total, 1))
                        if j["cancel"]:
                            raise RuntimeError("用户取消")
                src = s["full"] if s.get("full") is not None else s["preview"]
                t0 = time.time()
                out = SR.upscale(src, tile=int(max(64, req.tile)),
                                 overlap=int(max(8, req.overlap)),
                                 progress=_progress)
                ms = int((time.time() - t0) * 1000)
                saved = None
                if req.path:
                    saved = save_image(req.path, out,
                                       int(max(50, min(100, req.quality))))
                if req.commit:
                    s["preview"] = out
                with _UPSCALE_JOBS_LOCK:
                    j.update(phase="done", width=int(out.shape[1]),
                             height=int(out.shape[0]), ms=ms, path=saved)
            except Exception as exc:
                with _UPSCALE_JOBS_LOCK:
                    j.update(phase="error",
                             error=str(exc)[:200])

        threading.Thread(target=_work, daemon=True).start()
        return JSONResponse({"job": job_id})

    src = s["full"] if s.get("full") is not None else s["preview"]
    t0 = time.time()
    out = SR.upscale(src, tile=int(max(64, req.tile)),
                     overlap=int(max(8, req.overlap)))
    ms = int((time.time() - t0) * 1000)

    saved = None
    if req.path:
        saved = save_image(req.path, out, int(max(50, min(100, req.quality))))
    if req.commit:
        # a 4x result does not belong in an in-memory undo snapshot (a 24 MP
        # frame becomes 384 MP); committing is opt-in and records the step so
        # the session's own history stays consistent with what was done
        s["preview"] = out if req.commit else s["preview"]
    return JSONResponse({
        "width": int(out.shape[1]), "height": int(out.shape[0]),
        "ms": ms, "path": saved, "scale": 4,
        "model": "Real-ESRGAN x4plus (BSD-3)",
    })


class RemoverRequest(BaseModel):
    session: str
    # mask as base64 PNG (any single channel); same size as the preview
    mask_png: str = ""
    # or rectangles in preview pixels: [[x, y, w, h], ...]
    rects: List[List[float]] = []
    dilate: int = 4
    feather: float = 2.0
    # "lama" is local and fast; "diffusion" invents content and needs a prompt
    engine: str = "lama"
    prompt: str = "clean background, seamless, high quality"
    commit: bool = True


@app.get("/api/cats")
def cats():
    """右侧分类的两级结构（与参考软件的 12 个顶层分类对齐）."""
    from engine.cattree import EXTERNAL, TREE

    return JSONResponse({"tree": TREE, "external": EXTERNAL,
                         "parents": list(TREE.keys())})


@app.get("/api/idphoto/info")
def idphoto_info():
    from engine import idphoto as IP

    return JSONResponse(IP.info())


class IdPhotoRequest(BaseModel):
    session: str
    spec: str = "two_inch"
    colour: List[float] = [255.0, 255.0, 255.0]
    keep_lighting: bool = False
    out_path: str = ""
    commit: bool = False


class PetRequest(BaseModel):
    session: str
    shine: float = 0.0
    smooth: float = 0.0
    eye_glow: float = 0.0
    cutout: bool = False


@app.post("/api/pet")
def pet(req: PetRequest):
    """宠物修整：毛发光況顺毛/眼神光（颜色判据！ MODNet 抠图.

    本地没有宠物专用分割模型——这些都是确定性可验证的实现，如实标注、
    """
    from engine import pet as PE
    from engine.imgutil import encode_jpeg

    s = _get_session(req.session)
    img = s["preview"]
    try:
        if req.cutout:
            rgba, info = PE.cutout(img)
            # 合到白底出预览（透明信息在info 里，导出用PNG 通道！
            a = rgba[..., 3:4]
            out = rgba[..., :3] * a + np.array([1.0, 1.0, 1.0], np.float32) \
                * (1 - a)
        else:
            out, info = PE.groom(img, None, shine=req.shine,
                                 smooth=req.smooth, eye_glow=req.eye_glow)
    except Exception as e:
        raise HTTPException(500, "宠物修整失败: " + str(e)[:120])
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(np.clip(out, 0, 1), 92))
        .decode("ascii"),
        "info": info,
    })


class GenRequest(BaseModel):
    session: str
    feature: str          # backdrop | fill | prop
    prompt: str = ""
    strength: float = 1.0
    steps: int = 20
    seed: Optional[int] = None
    # fill/prop 的手涁mask（base64 PNG，可选）
    mask_png: str = ""
    commit: bool = False
    # 生成模型：sdxl（质量，默认）/ sd15（快速）/ flux（最高质量，需 11GB 模型）
    model: str = "sdxl"


@app.get("/api/gen/flux_status")
def gen_flux_status():
    """FLUX 高质量档可用性 + 缺失模型点名（面板档位 UI 的数据源）。"""
    from engine import genfill
    missing = genfill.flux_missing()
    return JSONResponse({"available": not missing, "missing": missing})


@app.post("/api/gen")
def gen(req: GenRequest):
    """生成式三件套（AI 布景/创成式填允智能道具修复！

    全部走本地 ComfyUI 皁SD1.5 inpaint + ControlNet；mask 外逐像素不变、
    ComfyUI 未运行或模型缺失时返回503，UI 提示启动方法、
    """
    import base64 as b64

    import cv2 as _cv

    from engine import gentools as GT
    from engine.imgutil import encode_jpeg

    s = _get_session(req.session)
    img = s["preview"]
    # AI 服务懒启动：没就绪时后台拉起并让前端等就绪重试，而不是把
    # 「去开终端」这个用户（打包版根本没有 Python）做不到的事甩回给他。
    from engine import comfy_mgr
    from engine import genfill as _gf
    if not _gf.available():
        comfy_mgr.start(background=True)
        raise HTTPException(503, "AI 服务正在启动，请稍候片刻再点生成")
    mask = None
    if req.mask_png:
        arr = np.frombuffer(b64.b64decode(req.mask_png), np.uint8)
        dec = _cv.imdecode(arr, _cv.IMREAD_GRAYSCALE)
        if dec is not None:
            mask = _cv.resize(dec, (img.shape[1], img.shape[0]),
                              interpolation=_cv.INTER_LINEAR)
            mask = mask.astype(np.float32) / 255.0
    subj = None
    try:
        from engine.pipeline import RetouchPipeline

        subj = RetouchPipeline.subject_mask(
            img.shape[:2], s.get("faces") or [])
    except Exception:
        subj = None
    try:
        if req.feature == "backdrop":
            if not req.prompt:
                raise HTTPException(400, "AI 布景需要场景描述")
            out, info = GT.ai_backdrop(img, req.prompt, subject_mask=subj,
                                       strength=req.strength,
                                       steps=req.steps, seed=req.seed,
                                       model=req.model)
        elif req.feature == "fill":
            if mask is None:
                raise HTTPException(400, "创成式填充需要手绘 mask")
            out, info = GT.generative_fill(img, mask, req.prompt,
                                           steps=req.steps, seed=req.seed,
                                           model=req.model)
        elif req.feature == "prop":
            out, info = GT.remove_prop(img, subject_mask=subj, mask=mask,
                                       strength=req.strength,
                                       steps=req.steps, seed=req.seed,
                                       model=req.model)
        else:
            raise HTTPException(400, "未知功能: " + req.feature)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        from engine import comfy_mgr as _cm
        _cm.start(background=True)
        raise HTTPException(503, "ComfyUI 不可用，正在尝试启动：" + str(e)[:80])
    except Exception as e:
        raise HTTPException(500, "生成失败: " + str(e)[:120])
    if req.commit:
        _commit(s, out, "生成弁" + req.feature)
    comfy_mgr.touch()      # 成功使用：空闲自动停止从这里重新计时
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info, "committed": bool(req.commit)})


@app.get("/api/automask/info")
def automask_info():
    from engine import automask as AM

    return JSONResponse(AM.info())


class AutoMaskRequest(BaseModel):
    session: str
    parts: List[str] = []
    feather: float = 1.0
    preview: bool = True


@app.post("/api/automask")
def automask(req: AutoMaskRequest):
    """自动蒙版：一次生成全部区域蒙版（人脸/皮肤/衣服/背景/头发/五官！

    类似 PS 的「选择主体」，但按部位分开给：两个分割模型分工——
    selfie_multiclass 管全身（含背晁衣服），SegFormer 管五官细节、
    返回彩色预览 + 每个区域的一比特 PNG（前端当缩略图用）、
    """
    import cv2 as _cv

    from engine import automask as AM
    from engine.imgutil import encode_jpeg

    s = _get_session(req.session)
    img = s["preview"]
    masks, info = AM.build(img, parts=req.parts or None,
                           feather=req.feather)
    payload = {"info": info, "image": ""}
    if req.preview:
        vis = AM.colourise(masks, img)
        payload["image"] = base64.b64encode(
            encode_jpeg(vis, 90)).decode("ascii")
    # 每个蒙版给一张小的灰度缩略图，UI 用来显示"这块长什么样"
    thumbs = {}
    for k, m in masks.items():
        small = _cv.resize(m, (160, 160), interpolation=_cv.INTER_AREA)
        thumbs[k] = base64.b64encode(
            encode_jpeg(np.dstack([small] * 3), 80)).decode("ascii")
    payload["thumbs"] = thumbs
    return JSONResponse(payload)


@app.post("/api/automask/export")
def automask_export(req: dict = Body(...)):
    """导出指定区域的蒙版（PNG 灰度 / RGBA 透明图）.

    `mode`: "mask" = 灰度 PNG（白=区域），"alpha" = 原图带alpha 通道、
    """
    import cv2 as _cv

    from engine import automask as AM

    s = _get_session(req.get("session") or "")
    part = str(req.get("part") or "")
    mode = str(req.get("mode") or "mask")
    img = s["preview"]
    masks, info = AM.build(img, parts=[part] if part else None)
    if not part:
        raise HTTPException(400, "请指定要导出的区域")
    if part not in masks:
        raise HTTPException(
            404, "该区域不可用: %s（%s）" % (
                part, info.get("unavailable") or "模型未给出"))
    m = masks[part]
    if mode == "alpha":
        rgba = AM.to_alpha(m, img)
        data = _cv.imencode(".png",
                            _cv.cvtColor(rgba, _cv.COLOR_RGBA2BGRA))[1]
    else:
        data = _cv.imencode(".png", (np.clip(m, 0, 1) * 255).astype(
            np.uint8))[1]
    name = (s.get("name") or "image").rsplit(".", 1)[0] + "_" + part + ".png"
    return Response(content=data.tobytes(), media_type="image/png",
                    headers={"Content-Disposition":
                             'attachment; filename="%s"' % name})


class SkyRequest(BaseModel):
    session: str
    preset: str = "clear_blue"
    strength: float = 1.0
    match_brightness: bool = True


class GrassRequest(BaseModel):
    session: str
    strength: float = 1.0


class RigRequest(BaseModel):
    session: str
    strength: float = 1.0


@app.post("/api/rigfix")
def rigfix(req: RigRequest):
    """去道具（灯架/柔光箁三脚架）：形状识刁+ 主体保护 + LaMa 补全."""
    from engine import rigfix as RF
    from engine.imgutil import encode_jpeg

    s = _get_session(req.session)
    img = s["preview"]
    subj = None
    try:
        from engine.pipeline import RetouchPipeline

        subj = RetouchPipeline.subject_mask(
            img.shape[:2], s.get("faces") or [])
    except Exception:
        subj = None
    try:
        out, info = RF.remove_rig(img, subject_mask=subj,
                                  strength=req.strength)
    except Exception as e:
        raise HTTPException(500, "去道具失败：" + str(e)[:120])
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info,
    })


@app.post("/api/grassfix")
def grassfix(req: GrassRequest):
    """AI 补草地：识别草地上的穿帮（杂牁人影），LaMa 用周围草地补允

    画面里草地占比过低时返回 applicable=false（如实说明，不硬猜）、
    """
    from engine import grassfix as GF
    from engine.imgutil import encode_jpeg

    s = _get_session(req.session)
    img = s["preview"]
    subj = None
    try:
        from engine.pipeline import RetouchPipeline

        subj = RetouchPipeline.subject_mask(
            img.shape[:2], s.get("faces") or [])
    except Exception:
        subj = None
    try:
        out, info = GF.fix_grass(img, subject_mask=subj,
                                 strength=req.strength)
    except Exception as e:
        raise HTTPException(500, "补草地失败：" + str(e)[:120])
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info,
    })


class SceneRequest(BaseModel):
    session: str
    preset: str = "sakura"
    strength: float = 1.0
    keep_structure: float = 0.25


@app.post("/api/scene")
def scene_replace(req: SceneRequest):
    """四季背景：分割保人物 + 程序化季节场晁+ 结构明暗保留."""
    from engine import scene as SC
    from engine.imgutil import encode_jpeg

    s = _get_session(req.session)
    img = s["preview"]
    try:
        out, info = SC.apply_scene(img, req.preset, strength=req.strength,
                                   keep_structure=req.keep_structure)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, "换背景失败：" + str(e)[:120])
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info,
    })


@app.post("/api/sky")
def sky_replace(req: SkyRequest):
    """换天空：确定性管线（分割 ∁颜色似然 ；地平线约杁；程序化天空）.

    不需要生成模型。`bg_mask` 用selfie_multiclass 皁background 类把
    人物/建筑从候选里排除；亮度匹配让新天空贴着原图的光感、
    """
    from engine import semantic as SEM
    from engine import sky as SK
    from engine.imgutil import encode_jpeg

    s = _get_session(req.session)
    img = s["preview"]
    try:
        bg = SEM.segment(img).norm("background")
    except Exception:
        bg = None
    try:
        out, info = SK.replace_sky(img, req.preset, bg_mask=bg,
                                   strength=req.strength,
                                   match_brightness=req.match_brightness)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, "换天空失败：" + str(e)[:120])
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info,
    })


@app.post("/api/idphoto")
def idphoto(req: IdPhotoRequest):
    """证件照：规格裁切 + 换底色，可直接导出到指定路径.

    支持一寁二寸/小二寁签证/护照；底色白/蓁红。头部占比按规范自动定位！
    头顶空间不足时用估计背景色补边（马上被换掉，不可见）、
    """
    import cv2 as _cv

    from engine import idphoto as IP
    from engine.imgutil import encode_jpeg, save_image

    s = _get_session(req.session)
    try:
        out, info = IP.make_id_photo(s["preview"], req.spec,
                                     tuple(req.colour),
                                     keep_lighting=req.keep_lighting)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, "证件照生成失败：" + str(e)[:120])
    if req.commit:
        _commit(s, _cv.resize(out, (s["preview"].shape[1],
                                    s["preview"].shape[0])), "证件照")
    path = ""
    if req.out_path:
        path = os.path.abspath(req.out_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_image(path, out, quality=96)
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 94)).decode("ascii"),
        "info": info,
        "width": int(out.shape[1]), "height": int(out.shape[0]),
        "path": path, "committed": bool(req.commit)})


@app.post("/api/remove_passersby")
def do_remove_passersby(req: dict = Body(...)):
    """AI 祛路人：把背景里的人抹掉.

    双尺度人脸检测找路人 -> 主体重叠判定 -> 头身比例掩膜 + 强制主体
    保护匁-> 填充。`model`：lama（快 ~5s）或 sdxl（质釁~40s，复杁
    背景明显更干净，需要ComfyUI 在8388 运行；失败自动回落 lamaa）、
    """
    from engine import passersby as PB

    s = _get_session(req.get("session") or "")
    strength = float(req.get("strength", 1.0))
    # 默认 lama：祛路人昁内容延续"任务，LaMa 正合适；SDXL 在窄人形
    # 掩膜里倾向画出替代的假人（实测合影发明了两个路人）。sdxl 档保畁
    # 给大区域重造场景的用法、
    model = str(req.get("model") or "lama")
    out, info = PB.remove_passersby(s["preview"], strength=strength,
                                    model=model)
    if req.get("commit", False) and info.get("removed", 0) > 0:
        _commit(s, out, "AI 祛路人")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info,
        "committed": bool(req.get("commit", False))})


@app.post("/api/remove")
def remove(req: RemoverRequest):
    """Object removal: LaMa locally, or the diffusion filler for content swaps.

    Two engines behind one endpoint because the choice is about intent, not
    quality —LaMa continues the surrounding texture (a cable on a wall), while
    diffusion invents new content (a different backdrop). Picking the wrong one
    gives smudge or nothing at all.
    """
    from engine.imgutil import load_image, fit_long_edge

    s = _get_session(req.session)
    if req.engine not in ("lama", "diffusion"):
        raise HTTPException(400, "engine must be lama or diffusion")

    img = s["preview"]
    h, w = img.shape[:2]
    mask = None
    if req.mask_png:
        raw = base64.b64decode(req.mask_png)
        arr = np.frombuffer(raw, np.uint8)
        dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if dec is None:
            raise HTTPException(400, "mask_png could not be decoded")
        if dec.ndim == 3:
            dec = dec[..., 0] if dec.shape[2] == 1 else dec.max(axis=2)
        if dec.shape[:2] != (h, w):
            dec = cv2.resize(dec, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (dec > 127).astype(np.float32)
    elif req.rects:
        mask = np.zeros((h, w), np.float32)
        for r in req.rects:
            x, y, rw, rh = [int(v) for v in (list(r) + [0, 0, 0, 0])[:4]]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + max(1, rw)), min(h, y + max(1, rh))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1.0
    else:
        raise HTTPException(400, "provide mask_png or rects")

    if mask.max() <= 0:
        raise HTTPException(400, "mask is empty")

    t0 = time.time()
    if req.engine == "lama":
        from engine import lama as LA

        if not LA.available():
            raise HTTPException(503, "LaMa model not installed")
        out, info = LA.fill(img, mask, dilate=int(req.dilate),
                            feather=float(req.feather))
    else:
        from engine import genfill

        if not genfill.available():
            raise HTTPException(503, "ComfyUI not reachable for diffusion fill")
        out, info = genfill.fill(img, mask, prompt=req.prompt)
    info["seconds"] = round(time.time() - t0, 2)
    info["engine"] = req.engine
    if req.commit:
        _commit(s, out, "消除 " + req.engine)
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info,
        "committed": bool(req.commit),
    })


class MaskedRequest(BaseModel):
    session: str
    # semantic classes to include; "subject" and "background" are derived
    masks: List[str] = []
    # adjustment controls, the same names the global panel uses
    controls: Dict[str, float] = {}
    feather: float = 1.5
    invert: bool = False
    commit: bool = True


class CropRequest(BaseModel):
    session: str
    crop: Optional[Dict[str, float]] = None
    rotate: float = 0.0
    flip_h: bool = False
    flip_v: bool = False
    ratio: str = "free"
    commit: bool = True


class GeometryInfo(BaseModel):
    pass


class SelectiveColorRequest(BaseModel):
    session: str
    bands: Dict[str, Dict[str, float]] = {}
    width: float = 34.0
    commit: bool = True


class ColorEditorRequest(BaseModel):
    session: str
    bands: Dict[str, Dict[str, float]] = {}
    width: float = 30.0
    commit: bool = True


class LookRequest(BaseModel):
    session: str
    profile: str = ""
    filter: str = ""
    commit: bool = True


class PostGradeRequest(BaseModel):
    session: str
    shadow_tint: List[float] = [0.0, 0.0, 0.0]
    highlight_tint: List[float] = [0.0, 0.0, 0.0]
    balance: float = 0.0
    vignette: float = 0.0
    glow: float = 0.0
    temperature: float = 0.0
    commit: bool = True


class PortraitRequest(BaseModel):
    session: str
    # 表情管理
    smile: float = 0.0
    eye_open: float = 0.0
    brow_raise: float = 0.0
    mouth_narrow: float = 0.0
    squash: float = 0.0
    # 妆容
    lip: float = 0.0
    lip_tone: str = "rose"
    blush: float = 0.0
    blush_tone: str = "coral"
    shadow: float = 0.0
    shadow_tone: str = "nude"
    brow: float = 0.0
    brow_tone: str = "brick"
    # 头发
    hair_volume: float = 0.0
    hair_shine: float = 0.0
    hair_colour: float = 0.0
    hair_rgb: Optional[List[float]] = None
    hair_frizz: float = 0.0
    # 衣物
    wrinkle: float = 0.0
    stain: float = 0.0
    structure: float = 0.0
    cloth_tint: float = 0.0
    cloth_rgb: Optional[List[float]] = None
    commit: bool = True


class DefringeRequest(BaseModel):
    session: str
    defringe: float = 0.0
    dehalo: float = 0.0
    contract: float = 0.0
    commit: bool = True


class BgCleanRequest(BaseModel):
    session: str
    showthrough: float = 0.0      # 祛穿带
    blemish: float = 0.0          # 祛瑕疁
    even_light: float = 0.0       # 光影均匀
    banding: float = 0.0          # 祛色彩断屁
    enhance: float = 0.0          # 背景增强
    use_subject: bool = True      # 用抠图结果当背景掩膜（最准）
    commit: bool = True


class BgReplaceRequest(BaseModel):
    session: str
    colour: List[float] = [1.0, 1.0, 1.0]   # 目标背景艁RGB
    feather: float = 1.5
    keep_lighting: bool = True     # 保留原背景的明暗关系
    commit: bool = True


@app.get("/api/library/info")
def library_info():
    from engine import library as LB

    return JSONResponse(LB.info())


_LIB_ROOTS_CACHE = {"at": 0.0, "data": None, "running": False}
_LIB_ROOTS_TTL = 300.0


def _scan_roots_background():
    """Fill the roots cache off the request thread.

    Scanning every drive synchronously took 71.8 s on first call —far past any
    sane HTTP timeout, and it blocked the whole event loop for photo serving.
    The endpoint now returns whatever is cached and kick-starts a background
    refresh, so the first call answers immediately with "scanning" and the UI
    fills in on the next poll.
    """
    import time as _t

    from engine import library as LB

    if _LIB_ROOTS_CACHE.get("running"):
        return
    _LIB_ROOTS_CACHE["running"] = True

    def work():
        try:
            roots = []
            for letter in string.ascii_uppercase:
                base = letter + ":\\"
                if not os.path.exists(base):
                    continue
                # 系统盘只下钻 2 层（几万个文件），数据盘 4 屁
                depth = 2 if letter in ("C",) else 4
                try:
                    gs = LB.groups(base, max_depth=depth)
                except Exception:
                    gs = []
                if gs:
                    roots.append({"path": base, "name": letter + ":",
                                  "groups": gs,
                                  "count": sum(g["count"] for g in gs)})
            _LIB_ROOTS_CACHE["data"] = {"roots": roots}
            _LIB_ROOTS_CACHE["at"] = _t.time()
        finally:
            _LIB_ROOTS_CACHE["running"] = False

    threading.Thread(target=work, daemon=True).start()


@app.get("/api/library/roots")
def library_roots(refresh: bool = False):
    """照片库根目录。首次调用返回 `scanning`，后台扫完后可再取。"""
    import time as _t

    now = _t.time()
    cached = _LIB_ROOTS_CACHE["data"]
    fresh = cached is not None and now - _LIB_ROOTS_CACHE["at"] < _LIB_ROOTS_TTL
    if refresh and not _LIB_ROOTS_CACHE.get("running"):
        _LIB_ROOTS_CACHE["at"] = 0.0
        fresh = False
    if not fresh:
        _scan_roots_background()
    if cached is None:
        return JSONResponse({"roots": [], "scanning": True})
    return JSONResponse({"roots": cached["roots"],
                         "scanning": bool(_LIB_ROOTS_CACHE.get("running"))})


@app.get("/api/library/scan")
def library_scan(path: str, limit: int = 300):
    """扫描一个目录成为图库：缩略图 + 技术指标 + 自动分组.

    分组判据（抖/欠曝/过曝/重复）用的是**这一批自身的分位数**，不是固定
    阈值——夜景和日光的"正常"不是一个数，固定阈值会把整批夜景标成欠曝。
    """
    from engine import library as LB

    if not path or not os.path.isdir(path):
        raise HTTPException(400, "目录不存在 " + str(path))
    res = LB.scan(path, limit=int(max(1, min(2000, limit))))
    # 缩略图由单独端点按需给，初始响应只带元数捁
    return JSONResponse({
        "root": res["root"], "count": res["count"],
        "groups": res["groups"],
        "items": [{k: v for k, v in it.items() if k != "index"}
                  for it in res["items"]],
    })


_THUMB_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_THUMB_CACHE_MAX = 600


@app.get("/api/library/thumb")
def library_thumb(path: str, size: int = 320):
    """缩略图，带惰态LRU 缓存.

    网格一屁20~30 张，每次滚动都重新解码RAW 会卡；缓存按 (path, size)
    孁JPEG 字节，上陁600 条（纁30 MB），超出按最早使用淘汰、
    """
    import time as _t

    from engine import library as LB
    from engine.imgutil import encode_jpeg

    size = int(max(64, min(720, size)))
    key = "%s|%d" % (path, size)
    hit = _THUMB_CACHE.get(key)
    if hit is not None:
        # 源文件更新过就作庁
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0
        if hit[0] == mtime:
            _THUMB_CACHE.move_to_end(key)
            return Response(content=hit[1], media_type="image/jpeg")

    img = LB.load_thumb(path, size)
    if img is None:
        raise HTTPException(404, "无法读取: " + str(path))
    data = encode_jpeg(img, 82)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    _THUMB_CACHE[key] = (mtime, data)
    _THUMB_CACHE.move_to_end(key)
    while len(_THUMB_CACHE) > _THUMB_CACHE_MAX:
        _THUMB_CACHE.popitem(last=False)
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/library/facecrop")
def library_facecrop(path: str, size: int = 220):
    """智能预览：自动找最大的人脸裁特冁

    参考软件图库的「智能预览」是脸部特写卡——缩略图看不清表惁对焦时，
    特写一眼可判。实现直接复用图库缩略图管线：EXIF 内嵌预览 ；YuNet
    最大的脁；按脸框外扁1.6 倍裁剪。没有脸就退回整图居中裁方、
    """
    from engine import library as LB
    from engine.face import FaceDetector
    from engine.imgutil import encode_jpeg

    size = int(max(64, min(512, size)))
    img = LB.load_thumb(path, 480)
    if img is None:
        raise HTTPException(404, "无法读取: " + str(path))
    h, w = img.shape[:2]
    try:
        faces = FaceDetector(dense=False).detect(img, max_faces=5)
    except Exception:
        faces = []
    if faces:
        f = max(faces, key=lambda f: f.box[2] * f.box[3])
        x, y, bw, bh = [float(v) for v in f.box]
        cx, cy = x + bw / 2, y + bh / 2
        half = max(bw, bh) * 0.85          # 外扩到含发际与下巁
        x0 = int(max(0, cx - half)); x1 = int(min(w, cx + half))
        y0 = int(max(0, cy - half)); y1 = int(min(h, cy + half))
        if x1 - x0 < 8 or y1 - y0 < 8:
            x0, y0, x1, y1 = 0, 0, w, h
        img = img[y0:y1, x0:x1]
    # 方形化：短边居中裁
    hh, ww = img.shape[:2]
    s = min(hh, ww)
    y0 = (hh - s) // 2; x0 = (ww - s) // 2
    img = img[y0:y0 + s, x0:x0 + s]
    import cv2 as _cv
    img = _cv.resize(img, (size, size), interpolation=_cv.INTER_AREA)
    return Response(content=encode_jpeg(img, 84), media_type="image/jpeg")


@app.get("/api/library/exif")
def library_exif(path: str):
    from engine import library as LB

    if not path or not os.path.isfile(path):
        raise HTTPException(404, "文件不存在")
    return JSONResponse(LB.exif(path))


_LIB_SOURCES_FILE = os.path.join(DATA_DIR, "_library_sources.json")


def _lib_sources() -> List[dict]:
    """用户自行导入的图库来源（文件夹），持久化。"""
    try:
        with open(_LIB_SOURCES_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _lib_sources_save(data: List[dict]) -> None:
    tmp = _LIB_SOURCES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, _LIB_SOURCES_FILE)


@app.get("/api/library/sources")
def library_sources():
    """图库来源列表（用户导入过的文件夹），带照片计数。"""
    from engine import library as LB

    out = []
    for src in _lib_sources():
        p = src.get("path") or ""
        if not os.path.isdir(p):
            out.append({**src, "count": -1, "missing": True})
            continue
        try:
            n = len(LB.list_images(p, recursive=bool(src.get("recursive")),
                                   limit=5000))
        except Exception:
            n = 0
        out.append({**src, "count": n})
    return JSONResponse({"sources": out})


class LibSourceRequest(BaseModel):
    path: str
    recursive: bool = False


@app.post("/api/library/import")
def library_import(req: LibSourceRequest):
    """用户自行导入一个照片文件夹到图庁

    与「自动扫描全盘」不同：图库一开始是空的，由用户指明哪个文件夹是
    照片库（参考软件同为用户导入）。导入即扫一遍该文件夹，立即出缩略图
    与技术分组，并把来源记到磁盘，下次启动直接可用、
    """
    from engine import library as LB

    path = os.path.abspath(req.path)
    if not os.path.isdir(path):
        raise HTTPException(404, "目录不存在 " + path)
    res = LB.scan(path, limit=int(min(2000, 800)),
                  workers=12)
    srcs = _lib_sources()
    if not any((s.get("path") or "") == path for s in srcs):
        srcs.append({"path": path, "recursive": req.recursive,
                     "added": time.time()})
        _lib_sources_save(srcs)
    return JSONResponse({
        "root": res["root"], "count": res["count"],
        "groups": res["groups"],
        "items": res["items"],
        "sources": _lib_sources(),
    })


@app.post("/api/library/source_remove")
def library_source_remove(req: dict = Body(...)):
    path = str(req.get("path") or "")
    srcs = [s for s in _lib_sources() if (s.get("path") or "") != path]
    _lib_sources_save(srcs)
    return JSONResponse({"ok": True, "sources": srcs})


_LIB_DONE_FILE = os.path.join(DATA_DIR, "_library_done.json")


def _lib_done() -> Dict[str, dict]:
    """源图路径 -> 导出记录（已修闭环）。"""
    try:
        with open(_LIB_DONE_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _lib_done_save(data: Dict[str, dict]) -> None:
    tmp = _LIB_DONE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, _LIB_DONE_FILE)


@app.post("/api/library/mark_done")
def library_mark_done(req: dict = Body(...)):
    """标记一张源回已修"：记录源路径 -> 导出路径/时间.

    图库格子按这个显礁✁角标，摄影师一眼看出哪些还没处理、
    """
    src = str(req.get("src") or "")
    out_path = str(req.get("out") or "")
    if not src or not out_path:
        raise HTTPException(400, "src 与 out 必填")
    data = _lib_done()
    data[os.path.abspath(src)] = {
        "out": os.path.abspath(out_path),
        "at": time.time(),
        "params": req.get("params") or {},
    }
    # 上限保护：只留最迁2000 杁
    if len(data) > 2000:
        keep = sorted(data.items(), key=lambda kv: -kv[1].get("at", 0))[:2000]
        data = dict(keep)
    _lib_done_save(data)
    return JSONResponse({"ok": True, "total": len(data)})


@app.get("/api/library/done")
def library_done():
    return JSONResponse({"done": _lib_done()})


@app.post("/api/library/open")
def library_open(req: dict = Body(...)):
    """把图库里的一张图载入成可编辑会话（和「打开」同样的路径！"""
    from engine.imgutil import load_image, fit_long_edge

    path = req.get("path") or ""
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "文件不存在 " + path)
    img = load_image(path)
    sid = _store_preview(img, path)
    s = _get_session(sid)
    return JSONResponse({
        "session": sid,
        "width": int(s["preview"].shape[1]),
        "height": int(s["preview"].shape[0]),
        "faces": s["faces"],
        "name": os.path.basename(path),
        "image": base64.b64encode(encode_jpeg(s["preview"], 88)).decode("ascii"),
    })


_DEV_CACHE = {"at": 0.0, "data": None, "running": False}
_DEV_TTL = 20.0


@app.get("/api/library/groups")
def library_groups():
    """「我的分组」：命名的选片集（持久化）。"""
    from engine import lgroup as LG

    return JSONResponse({"groups": LG.list_groups()})


@app.get("/api/library/group_members")
def library_group_members(name: str):
    from engine import lgroup as LG

    try:
        return JSONResponse({"name": name, "paths": LG.members(name)})
    except Exception:
        raise HTTPException(404, "分组不存在 " + name)


class GroupAddRequest(BaseModel):
    name: str
    paths: List[str]


@app.post("/api/library/group_add")
def library_group_add(req: GroupAddRequest):
    from engine import lgroup as LG

    try:
        return JSONResponse(LG.add(req.name, req.paths))
    except ValueError as e:
        raise HTTPException(400, str(e))


class GroupPathsRequest(BaseModel):
    name: str
    paths: List[str] = []


@app.post("/api/library/group_remove")
def library_group_remove(req: GroupPathsRequest):
    from engine import lgroup as LG

    try:
        return JSONResponse(LG.remove(req.name, req.paths))
    except KeyError:
        raise HTTPException(404, "分组不存在 " + req.name)


@app.post("/api/library/group_delete")
def library_group_delete(req: dict = Body(...)):
    from engine import lgroup as LG

    try:
        return JSONResponse(LG.delete_group(str(req.get("name") or "")))
    except KeyError:
        raise HTTPException(404, "分组不存在")


class GroupRenameRequest(BaseModel):
    old: str
    new: str


@app.post("/api/library/group_rename")
def library_group_rename(req: GroupRenameRequest):
    from engine import lgroup as LG

    try:
        return JSONResponse(LG.rename(req.old, req.new))
    except KeyError:
        raise HTTPException(404, "分组不存在 " + req.old)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/device/status")
def device_status(refresh: bool = False):
    """相机当前处于什么模式，以及为什么不能能做什么

    这台 Z6 III 在两私USB 模式下的表现实测为：
      MTP/PTP —WPD 设备枚举正常，但存储内容丁0 项；WIA 识别凁
                Nikon Z6_3（DeviceInfo 类型 2 = 数码相机），但命令数 0、
                项目数0，即**不提供第三方拍照与取文件**、
      可见但不可用 ；快门控制必须走官方SDK 或digiCamControl、

    查询要起 PowerShell 用WIA，首次约 3~5 秒。面板一打开就等这个会让
    "检测中" 挂住，所以首次立即返回检测中"并在后台刷新，之后命中缓存、
    """
    import time as _t

    def probe():
        from engine import wpd as WPD

        try:
            devices = WPD.list_devices()
        except Exception:
            devices = []
        wia = []
        try:
            ps = ("$m = New-Object -ComObject WIA.DeviceManager;"
                  "$m.DeviceInfos | ForEach-Object { [pscustomobject]@{"
                  " Name=$_.Properties.Item('Name').Value;"
                  " Type=$_.Type;"
                  " Cmds=$_.Commands.Count;"
                  " Items=$_.Items.Count } } | ConvertTo-Json -Compress")
            out = WPD._ps(ps, timeout=30)
            import json as _json
            data = _json.loads(out) if out.strip() else []
            if isinstance(data, dict):
                data = [data]
            wia = data
        except Exception:
            pass
        usable = [w for w in wia if (w.get("Cmds") or 0) > 0]
        return {
            "wpd": devices, "wia": wia,
            "can_capture": bool(usable),
            "reason": None if usable else (
                "相机被 Windows 识别为 Nikon，但驱动未提供拍照命令与文件"
                "列表（WIA 命令数 0、项目数 0）。第三方无法通过系统接口控"
                "快门，需要改用 Nikon 官方 SDK 或 digiCamControl。"),
        }

    now = _t.time()
    cached = _DEV_CACHE["data"]
    fresh = cached is not None and now - _DEV_CACHE["at"] < _DEV_TTL
    if refresh:
        fresh = False
    if not fresh and not _DEV_CACHE.get("running"):
        _DEV_CACHE["running"] = True

        def work():
            try:
                _DEV_CACHE["data"] = probe()
                _DEV_CACHE["at"] = _t.time()
            finally:
                _DEV_CACHE["running"] = False

        threading.Thread(target=work, daemon=True).start()
    if cached is None:
        return JSONResponse({"wpd": [], "wia": [], "can_capture": False,
                             "reason": "检测中", "scanning": True})
    out = dict(cached)
    out["scanning"] = bool(_DEV_CACHE.get("running"))
    return JSONResponse(out)


@app.get("/api/bgclean/info")
def bgclean_info():
    from engine import bgclean as BC
    from engine import matting as MT

    return JSONResponse({**BC.info(),
                         "matting": MT.available(),
                         "background_mask": "corner-colour or matte"})


@app.post("/api/bgclean")
def do_bgclean(req: BgCleanRequest):
    """纯色背景的四类缺陷修夁+ 背景增强.

    前提是准确的背景掩膜：优先用抠图模型（`use_subject`），否则从四见
    采背景色。纯色背景下这两条路都可靠，所以这几项能做得比通用分割好——
    不依赖模型识刁这是背景"、
    """
    from engine import bgclean as BC
    from engine import matting as MT

    s = _get_session(req.session)
    img = s["preview"]
    subject = None
    if req.use_subject and MT.available():
        try:
            subject = MT.subject_alpha(img, mode="hybrid")
        except Exception:
            subject = None
    out, info = BC.apply(img, subject=subject,
                         showthrough=req.showthrough,
                         blemish=req.blemish,
                         even_light=req.even_light,
                         banding=req.banding,
                         enhance=req.enhance)
    info["subject_mask"] = subject is not None
    if req.commit:
        _commit(s, out, "背景消除")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info, "committed": bool(req.commit)})


@app.post("/api/bgreplace")
def do_bgreplace(req: BgReplaceRequest):
    """把背景换成指定纯色（证件煁白底红底蓝底！

    掩膜来源按背景纯度择优：
      * 纯色背景（证件照/影棚）→ **色键**。逐像素按色差判定，边缘分辨率
        高于任何模型，且没有模型不确定性、
      * 复杂背景 ；抠图模型、
    之前无条件用抠图模型，合成测试里身体区域 alpha 掉到 0，底色直接透到
    身上——这种错误比"边缘不够干净"严重得多、
    """
    from engine import bgclean as BC
    from engine import matting as MT
    from engine.imgutil import luminance

    s = _get_session(req.session)
    img = s["preview"]

    bgmask = BC.background_mask(img)
    pur = BC.purity(img, bgmask)
    source = "color_key"
    info = {"purity": round(pur, 3)}
    if pur >= 0.5:
        alpha, kinfo = BC.color_key_alpha(img)
        info.update(kinfo)
    elif MT.available():
        alpha = MT.subject_alpha(img, mode="hybrid")
        source = "matting"
    else:
        raise HTTPException(503, "抠图模型不可用，且背景不是纯色")
    info["source"] = source

    a = np.clip(alpha, 0.0, 1.0)
    if req.feather > 0:
        import cv2 as _cv
        a = _cv.GaussianBlur(a, (0, 0), float(req.feather))
    a = np.clip(a, 0, 1)[..., None]

    colour = np.clip(np.array(req.colour, np.float32), 0, 1).reshape(1, 1, 3)
    newbg = np.tile(colour, img.shape[:2] + (1,))
    if req.keep_lighting:
        # 原背景的亮度起伏（去掉均值）乘到新颜色上：保留打光感
        lum = luminance(img)[..., None]
        bgsel = a[..., 0] < 0.5
        if bgsel.sum() > 100:
            base = float(np.median(lum[bgsel]))
            rel = np.clip(lum / max(base, 1e-3), 0.6, 1.4)
            newbg = np.clip(newbg * rel, 0, 1)
    out = np.clip(img * a + newbg * (1 - a), 0, 1)
    if req.commit:
        _commit(s, out, "背景替换")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info, "committed": bool(req.commit)})


@app.get("/api/crop/info")
def crop_info():
    from engine import crop as CR

    return JSONResponse(CR.info())


@app.post("/api/crop")
def do_crop(req: CropRequest):
    """Non-destructive geometry.

    The session keeps the *original* preview and applies geometry on every
    render from it, so a crop can be reopened, adjusted or removed at any time —
    cropping the already-cropped image would lose the discarded pixels for
    good and make "undo the crop" impossible.
    """
    from engine import crop as CR

    s = _get_session(req.session)
    base = s.get("preview_orig")
    if base is None:
        base = s["preview"]
        s["preview_orig"] = base.copy()

    ratio = CR.ratio_value(req.ratio, base.shape[:2])
    box = req.crop
    if box:
        box = CR.clamp_box(box)
        if ratio:
            box = CR.fit_ratio(box, ratio, base.shape[:2])

    geo = {"crop": box, "rotate": float(req.rotate),
           "flip_h": bool(req.flip_h), "flip_v": bool(req.flip_v),
           "ratio": req.ratio}
    s["geometry"] = geo
    if not req.commit:
        out = CR.apply(base, box, req.rotate, req.flip_h, req.flip_v)
        # 预览用的图也要缩放，否则前端坐标对不丁
        return JSONResponse({
            "image": base64.b64encode(encode_jpeg(
                fit_long_edge(out, PREVIEW_LONG_EDGE), 92)).decode("ascii"),
            "width": int(out.shape[1]), "height": int(out.shape[0]),
            "geometry": geo})

    out = CR.apply(base, box, req.rotate, req.flip_h, req.flip_v)
    s["preview"] = fit_long_edge(out, PREVIEW_LONG_EDGE)
    _commit(s, s["preview"], "裁剪")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(s["preview"], 92)).decode("ascii"),
        "width": int(out.shape[1]), "height": int(out.shape[0]),
        "geometry": geo})


@app.post("/api/crop/reset")
def crop_reset(req: CropRequest):
    s = _get_session(req.session)
    s["geometry"] = None
    if s.get("preview_orig") is not None:
        s["preview"] = s["preview_orig"].copy()
        _commit(s, s["preview"], "重置裁剪")
    return JSONResponse({"ok": True,
                         "width": int(s["preview"].shape[1]),
                         "height": int(s["preview"].shape[0])})


@app.post("/api/crop/preview")
def crop_preview(req: CropRequest):
    """Live crop preview without touching the session."""
    from engine import crop as CR

    s = _get_session(req.session)
    base = s.get("preview_orig")
    if base is None:
        base = s["preview"]
    ratio = CR.ratio_value(req.ratio, base.shape[:2])
    box = req.crop
    if box:
        box = CR.clamp_box(box)
        if ratio:
            box = CR.fit_ratio(box, ratio, base.shape[:2])
    out = CR.apply(base, box, req.rotate, req.flip_h, req.flip_v)
    small = fit_long_edge(out, 640)
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(small, 85)).decode("ascii"),
        "width": int(out.shape[1]), "height": int(out.shape[0])})


@app.get("/api/grading/info")
def grading_info():
    from engine import grading as GR

    return JSONResponse(GR.info())


@app.post("/api/selective_color")
def do_selective_color(req: SelectiveColorRequest):
    from engine import grading as GR

    s = _get_session(req.session)
    out = GR.selective_color(s["preview"], req.bands, req.width)
    info = {"bands": req.bands, "changed": [
        k for k, v in req.bands.items() if any(abs(x) > 1e-6
                                               for x in (v or {}).values())]}
    if req.commit:
        _commit(s, out, "可选颜色")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info, "committed": bool(req.commit)})


@app.post("/api/color_editor")
def do_color_editor(req: ColorEditorRequest):
    from engine import grading as GR

    s = _get_session(req.session)
    out = GR.color_editor(s["preview"], req.bands, req.width)
    info = {"bands": req.bands}
    if req.commit:
        _commit(s, out, "色彩编辑器")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info, "committed": bool(req.commit)})


@app.post("/api/look")
def do_look(req: LookRequest):
    """配置文件 + 滤镜，两丁LUT 合成一次重采样."""
    from engine import grading as GR

    s = _get_session(req.session)
    out = s["preview"]
    applied = []
    if req.profile:
        lut = GR.profile_lut(req.profile)
        if lut is not None:
            out = GR.apply_lut(out, lut)
            applied.append("profile:" + req.profile)
    if req.filter:
        lut = GR.filter_lut(req.filter)
        if lut is not None:
            out = GR.apply_lut(out, lut)
            applied.append("filter:" + req.filter)
    if req.commit:
        _commit(s, out, "配置文件/滤镜")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "applied": applied, "committed": bool(req.commit)})


@app.post("/api/post_grade")
def do_post_grade(req: PostGradeRequest):
    from engine import grading as GR

    s = _get_session(req.session)
    out = GR.post_grade(s["preview"], req.shadow_tint, req.highlight_tint,
                        req.balance, req.vignette, req.glow, req.temperature)
    if req.commit:
        _commit(s, out, "后调色")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "committed": bool(req.commit)})


def _face_box_xyxy(box) -> np.ndarray:
    """Detector box -> (x0, y0, x1, y1).

    The detector reports (x, y, w, h) while the landmark model's `align_crop`
    expects (x0, y0, x1, y1). Handing it the raw detector box fed w and h in as
    the right/bottom edges, so the aligned crop landed far off the face and the
    predicted landmarks were positioned outside it —the deforming warp then
    moved the wrong region entirely.
    """
    x, y, w, h = [float(v) for v in box[:4]]
    return np.asarray([x, y, x + w, y + h], np.float32)


def _face_points5(img):
    """The 5 YuNet landmarks (eye centres, nose tip, mouth corners), or None.

    These replaced the 106-point model for portrait work. That model's output
    was measured to be *not* the InsightFace layout its indices assume —the
    point taken as "inner lip" landed in the upper half of the face and "left
    eye" spanned more than half the face height —so every anchor built from it
    was in the wrong place. YuNet's own five points carry unambiguous meaning.
    """
    try:
        det = _pipe.detector()
        if det is None:
            return None
        faces = det.detect(img, dense=False)
        if not faces:
            return None
        pts = np.asarray(faces[0].landmarks, np.float32)
        return pts if pts.shape[0] >= 5 else None
    except Exception:
        return None


@app.post("/api/portrait")
def do_portrait(req: PortraitRequest):
    """表情 / 妆容 / 头发 / 衣物，一次调用完成，共享同一份关键点与掩膁"""
    from engine import portrait as PT
    from engine import semantic as SEM

    s = _get_session(req.session)
    img = s["preview"]
    out = img
    applied = []
    skipped = []

    lm = _face_points5(img)
    wants_shape = any([req.smile, req.eye_open, req.brow_raise,
                       req.mouth_narrow, req.squash])
    wants_makeup = any([req.lip, req.blush, req.shadow, req.brow])

    # 妆容现在用geoface 的几何遮罩，不再需要face-parsing 的分割结果：
    # 那个模型内部 128 网格，嘴唇只有约 15 格宽，实测眉毁口腔覆盖率为 0、
    if wants_shape:
        if lm is None:
            # 明确报告而不是静默跳过：没有检测到人脸时这些滑杆不可能生效！
            # 用户需要知道是"没脸"而不昁坏了"
            skipped.append("expression:未检测到人脸")
        else:
            out = PT.expression(out, lm, req.smile, req.eye_open,
                                req.brow_raise, req.mouth_narrow, req.squash)
            applied.append("expression")

    if wants_makeup:
        if lm is None:
            skipped.append("makeup:未检测到人脸")
        else:
            out = PT.makeup(out, lm, None, req.lip, req.lip_tone,
                            req.blush, req.blush_tone, req.shadow,
                            req.shadow_tone, req.brow, req.brow_tone)
            applied.append("makeup")

    if any([req.hair_volume, req.hair_shine, req.hair_colour, req.hair_frizz]):
        hm = None
        if SEM.available():
            reg = SEM.segment(out)
            if reg is not None:
                hm = reg.norm("hair")
        if hm is not None and hm.max() > 0.02:
            rgb = tuple(req.hair_rgb) if req.hair_rgb else None
            out = PT.hair(out, hm, req.hair_volume, req.hair_shine,
                          req.hair_colour, rgb, req.hair_frizz)
            applied.append("hair")
        else:
            skipped.append("hair:未分割到头发")

    if any([req.wrinkle, req.stain, req.structure, req.cloth_tint]):
        cm = None
        if SEM.available():
            reg = SEM.segment(out)
            if reg is not None:
                cm = reg.norm("clothes")
        if cm is not None and cm.max() > 0.02:
            rgb = tuple(req.cloth_rgb) if req.cloth_rgb else None
            out = PT.clothes(out, cm, req.wrinkle, req.stain, req.structure,
                             req.cloth_tint, rgb)
            applied.append("clothes")
        else:
            skipped.append("clothes:未分割到衣物")

    if req.commit:
        _commit(s, out, "人像精修")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "applied": applied, "skipped": skipped,
        "has_face": lm is not None,
        "committed": bool(req.commit)})


@app.post("/api/defringe")
def do_defringe(req: DefringeRequest):
    from engine import defringe as DF

    s = _get_session(req.session)
    out = DF.apply(s["preview"], req.defringe, req.dehalo, req.contract)
    if req.commit:
        _commit(s, out, "祛边")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "committed": bool(req.commit)})


@app.get("/api/history/{sid}")
def history(sid: str):
    s = _get_session(sid)
    h = s["history"]
    return JSONResponse({
        "steps": h.list(),
        "index": h.index,
        "can_undo": h.can_undo(),
        "can_redo": h.can_redo(),
        "depth": h.depth,
    })


@app.post("/api/history/{sid}/goto")
def history_goto(sid: str, req: dict = Body(...)):
    """Jump to any step in the undo stack —the 历史 panel's core action."""
    s = _get_session(sid)
    h = s["history"]
    idx = int(req.get("index", -1))
    if idx < 0 or idx >= len(h.steps):
        raise HTTPException(400, "index out of range")
    img = h.goto(idx)
    if img is None:
        raise HTTPException(500, "history step unreadable")
    s["preview"] = img
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(img, 92)).decode("ascii"),
        "index": h.index, "steps": h.list()})


@app.get("/api/history/{sid}/thumb/{index}")
def history_thumb(sid: str, index: int, full: int = 0):
    s = _get_session(sid)
    h = s["history"]
    # 前后对比（\ 键）用640px：清晰可判；全尺寸会增加传输没有意义！
    # 对比时画布本身也就是一屏大小。快照按需编码，不占常驻内存、
    b = h.thumbnail(int(index), 640 if full else 160)
    if b is None:
        raise HTTPException(404, "no thumbnail")
    return Response(content=b, media_type="image/jpeg")


@app.get("/api/masks/{sid}")
def mask_catalog(sid: str):
    """What this frame can be masked by, with coverage per class.

    Two models, merged rather than replaced: selfie_multiclass answers the
    whole-frame questions (hair / clothes / skin areas) and face-parsing
    answers the inside-the-face ones (brows / eyes / lips / teeth) that the
    first model cannot see. A class with no pixels here is reported at zero
    rather than offered and then silently doing nothing.
    """
    from engine import semantic as SEM
    from engine import matting as MT
    from engine import faceparse as FP

    s = _get_session(sid)
    cov = {}
    if SEM.available():
        reg = SEM.segment(s["preview"])
        if reg is not None:
            cov = reg.coverage()
    sub = {}
    if FP.available():
        try:
            parts = FP.parse(s["preview"], face_box=_first_face_box(s["preview"]))
            if parts is not None:
                sub = parts.coverage()
        except Exception as exc:
            sub = {"_error": str(exc)[:120]}
    return JSONResponse({
        "available": SEM.available(),
        "classes": list(SEM.CLASS_NAMES),
        "coverage": cov,
        "submasks": FP.GROUP_LABELS if FP.available() else {},
        "submask_coverage": sub,
        "matting": MT.available(),
    })


@app.post("/api/masked")
def masked_adjust(req: MaskedRequest):
    """Apply global controls inside a semantic mask."""
    from engine import semantic as SEM
    from engine import local as L
    from engine import faceparse as FP
    from engine.imgutil import fit_long_edge

    s = _get_session(req.session)
    img = s["preview"]
    h, w = img.shape[:2]

    keys = [str(k) for k in req.masks]
    if not keys:
        raise HTTPException(400, "select at least one mask")

    mask = None
    reg = None
    source = "semantic"
    if SEM.available():
        reg = SEM.segment(img)
        if reg is not None:
            sel = []
            for k in keys:
                if k == "subject":
                    sel.append(reg.subject())
                elif k == "background":
                    sel.append(1.0 - reg.subject())
                elif k == "skin":
                    sel.append(reg.skin())
                elif k in reg.masks:
                    sel.append(reg.norm(k))
            if sel:
                mask = sel[0]
                for extra in sel[1:]:
                    mask = np.maximum(mask, extra)
    if mask is None:
        # fall back to the portrait matte when the semantic model is absent, so
        # "subject" still means something rather than failing
        from engine import matting as MT

        if not MT.available():
            raise HTTPException(503, "no masking model installed")
        mask = MT.subject_alpha(img, mode="hybrid")
        source = "matte"

    # face sub-masks (brows / eyes / lips / teeth) come from the fine parser.
    # When both models produce masks the result is the face sub-mask intersected
    # with the subject, so a parser misfire elsewhere in the frame cannot light
    # up the backdrop; on its own the sub-mask stands alone.
    face_keys = [k for k in keys if k in FP.GROUP_LABELS]
    if face_keys:
        if not FP.available():
            raise HTTPException(503,
                                "face-parsing model not installed; sub-masks "
                                "(" + ", ".join(face_keys) + ") unavailable")
        parts = FP.parse(img, face_box=(_first_face_box(img)))
        if parts is None:
            raise HTTPException(503, "face parsing failed")
        face = parts.union(*face_keys)
        if mask is not None:
            subject = np.clip(reg.subject(), 0, 1) if reg is not None else None
            if subject is not None:
                mask = face * subject      # AND: stay on the person
            else:
                mask = face
        else:
            mask = face
        source = "faceparse"
        keys = face_keys

    out, touched = L.apply_through_mask(img, mask, req.controls,
                                        feather=float(req.feather),
                                        invert=bool(req.invert),
                                        return_mask=True)
    info = {"source": source, "masks": keys,
            "coverage": round(float((mask > 0.5).mean()), 4),
            "touched": round(float(touched.mean()), 4),
            "controls": {k: v for k, v in req.controls.items() if v}}
    if req.commit:
        _commit(s, out, "蒙版调整")
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "info": info,
        "committed": bool(req.commit),
    })


@app.get("/api/histogram/{sid}")
def histogram(sid: str, bins: int = 64):
    """Histogram of the session's stored preview.

    This describes the **committed** image. A live slider edit is a preview
    render that is intentionally not written back, so a caller showing the
    histogram next to live edits should use `/api/render?with_histogram` instead;
    this endpoint is right for "what does the frame I saved/committed look like".
    """
    s = _get_session(sid)
    out = _histogram_of(s["preview"], bins)
    d = s.get("decode") or {}
    out["meta"] = {
        "width": int(s["preview"].shape[1]),
        "height": int(s["preview"].shape[0]),
        "raw": bool(d.get("raw")),
        "iso": d.get("iso"), "focal": d.get("focal"),
        "aperture": d.get("aperture"), "shutter": d.get("shutter"),
        "camera": d.get("camera"), "lens": d.get("lens"),
    }
    return JSONResponse(out)


def _first_face_box(img):
    """The most confident detected face box, or None.

    Face parsing runs on a 512x512 input, so a face occupying a small part of a
    full frame loses its internal structure in the resize; cropping to this box
    first is what makes brows/eyes/lips resolve at all.
    """
    try:
        det = _pipe.detector()
        if det is None:
            return None
        faces = det.detect(img)
        return faces[0].box if faces else None
    except Exception:
        return None


class TetherConfig(BaseModel):
    kind: str = "folder"            # folder | digicamcontrol | ptp
    path: str = ""                  # watch dir, or download dir for DCC
    exe: str = ""                   # digiCamControl executable override
    settle_s: float = 1.2
    # what happens to a photo the moment it arrives
    auto_import: bool = True
    auto_preset: str = ""
    auto_formula: str = ""
    auto_export_dir: str = ""
    # keep the imported session's history clean by recording the auto-edit
    reset_params: bool = True


_TETHER: dict = {
    "source": None,
    "config": {},
    "inbox": [],            # [{"path", "name", "session", "at", "preset", "out"}]
    "enabled": False,
}


def _tether_source(cfg: Optional[dict] = None):
    """Get or (re)build the active source from the stored config."""
    from engine import tether as TT

    cfg = cfg or _TETHER.get("config") or {}
    cur = _TETHER.get("source")
    want = cfg.get("kind", "folder")
    if cur is not None and cur.kind == want:
        # keep a folder source pointed at the configured directory
        if want == "folder" and getattr(cur, "path", None) != cfg.get("path", ""):
            cur.path = cfg.get("path", "")
            cur.reset()
        if want == "digicamcontrol":
            cur.download_dir = cfg.get("path", "") or cur.download_dir
            cur.settle_s = float(cfg.get("settle_s", 1.2))
        return cur
    src = TT.make_source(want, path=cfg.get("path", ""),
                         exe=cfg.get("exe", ""),
                         download_dir=cfg.get("path", ""),
                         settle_s=float(cfg.get("settle_s", 1.2)))
    _TETHER["source"] = src
    return src


@app.get("/api/tether/status")
def tether_status():
    """Which source is active, whether it can shoot, and what the OS can see."""
    from engine import tether as TT

    src = _tether_source()
    cfg = _TETHER.get("config") or {}
    st = src.status()
    st["enabled"] = bool(_TETHER.get("enabled"))
    st["config"] = cfg
    st["inbox_count"] = len(_TETHER["inbox"])
    st["cameras"] = TT.detect_cameras()
    st["digicamcontrol"] = TT.dcc_installed()
    st["inbox"] = _TETHER["inbox"][-24:]
    return JSONResponse(st)


@app.post("/api/tether/config")
def tether_config(cfg: TetherConfig):
    """Point the source at a directory / switch backend, and arm the auto chain."""
    from engine import tether as TT

    if cfg.kind not in ("folder", "digicamcontrol", "ptp"):
        raise HTTPException(400, "kind must be folder, digicamcontrol or ptp")
    _TETHER["config"] = cfg.model_dump()
    src = _tether_source(_TETHER["config"])
    src.reset()
    # baseline what is already in the folder, so arming a watch does not import
    # the back catalogue —only frames that arrive from here on
    primed = src.prime()
    _TETHER["enabled"] = True
    if cfg.auto_preset:
        from engine.pipeline import PRESETS
        if cfg.auto_preset not in PRESETS:
            raise HTTPException(404, "unknown preset: " + cfg.auto_preset)
    return JSONResponse({"ok": True, "status": src.status(),
                         "primed": primed,
                         "config": _TETHER["config"]})


@app.post("/api/tether/stop")
def tether_stop():
    _TETHER["enabled"] = False
    return JSONResponse({"ok": True})


@app.post("/api/tether/capture")
def tether_capture():
    """Trigger a shot, then run whatever has arrived through the auto chain."""
    src = _tether_source()
    res = src.capture()
    time.sleep(0.4)
    arrived = _tether_ingest(src.poll())
    res["ingested"] = arrived
    return JSONResponse(res)


def _tether_ingest(paths: List[str]) -> List[dict]:
    """Auto-process newly arrived files.

    The whole point of tethering is that the retoucher looks at a finished
    frame, so an arriving file is uploaded as a session, the configured
    preset/formula is applied, and the result is exported when an export
    directory is set. Failures are recorded per file —one unreadable frame
    must not stop the shoot.
    """
    from engine.imgutil import load_image, save_image

    cfg = _TETHER.get("config") or {}
    done: List[dict] = []
    for p in paths:
        entry = {"path": p, "name": os.path.basename(p),
                 "at": time.time(), "ok": False}
        try:
            img = load_image(p)
            sid = _store_preview(img, p)
            entry["session"] = sid
            s = _get_session(sid)

            if cfg.get("auto_preset"):
                from engine.pipeline import apply_preset
                out, _info = _pipe.run(img, apply_preset(cfg["auto_preset"]),
                                       detect=True)
                _commit(s, out, "联机 " + cfg["auto_preset"])
                entry["preset"] = cfg["auto_preset"]
            elif cfg.get("auto_formula"):
                from engine import formula as formula_engine
                lib = formula_engine.load_library()
                if cfg["auto_formula"] in lib:
                    out, _info = formula_engine.run(lib[cfg["auto_formula"]],
                                                    img, pipeline=_pipe)
                    _commit(s, out, "联机配方 " + cfg["auto_formula"])
                    entry["formula"] = cfg["auto_formula"]

            if cfg.get("auto_export_dir"):
                os.makedirs(cfg["auto_export_dir"], exist_ok=True)
                base = os.path.splitext(os.path.basename(p))[0]
                dest = os.path.join(cfg["auto_export_dir"], base + "_out.jpg")
                save_image(dest, s["preview"], 95)
                entry["out"] = dest
            entry["ok"] = True
        except Exception as exc:
            entry["error"] = str(exc)[:200]
        _TETHER["inbox"].append(entry)
        done.append(entry)
    if len(_TETHER["inbox"]) > 200:
        del _TETHER["inbox"][:-200]
    return done


def _store_preview(img, src_path: str) -> str:
    """Register an already-decoded image as a session, as upload would."""
    from engine.imgutil import fit_long_edge

    h, w = img.shape[:2]
    preview = fit_long_edge(img, PREVIEW_LONG_EDGE)
    sid = uuid.uuid4().hex[:12]
    faces_info = []
    try:
        det = _pipe.detector()
        if det is not None:
            for f in det.detect(preview):
                faces_info.append({"box": [round(v, 1) for v in f.box],
                                   "score": round(f.score, 3)})
    except Exception:
        pass
    _put_session(sid, {
        "full": img, "preview": preview, "src": src_path,
        "name": os.path.basename(src_path),
        "faces": faces_info, "decode": {}, "created": time.time(),
        "history": _new_history(preview),
    })
    return sid


@app.post("/api/tether/poll")
def tether_poll():
    """Report anything that appeared since the last poll, and process it.

    POST rather than GET even though it only reads the folder: handling an
    arrival has **side effects** (stores a session, applies the preset, writes
    an export). A GET that mutates state can be replayed by a browser
    prefetcher, a proxy retry, or a crawler and import the same frame twice.
    The poll is a command, so it is not safe or idempotent by HTTP's definition.
    """
    if not _TETHER.get("enabled"):
        return JSONResponse({"enabled": False, "new": []})
    src = _tether_source()
    paths = src.poll()
    new = _tether_ingest(paths) if paths else []
    return JSONResponse({"enabled": True, "new": new,
                         "status": src.status(),
                         "inbox": _TETHER["inbox"][-12:]})


@app.get("/api/tether/live")
def tether_live():
    """One live-view frame, or 204 when the source has none."""
    src = _tether_source()
    frame = src.live_frame()
    if not frame:
        return Response(status_code=204)
    return Response(content=frame, media_type="image/jpeg")


@app.post("/api/tether/clear")
def tether_clear():
    _TETHER["inbox"] = []
    return JSONResponse({"ok": True})


@app.get("/api/auto_levels/{sid}")
def auto_levels(sid: str, clip: float = 0.005):
    """Level settings derived from the session's own histogram."""
    from engine import curves as CV

    s = _get_session(sid)
    return JSONResponse(CV.auto_levels(s["preview"], clip=clip))


class AutoRequest(BaseModel):
    session: str
    white_balance: bool = True
    exposure: bool = True
    contrast: bool = True
    method: str = "shades"        # shades | grey | maxrgb
    strength: float = 1.0
    # apply to the working image (and record an undo step) instead of only
    # returning a suggestion
    commit: bool = True


@app.post("/api/auto")
def auto_fix(req: AutoRequest):
    """One-click white balance + exposure + contrast.

    The response carries both the corrected image and the parameters that
    produced it, so the UI can show which sliders moved rather than silently
    changing the frame.
    """
    from engine import autofix as AF

    s = _get_session(req.session)
    if req.method not in ("shades", "grey", "maxrgb"):
        raise HTTPException(400, "method must be shades, grey or maxrgb")

    params = AF.suggest(s["preview"], wb=req.white_balance,
                        exposure=req.exposure, contrast=req.contrast,
                        method=req.method, strength=req.strength)
    out = AF.apply(s["preview"], params)
    if req.commit:
        _commit(s, out, "自动校正")
    # strip the internal gain key from the returned params
    shown = {k: v for k, v in params.items() if not k.startswith("_")}
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "params": shown,
        "gains": params.get("_gains"),
        "ev": params.get("_ev"),
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
    })


class BrushRequest(BaseModel):
    session: str
    mode: str = "heal"            # heal | clone
    points: List[List[float]]     # stroke path in image pixels
    source: List[float]           # source point for the stroke anchor
    radius: int = 40
    hardness: float = 0.65
    opacity: float = 1.0
    low_freq: float = 0.5
    commit: bool = True


@app.post("/api/brush")
def brush_stroke(req: BrushRequest):
    """Apply a clone/heal stroke to the session's working image."""
    from engine import brush as BR

    s = _get_session(req.session)
    if req.mode not in ("heal", "clone"):
        raise HTTPException(400, "mode must be heal or clone")
    if len(req.points) < 1 or len(req.source) < 2:
        raise HTTPException(400, "points and source are required")

    out = BR.stroke(s["preview"], req.points, req.source,
                    radius=int(max(2, req.radius)),
                    hardness=float(req.hardness),
                    opacity=float(req.opacity),
                    mode=req.mode,
                    low_freq=float(req.low_freq))
    if req.commit:
        _commit(s, out, ("修复画笔" if req.mode == "heal" else "仿制图章"))
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 92)).decode("ascii"),
        "committed": bool(req.commit),
        "mode": req.mode,
    })


class BatchRequest(BaseModel):
    # a directory, a single file, or an explicit list; `path` wins when set
    path: Optional[str] = None
    paths: List[str] = []
    out_dir: str
    # recipe: exactly one of these
    preset: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    formula: Optional[str] = None
    # 0 = full resolution; a positive value is the draft long edge
    long_edge: int = 0
    recursive: bool = False
    # write next to the source instead of into out_dir
    overwrite: bool = False
    quality: int = 95
    # ---- 导出后处理（2026-10-26）----
    watermark_text: str = ""          # 文字水印（与 logo 二选一或同用）
    watermark_logo: str = ""          # Logo 图片路径
    watermark_pos: str = "br"         # tl/tc/tr/ml/mc/mr/bl/bc/br
    watermark_opacity: float = 0.65
    naming: str = "{原名}_out"        # {原名}/{序号3}/{日期}/{时间}/{预设}
    size_w: int = 0                   # 尺寸模板：0 = 原尺寸
    size_h: int = 0


@app.post("/api/batch")
def batch_start(req: BatchRequest):
    """Start a batch run and return its id immediately.

    The work happens on a background thread and is polled through
    `/api/batch/{id}`, so a folder of two hundred frames does not hold an HTTP
    request open for half an hour.
    """
    from engine import batch as BT
    from engine.imgutil import load_image, encode_jpeg
    from engine.pipeline import apply_preset

    paths: List[str] = []
    if req.path:
        paths = BT.list_images(req.path, recursive=req.recursive)
    elif req.paths:
        for p in req.paths:
            paths.extend(BT.list_images(p, recursive=req.recursive))
    if not paths:
        raise HTTPException(400, "no processable files found")

    # resolve the recipe once, so every file is guaranteed the same look
    label = "自定义参数"
    fixed: Optional[dict] = None
    if req.preset:
        fixed = apply_preset(req.preset)
        label = "预设 " + req.preset
    elif req.formula:
        from engine import formula as formula_engine
        lib = formula_engine.load_library()
        if req.formula not in lib:
            raise HTTPException(404, "formula not found: " + req.formula)
        fixed = None            # resolved per file below
        label = "配方 " + req.formula
    elif req.params is not None:
        fixed = dict(req.params)

    q = int(max(50, min(100, req.quality)))

    def params_for(path: str) -> dict:
        if req.formula:
            return {"__formula__": req.formula}
        return fixed or {}

    seq = [0]

    def save_fn(path: str, params: dict) -> str:
        import cv2 as _cv

        from engine import exportfx as FX
        from engine import formula as formula_engine
        from engine.imgutil import fit_long_edge

        src = load_image(path)
        if req.long_edge:
            src = fit_long_edge(src, req.long_edge)
        if req.formula:
            lib = formula_engine.load_library()
            out, _info = formula_engine.run(lib[req.formula], src,
                                            pipeline=_pipe)
        else:
            out, _info = _pipe.run(src, params, detect=True)

        # 尺寸模板：等比缩放到目标框（长边贴合），透明填充不裁切
        if req.size_w and req.size_h:
            ih, iw = out.shape[:2]
            scale = min(req.size_w / iw, req.size_h / ih)
            if scale < 0.999:
                out = _cv.resize(out, (max(1, int(iw * scale)),
                                       max(1, int(ih * scale))),
                                 interpolation=_cv.INTER_AREA)
        # 水印（在最终尺寸上叠，保证位置一致）
        if req.watermark_text or req.watermark_logo:
            try:
                out = FX.apply_watermark(
                    out, text=req.watermark_text,
                    logo_path=req.watermark_logo,
                    position=req.watermark_pos,
                    opacity=req.watermark_opacity)
            except Exception as exc:
                print("watermark failed on %s: %s" % (path, exc))

        seq[0] += 1
        name = FX.build_filename(req.naming, path, seq[0],
                                 req.preset or "")
        if req.overwrite:
            dest_dir = os.path.dirname(path)
        else:
            dest_dir = req.out_dir
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name + ".jpg")
        with open(dest, "wb") as fh:
            fh.write(encode_jpeg(out, q))
        return dest

    run = BT.start(paths, req.out_dir, label, int(req.long_edge),
                   params_for, save_fn)
    return JSONResponse({"id": run.id, "total": run.total, "recipe": label,
                         "out_dir": req.out_dir})


@app.get("/api/batch/{run_id}")
def batch_status(run_id: str):
    from engine import batch as BT

    run = BT.get(run_id)
    if run is None:
        raise HTTPException(404, "batch run not found")
    return JSONResponse(run.snapshot())


@app.post("/api/batch/{run_id}/cancel")
def batch_cancel(run_id: str):
    from engine import batch as BT

    run = BT.get(run_id)
    if run is None:
        raise HTTPException(404, "batch run not found")
    run.cancel()
    return JSONResponse(run.snapshot())


@app.post("/api/cull")
def cull(req: CullRequest):
    """Score and rank a set of photos, flagging near-duplicates.

    Sessions are read from the in-memory store; `paths` are read from disk.
    Both may be supplied together.
    """
    from engine import match as MT
    from engine import batch as BT
    from engine.imgutil import load_image, fit_long_edge

    images, labels = [], []
    for sid in req.sessions:
        s = _get_session(sid)
        images.append(s["preview"])
        labels.append(sid)
    # A caller may hand over a directory —the batch panel does, and it is the
    # natural way to ask "score this shoot". Reading one as a file raised
    # PermissionError and surfaced as a bare 500; expanding it is what the
    # caller meant, and a path that is neither file nor directory now says so.
    for p in req.paths:
        if os.path.isdir(p):
            found = BT.list_images(p, recursive=False)
            if not found:
                raise HTTPException(400, "no images in directory: " + p)
            for fp in found:
                images.append(fit_long_edge(load_image(fp), 900))
                labels.append(fp)
            continue
        if not os.path.exists(p):
            raise HTTPException(404, "image not found: " + p)
        if not os.path.isfile(p):
            raise HTTPException(400, "not a file or directory: " + p)
        images.append(fit_long_edge(load_image(p), 900))
        labels.append(p)

    if not images:
        raise HTTPException(400, "provide sessions or paths")

    scores = MT.score_photos(images, labels, use_embeddings=bool(req.duplicates))
    ranked = MT.rank(scores)

    # A thumbnail per result, so the UI can show a contact sheet instead of a
    # text list —picking frames is a visual decision. Kept small (160px) and
    # capped in count, since this rides in one JSON response.
    thumbs = {}
    if req.thumbs:
        for lab, im in zip(labels, images):
            if len(thumbs) >= req.max_thumbs:
                break
            small = fit_long_edge(im, 160)
            thumbs[lab] = base64.b64encode(encode_jpeg(small, 72)).decode("ascii")

    return JSONResponse({
        "count": len(ranked),
        "best": ranked[0].path if ranked else None,
        "results": [{"path": s.path, **s.as_dict(),
                     **({"thumb": thumbs[s.path]} if s.path in thumbs else {})}
                    for s in ranked],
    })


class FillRequest(BaseModel):
    session: str
    # mask as base64 PNG (single channel in R), same size as the preview
    mask_png: str
    prompt: str = "clean background, seamless, high quality"
    negative: str = "blurry, artifacts, watermark, text, deformed"
    steps: int = 20
    cfg: float = 7.0
    denoise: float = 1.0
    controlnet: float = 1.0
    seed: Optional[int] = None


@app.post("/api/fill")
def fill(req: FillRequest):
    """Generative fill: synthesise the masked region via local ComfyUI."""
    import base64 as _b64
    import cv2 as _cv2
    from engine import comfy_mgr
    from engine import genfill

    if not genfill.available():
        # 懒启动 + 明确告知（与 /api/gen 同一套语义）
        comfy_mgr.start(background=True)
        raise HTTPException(503, "AI 服务正在启动，请稍候片刻再试")
    s = _get_session(req.session)
    img = s["preview"]

    data = _b64.b64decode(req.mask_png)
    arr = np.frombuffer(data, np.uint8)
    bgr = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "mask_png did not decode")
    mask = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if mask.shape[:2] != img.shape[:2]:
        mask = _cv2.resize(mask, (img.shape[1], img.shape[0]),
                           interpolation=_cv2.INTER_AREA)

    try:
        out, info = genfill.fill(
            img, mask, prompt=req.prompt, negative=req.negative,
            steps=req.steps, cfg=req.cfg, denoise=req.denoise,
            controlnet_strength=req.controlnet, seed=req.seed)
    except Exception as exc:
        raise HTTPException(502, f"fill failed: {exc}")

    # the filled preview becomes the session's working image, so subsequent
    # renders build on it; the original full-res file is untouched
    _commit(s, out, "生成填充")
    comfy_mgr.touch()
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 90)).decode("ascii"),
        "info": info,
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
    })


@app.get("/api/fill/status")
def fill_status():
    from engine import genfill
    return JSONResponse({"available": genfill.available(),
                         "server": genfill.SERVER,
                         "checkpoint": genfill.CHECKPOINT})


class AiServiceAction(BaseModel):
    action: str            # "start" | "stop"
    wait: bool = False     # start 时同步等就绪（最长 5 分钟）


@app.get("/api/ai/service")
def ai_service_status():
    """AI 服务（ComfyUI）状态：面板状态条的数据源。

    ready/starting/comfy_dir 一次给全，前端据此渲染「就绪/启动中/
    未安装」三态，不再让用户对着报错猜。
    """
    from engine import comfy_mgr
    return JSONResponse(comfy_mgr.status())


@app.post("/api/ai/service")
def ai_service_action(req: AiServiceAction):
    """启动/停止 ComfyUI。启动默认异步（立即返回 starting），
    前端轮询 GET 到 ready；wait=True 供脚本化场景同步等待。"""
    from engine import comfy_mgr
    if req.action == "start":
        return JSONResponse(comfy_mgr.start(background=not req.wait))
    if req.action == "stop":
        return JSONResponse(comfy_mgr.stop())
    raise HTTPException(400, "action 必须是 start 或 stop")


@app.get("/api/models")
def models_status():
    """模型安装状态（瘦发行版首次初始化的数据源）。"""
    from engine import models_setup
    try:
        return JSONResponse(models_setup.status())
    except Exception as exc:
        return JSONResponse({"phase": "error", "error": str(exc)[:200]},
                            status_code=500)


@app.post("/api/models")
def models_fetch():
    """开始/继续下载缺失模型（后台线程），进度走 GET 轮询。"""
    from engine import models_setup
    try:
        return JSONResponse(models_setup.fetch())
    except Exception as exc:
        return JSONResponse({"phase": "error", "error": str(exc)[:200]},
                            status_code=500)


@app.get("/api/sky/library")
def sky_library():
    """天空/背景素材库：程序生成，含缩略图（换天空面板的可视化选择）。"""
    import base64 as _b64

    from engine import sky as SK
    from engine.imgutil import encode_jpeg
    items = []
    for key, label, top, low, sun, clouds in SK.PRESETS:
        th = SK._sky_texture(96, 160, top, low, sun, clouds,
                             seed=abs(hash(key)) % 9999)
        items.append({"id": key, "name": label,
                      "thumb": _b64.b64encode(
                          encode_jpeg(th, 82)).decode("ascii")})
    return JSONResponse({"skies": items})


# ---- 图库星级（选片结论：AI 排序之后由人拍板） -------------------------
_LIB_STARS_FILE = os.path.join(DATA_DIR, "_library_stars.json")


def _lib_stars() -> Dict[str, int]:
    try:
        with open(_LIB_STARS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _lib_stars_save(data: Dict[str, int]) -> None:
    tmp = _LIB_STARS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, _LIB_STARS_FILE)


@app.get("/api/library/stars")
def library_stars():
    return JSONResponse({"stars": _lib_stars()})


@app.post("/api/library/star")
def library_set_star(req: dict = Body(...)):
    """src -> 星级（0-5，0 = 清除）。"""
    src = os.path.abspath(str(req.get("src") or ""))
    value = int(req.get("value", 0))
    if not src or not (0 <= value <= 5):
        raise HTTPException(400, "src 必填, value 0-5")
    data = _lib_stars()
    if value:
        data[src] = value
    else:
        data.pop(src, None)
    _lib_stars_save(data)
    return JSONResponse({"ok": True, "total": len(data)})


class ObjectPickRequest(BaseModel):
    session: str
    x: float
    y: float


@app.post("/api/remove/object")
def remove_object_pick(req: ObjectPickRequest):
    """点选物体：一个种子点 → grabCut 物体掩膜（给智能消除当 mask）。

    返回 PNG（R 通道即掩膜，同 /api/fill 契约）；覆盖率异常时
    empty=true，UI 提示改用手涂。
    """
    import cv2 as _cv

    from engine import objectpick as OP
    s = _get_session(req.session)
    img = s["preview"]
    mask = OP.pick(img, req.x * img.shape[1], req.y * img.shape[0])
    cov = float((mask > 0.4).mean())
    ok, png = _cv.imencode(".png",
                           (np.clip(mask, 0, 1) * 255).astype(np.uint8))
    return JSONResponse({
        "empty": cov < 0.001,
        "coverage": round(cov, 4),
        "mask_png": base64.b64encode(png).decode("ascii") if ok else "",
    })


class IdSheetRequest(BaseModel):
    session: str
    spec: str = "one_inch"
    paper: str = "six_inch"       # six_inch(1800x1200) | five_inch(1500x1050)
    gap: int = 6                  # 张与张的间隙 px（裁切余量）


@app.post("/api/idphoto/sheet")
def idphoto_sheet(req: IdSheetRequest):
    """证件照联排版：把当前证件照平铺到整张相纸 + 裁切角标。

    输入是已按规格生成的证件照（先跑 /api/idphoto 并 commit），
    这里只做排版——相纸 6寸/5寸 @300dpi，白底，照片按行列铺满，
    每张四角画 2px 灰角标，照相馆/打印店照着裁。
    """
    import cv2 as _cv

    from engine import idphoto as IP
    from engine.imgutil import encode_jpeg

    papers = {"six_inch": (1800, 1200), "five_inch": (1500, 1050)}
    pw, ph = papers.get(req.paper, papers["six_inch"])
    s = _get_session(req.session)
    try:
        photo, info = IP.make_id_photo(s["preview"], req.spec,
                                       (1.0, 1.0, 1.0))
    except Exception as exc:
        raise HTTPException(400, "先在证件照面板生成证件照: " + str(exc)[:80])
    ih, iw = photo.shape[:2]
    g = int(req.gap)
    cols = max(1, (pw + g) // (iw + g))
    rows = max(1, (ph + g) // (ih + g))
    sheet = np.ones((ph, pw, 3), np.float32)
    y = (ph - (rows * ih + (rows - 1) * g)) // 2
    count = 0
    for r in range(rows):
        x = (pw - (cols * iw + (cols - 1) * g)) // 2
        for c in range(cols):
            sheet[y:y + ih, x:x + iw] = photo
            for cx, cy, dx, dy in ((0, 0, 1, 1), (iw, 0, -1, 1),
                                   (0, ih, 1, -1), (iw, ih, -1, -1)):
                _cv.line(sheet, (x + cx, y + cy),
                         (x + cx + dx * 10, y + cy), (0.6, 0.6, 0.6), 1)
                _cv.line(sheet, (x + cx, y + cy),
                         (x + cx, y + cy + dy * 10), (0.6, 0.6, 0.6), 1)
            x += iw + g
            count += 1
        y += ih + g
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(sheet, 95)).decode("ascii"),
        "cols": cols, "rows": rows, "count": count,
        "paper": "%dx%d" % (pw, ph),
    })


class SkinUnifyRequest(BaseModel):
    ref_session: str
    paths: List[str] = []
    out_dir: str
    strength: float = 0.8
    quality: int = 95


@app.post("/api/skin/unify")
def skin_unify(req: SkinUnifyRequest):
    """套图肤色统一：以 ref_session 的肤色为基准，整批对齐。

    只动皮肤掩膜内（Lab 线性统计对齐，a/b 色相权重高于 L），
    背景/衣服逐像素不变。输出写 out_dir，文件名 = 原名_skin。
    """
    from engine import batch as BT
    from engine import skintone as ST
    from engine.imgutil import encode_jpeg, fit_long_edge, load_image

    s = _get_session(req.ref_session)
    ref_stats = ST.stats(fit_long_edge(s["preview"], 1400))
    paths = [p for p in req.paths if os.path.isfile(p)]
    if not paths:
        raise HTTPException(400, "paths 为空或文件不存在")

    os.makedirs(req.out_dir, exist_ok=True)
    q = int(max(50, min(100, req.quality)))
    done = 0
    for path in paths:
        try:
            img = load_image(path)
            out = ST.unify(fit_long_edge(img, 2400), ref_stats,
                           float(req.strength))
            base = os.path.splitext(os.path.basename(path))[0]
            dest = os.path.join(req.out_dir, base + "_skin.jpg")
            with open(dest, "wb") as fh:
                fh.write(encode_jpeg(out, q))
            done += 1
        except Exception as exc:
            return JSONResponse({"ok": False, "done": done,
                                 "error": "%s: %s" % (path, str(exc)[:120])},
                                status_code=500)
    return JSONResponse({"ok": True, "done": done, "total": len(paths),
                         "out_dir": req.out_dir})


# --------------------------------------------------------------------------
# formula (recipe) engine: stored, composable, parametric edit sequences
# --------------------------------------------------------------------------

BUILTIN_FORMULAS = {
    "skin_prep": {
        "name": "skin_prep",
        "description": "基础美肤：祛痁+ 匀肁+ 轻磨皮，保留质感",
        "steps": [
            {"kind": "params", "params": {
                "spotless_acne": 55, "spotless_spot": 35,
                "spotless_texture": 40, "even_tone": 30, "smooth": 32,
                "texture": 70}},
        ],
    },
    "clean_colour": {
        "name": "clean_colour",
        "description": "干净调色：轻暖色温 + 提亮阴影 + 鲜艳度",
        "steps": [
            {"kind": "params", "params": {
                "temp": 5, "exposure": 5, "shadows": 14, "vibrance": 12,
                "contrast": 6}},
        ],
    },
    "cinematic": {
        "name": "cinematic",
        "description": "电影感：冷中间调 + 暖高光（三区颜色分级）",
        "steps": [
            {"kind": "params", "params": {
                "grade_mid_hue": 172, "grade_mid_sat": 22,
                "grade_high_hue": -28, "grade_high_sat": 18,
                "grade_balance": -10}},
        ],
    },
    "corner_vignette": {
        "name": "corner_vignette",
        "description": "四角压暗（径向局部，可调强度）",
        "steps": [
            {"kind": "local", "strength": "{{strength}}", "entries": [
                {"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.62,
                 "ry": 0.62, "invert": True, "exposure": -28,
                 "saturation": -18}]},
        ],
    },
    "portrait_full": {
        "name": "portrait_full",
        "description": "组合：美肁；干净调色 ；暗角（演示配方组合）",
        "steps": [
            {"kind": "formula", "formula": "skin_prep"},
            {"kind": "formula", "formula": "clean_colour"},
            {"kind": "formula", "formula": "corner_vignette",
             "slots": {"strength": 70}},
        ],
    },
}


class FormulaUpsertRequest(BaseModel):
    formula: Dict[str, Any]


class FormulaRunRequest(BaseModel):
    name: str
    session: str
    slots: Dict[str, Any] = {}


@app.get("/api/formulas")
def formulas_list():
    """Builtin recipes plus everything stored in formulas.json."""
    lib = dict(BUILTIN_FORMULAS)
    lib.update(formula_engine.load_library())
    return JSONResponse({
        "count": len(lib),
        "formulas": [{"name": n,
                      "description": f.get("description", ""),
                      "steps": len(f.get("steps", []))}
                     for n, f in sorted(lib.items())],
        "library": lib,
    })


@app.post("/api/formulas/save")
def formulas_save(req: FormulaUpsertRequest):
    try:
        lib = formula_engine.upsert(req.formula)
    except formula_engine.FormulaError as exc:
        raise HTTPException(400, str(exc))
    return JSONResponse({"ok": True, "count": len(lib)})


@app.post("/api/formulas/delete")
def formulas_delete(req: FormulaUpsertRequest):
    name = (req.formula or {}).get("name", "")
    if name in BUILTIN_FORMULAS:
        raise HTTPException(400, "builtin formulas cannot be deleted")
    lib = formula_engine.delete(name)
    return JSONResponse({"ok": True, "count": len(lib)})


@app.post("/api/formulas/run")
def formulas_run(req: FormulaRunRequest):
    """Run a formula against a session; the result becomes the new preview."""
    lib = dict(BUILTIN_FORMULAS)
    lib.update(formula_engine.load_library())
    if req.name not in lib:
        raise HTTPException(404, f"formula {req.name!r} not found")

    s = _get_session(req.session)

    def fill_runner(image, mask, prompt):
        from engine import genfill
        if not genfill.available():
            raise HTTPException(503, "generative fill needs ComfyUI")
        out, _ = genfill.fill(image, mask, prompt=prompt)
        return out

    try:
        out, info = formula_engine.run(lib[req.name], s["preview"],
                                       pipeline=_pipe, slots=req.slots,
                                       library=lib, fill_runner=fill_runner)
    except formula_engine.FormulaError as exc:
        raise HTTPException(400, str(exc))

    _commit(s, out, "配方 " + req.name)
    return JSONResponse({
        "image": base64.b64encode(encode_jpeg(out, 90)).decode("ascii"),
        "applied": info["applied"],
        "slots": info["slots"],
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
    })
