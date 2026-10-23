# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（onedir 模式）。

为什么用 onedir 而不是 onefile：
  - onefile 每次启动都要把 2~5GB 解压到临时目录，启动从 6 秒变 60 秒+
  - onedir 启动时直接映射磁盘文件，且模型目录可以原地替换

关键取舍：
  - **不打包 torch**：生成式三件套走可选组件（requirements-generative.txt）。
    打包时排除，主程序体积从 5GB+ 降到 ~1.2GB。
  - **models/ 和 web/ 不打进 _internal**：放到 exe 同级的可写目录，
    这样用户能换模型、装生成式组件、程序也能写 data/。
  - hiddenimports 里列出的是"运行期动态 import"的模块，静态分析找不到。
"""
import os

ROOT = os.path.abspath(os.path.dirname(SPEC))          # noqa: F821
HERE = ROOT

# ---- 收集数据文件 ----
datas = [
    (os.path.join(HERE, "web"), "web"),
]

# ---- 明确排除的大件 ----
excludes = [
    # **只排除生成式推理栈**，不排除 torch 本身：
    #   superres（超清）、nndenoise（AI 降噪）、faceparse（面部解析）、
    #   grading（3D LUT）都直接跑 torch 模型，是核心功能。
    #   之前的实测结论是"排掉 torch 主程序能瘦 1GB"，但代价是这四个
    #   功能全部失效，不划算。
    "diffusers", "accelerate", "torchsde", "triton", "onnxscript",
    "transformers", "safetensors", "tokenizers",
    # huggingface_hub 保留：transformers 缺席时它只是一堆轻量 HTTP 工具，
    # 而且 match.py 的降级路径会 import 到它
    # 训练侧代码不随发行版走
    "training",
    # 常见误入包：体积大、与功能无关
    "matplotlib", "pandas", "IPython", "jupyter", "notebook",
    "pytest", "pydoc_data", "tkinter", "test",
    "PyQt5", "PyQt6", "PySide2", "PySide6",
    # torch 里只给训练/导出用的子模块
    "torch.utils.tensorboard", "torch._inductor", "torch._dynamo",
    "torch.onnx",
]

hidden = [
    # uvicorn 的协议实现是字符串动态加载的
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # onnxruntime 的 provider 注册
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    # mediapipe 的任务 API
    "mediapipe",
    "mediapipe.tasks",
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
    # rawpy
    "rawpy",
    # engine 各模块（被动态 import 的）
    "engine.autofix", "engine.automask", "engine.batch", "engine.bgclean",
    "engine.brush", "engine.cattree", "engine.cliponnx", "engine.crop",
    "engine.curves", "engine.defringe", "engine.eyes", "engine.face",
    "engine.faceparse", "engine.formula", "engine.genfill", "engine.gentools",
    "engine.geoface", "engine.grade", "engine.grading", "engine.grassfix",
    "engine.history", "engine.idphoto", "engine.imgutil", "engine.lama",
    "engine.landmarks", "engine.lgroup", "engine.library", "engine.liquify",
    "engine.local", "engine.match", "engine.matting", "engine.nndenoise",
    "engine.passersby", "engine.paths", "engine.pet", "engine.pipeline",
    "engine.pose", "engine.portrait", "engine.rawexif", "engine.rawio",
    "engine.rigfix", "engine.scene", "engine.semantic", "engine.skin",
    "engine.sky", "engine.superres", "engine.tether", "engine.tone",
    "engine.wheel", "engine.wpd",
]

a = Analysis(                                  # noqa: F821
    [os.path.join(HERE, "tools", "launch.py")],
    pathex=[HERE],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)                              # noqa: F821

exe = EXE(                                     # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RetouchLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX 会拖慢启动，也可能被杀软误报
    console=True,              # 保留控制台：用户要能关服务、看错误
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(                                # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RetouchLab",
)
