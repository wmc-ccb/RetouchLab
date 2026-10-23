# 生成式组件（可选）安装指南

RetouchLab 的三个 AI 功能——**AI 布景**、**创成式填充**、**道具修复**——
不随主程序发行。它们需要一个外挂的 ComfyUI 服务（默认
`http://127.0.0.1:8388`）和两个模型文件，合计约 13 GB 下载量。
对不用这三个功能的用户，主程序完全不受影响。

## 一句话安装（推荐）

```powershell
cd C:\Users\Rana\RetouchLab     # 或安装目录
python tools\install_generative.py install
python tools\install_generative.py start
```

`install` 会下载：ComfyUI 便携版（含独立 Python，不污染系统环境，
约 1.5 GB）→ SD1.5 inpainting 模型（2.0 GB）→ inpaint ControlNet
（0.7 GB），全部来自官方发布源（GitHub Release / HuggingFace）。
装到 `C:\ComfyUI`，可用 `--dir D:\ComfyUI` 换位置，已有程序或模型
会自动跳过。

## 只想体检 / 手动启动

```powershell
python tools\install_generative.py check    # 缺什么报什么
python tools\install_generative.py start    # 后台启动并等就绪
python tools\install_generative.py stop     # 停掉 8388 服务
```

`check` 的输出就是排障清单：ComfyUI 目录、两个模型文件、服务应答，
逐项 OK / 缺失。启动记录在 `data\generative.json`。

## 已有 ComfyUI 的机器

1. 把 inpaint 主模型放进 `ComfyUI\models\checkpoints\`，
   文件名必须是 **`sd15_inpaint.ckpt`**
2. 把 inpaint ControlNet 放进 `ComfyUI\models\controlnet\`，
   文件名必须是 **`control_v11p_sd15_inpaint_fp16.safetensors`**
3. 让 ComfyUI 监听 `127.0.0.1:8388` 启动（默认 8188 不行；
   端口换过位——8188 落在本机 Hyper-V 保留端口段内会 bind 失败）。
   服务地址也可用环境变量 `RETOUCHLAB_COMFYUI` 覆盖。

然后运行 `check` 确认三项全 OK，回到 RetouchLab 刷新页面即可。

## 质量预期（实测，2026-10）

- SD1.5：生成区锐度比 0.32，大区域会污染 mask 外内容
- 祛路人**不走**生成式（默认 LaMa，确定性且不发明假人）
- 多人合影的大区域删除不可靠，界面会给出 guidance 而不是硬做

结论：适合小区域（道具、背景布景、局部补全），不适合"把整面墙换了"。

## 常见问题

**启动后 /api/health 里 generative_fill 还是 false？**
`check` 三项必须全 OK；服务注册 checkpoint 需要几秒，等 30s 再刷新。

**显存不足 / 崩溃？**
`start` 默认带 `--lowvram`。8GB 卡够用，再低可以自己改启动命令
（`python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build
--lowvram --listen 127.0.0.1 --port 8388`）加 `--novram`。

**想换更强的模型？**
`engine/genfill.py` 顶部的 `CHECKPOINT` / `CONTROLNET` 常量与
ComfyUI 里的文件名一致即可，无需改其它代码。SDXL 实测过：祛路人
会发明假人，不推荐。
