# 生成式组件（可选）安装指南

ZhaJinWenQianCake 的三个 AI 功能——**AI 布景**、**创成式填充**、**道具修复**——
不随主程序发行。它们需要一个外挂的 ComfyUI 服务（默认
`http://127.0.0.1:8388`）和两个模型文件，合计约 13 GB 下载量。
对不用这三个功能的用户，主程序完全不受影响。

## 日常使用（装好之后，零命令）

**不需要任何手动启动**：打开「AI 布景 / 创成式填充 / 道具修复」面板——

- 状态条显示「● AI 服务就绪」：直接用
- 显示「○ AI 服务未运行」：点**「启动 AI 服务」**按钮，等 1-2 分钟
- 直接点「生成」也行：AI 服务会自动启动，界面显示进度并在就绪后
  自动重试
- 空闲 15 分钟自动停止（还回约 8GB 内存），下次用再自动拉起

状态条显示「AI 组件未安装」时，才需要看下面的首次安装。

## 首次安装（一次性）

```powershell
cd C:\Users\Rana\ZhaJinWenQianCake     # 或安装目录
python tools\install_generative.py install
```

`install` 会下载：ComfyUI 便携版（含独立 Python，不污染系统环境，
约 1.5 GB）→ SD1.5 inpainting 模型（2.0 GB）→ inpaint ControlNet
（0.7 GB），全部来自官方发布源（GitHub Release / HuggingFace）。
装到 `C:\ComfyUI`，可用 `--dir D:\ComfyUI` 换位置，已有程序或模型
会自动跳过。装完回到界面点「启动 AI 服务」即可。

> 打包发行版的用户如果装过 ComfyUI 便携版（或让维护者帮装一次），
> 之后的一切启动/停止都由界面按钮完成，不需要 Python。

## 体检 / 手动管理（可选）

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
3. 在 `data\generative.json` 里写 `{"comfy_dir": "你的ComfyUI目录"}`，
   或设环境变量 `RETOUCHLAB_COMFYUI_DIR`。服务地址可用
   `RETOUCHLAB_COMFYUI` 覆盖（默认 `http://127.0.0.1:8388`；
   端口换过位——8188 落在本机 Hyper-V 保留端口段内会 bind 失败）。

然后用界面按钮启动即可。

## 质量预期（实测，2026-10）

- SD1.5：生成区锐度比 0.32，大区域会污染 mask 外内容
- 祛路人**不走**生成式（默认 LaMa，确定性且不发明假人）
- 多人合影的大区域删除不可靠，界面会给出 guidance 而不是硬做
- prompt 只懂英文——界面里的场景预设（白墙影棚/户外草坪/黄昏窗边…）
  已经替你写好英文，点一下就行

结论：适合小区域（道具、背景布景、局部补全），不适合"把整面墙换了"。

## 常见问题

**点了启动，等了很久一直"启动中"？**
首次启动要加载 2.7GB 模型，冷盘 1-2 分钟正常（状态条会一直转）；
超过 5 分钟看 `ComfyUI 目录\retouchlab_comfyui.log`。

**启动后 /api/health 里 generative_fill 还是 false？**
`check` 三项必须全 OK；服务注册 checkpoint 需要几秒，等 30s 再刷新。

**显存不足 / 崩溃？**
自动启动默认带 `--lowvram --disable-assets`。8GB 卡够用。

**想换更强的模型？**
`engine/genfill.py` 顶部的 `CHECKPOINT` / `CONTROLNET` 常量与
ComfyUI 里的文件名一致即可，无需改其它代码。SDXL 实测过：祛路人
会发明假人，不推荐。

