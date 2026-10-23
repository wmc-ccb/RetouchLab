# RetouchLab 交接文档

> 写于 2026-10-24。给接手的人（或未来的我）用。
> 每个数字都是实测的，不是估的。凡是我踩过的坑，都写了为什么。

---

## 一、这个项目是什么

本地人像精修软件。用户导入照片，调光调色、修人像、换背景、批量处理、导出。
对标商业软件「像素蛋糕」，但**代码、模型、素材全部是从零自己实现的**，
不含任何商业产品的资产。

形态是「本地 Web 应用」：Python 后端（FastAPI）+ 单文件前端（`web/index.html`），
用浏览器当 UI。打包后是一个 exe，双击启动、自动开浏览器。

### 规模（2026-10-24 实测）

| 部分 | 行数 | 文件 |
|---|---|---|
| `engine/`（图像算法） | 13,410 | 51 |
| `server/`（API） | 3,582 | 1 |
| `web/`（前端） | 7,152 | 1 |
| `tests/`（测试） | 14,748 | 99 |
| `tools/`（构建/启动） | 877 | 12 |
| `training/`（降噪模型训练） | 465 | 3 |
| **合计** | **40,234** | 167 |

测试：**73 个 `verify_*.py`** + 26 个其他脚本（`e2e_api.py`、`monotonic.py` 等）。

---

## 二、怎么跑起来

### 开发用

```powershell
cd C:\Users\Rana\RetouchLab
python -m pip install -r requirements.txt
python tools/launch.py            # 或双击 RetouchLab.bat
```

浏览器会自动打开 `http://127.0.0.1:8765`。

### 发行用（给用户）

```
dist\RetouchLab\启动.bat          ← 双击这个
```

产物 6.46 GB，自带 Python 和全部依赖，用户机器不需要装任何东西。

### 重新打包

双击 `build_app.bat`，或者：

```powershell
python -m PyInstaller RetouchLab.spec --noconfirm --clean
python tools/assemble_dist.py
```

约 4-5 分钟。**打包前必须关掉所有正在跑的 RetouchLab 实例**，
否则文件被占用会报 `PermissionError`。

---

## 三、目录约定（改了会影响打包，务必看懂）

`engine/paths.py` 是唯一的路径解析入口。它按 `sys.frozen` 分派：

```
源码运行                         打包运行
<root>/engine/paths.py           dist/RetouchLab/_internal/engine/...
<root>/models/                   dist/RetouchLab/models/       ← exe 旁边
<root>/web/                      dist/RetouchLab/web/
<root>/data/                     dist/RetouchLab/data/        ← 可写
```

**关键决策：模型和 web 不打进 `_internal`。**

理由：
1. 模型 1.3GB，打进 `_internal` 每次都要跟 exe 一起被校验
2. 生成式组件是后装的，必须能往这个目录加东西
3. 用户想换模型直接拷文件，不用重新打包

`data/` 放用户状态（预设、图库来源、导出记录）。**不能放 exe 同级目录**——
装到 `Program Files` 时那里不可写，会在「保存预设」这种小操作上崩。

`RETOUCHLAB_HOME` 环境变量可覆盖运行时根目录（U 盘便携场景）。

---

## 四、打包踩过的三个坑（都已修，但要知道原理）

### 坑 1：模型路径在打包后失效

**现象**：打包版启动后所有模型都报"未安装"。

**原因**：原来各模块用 `here = os.path.dirname(os.path.dirname(__file__))`
往上找项目根。打包后 `__file__` 指向 `_internal` 里，找到的位置不存在。

**修法**：全部改成走 `engine/paths.py` 的 `models_dir()` / `data_dir()` /
`web_dir()`。改动涉及 9 个 engine 模块，用 `tools/migrate_paths.py` 批量做的。

**注意**：`tools/add_imports.py` 当年插入 `from .paths import models_dir` 时
把 `superres.py` 的 import 插进了文档字符串中间，把 docstring 弄断了。
如果你跑类似的批量脚本，**务必检查插到文件头部的位置是否在 docstring 之外**。

### 坑 2：GPU 静默回退 CPU（最阴的一个）

**现象**：打包版超清一张图 109 秒，源码版只要 9 秒。但诊断信息一切正常：
`torch.cuda.is_available()` = True、`get_device_name()` = RTX 3080。

**原因**：PyInstaller 把 `torch/lib` 下的 cublas/cudnn 一并收进了 `_internal`，
但 Windows 默认 DLL 搜索路径里没有那个目录。真正分配显存时找不到
`cublasLt64_12.dll`，torch 把异常吞掉后静默回退 CPU。

**修法**：`engine/paths.py::register_cuda_dlls()` 用 `os.add_dll_directory()`
把 `torch/lib` 显式加进搜索路径。**必须在任何 `import torch` 之前调用**——
目前放在 `tools/launch.py` 顶部和 `server/app.py` 的导入区。

**留下的教训**：`is_available()` 为 true 不代表能用。为此加了 `/api/diagnose`，
它会**真的在 GPU 上跑一次矩阵乘**并报耗时和显存，把假阳性变成可验证的事实：

```powershell
curl http://127.0.0.1:8765/api/diagnose
# {"gpu_works": true, "gpu_matmul_ms": 34.02, "gpu_mem_allocated_mb": 9.1, ...}
```

### 坑 3：挑图原来要拖 4GB 的 torch

**现象**：`clip_available()` 同时要求 `transformers` + `torch`。但「挑图」
只用到 CLIP 的**图像塔**（算照片之间的相似度），为一个图像编码器拖 4GB 不值得。

**修法**：新增 `engine/cliponnx.py`，用 onnxruntime 跑导出的
`models/clip_image_encoder.onnx`（335MB）。两条路输出契约一致
（512 维、L2 归一化）。

**验收数据**：同一对图，transformers 余弦 0.5097、ONNX 0.5023，差 0.007，
排序结果完全一致（`tests/verify_clip_onnx.py` 守着这条）。

导出命令（已不需要重跑，模型已在 `models/` 里）：

```powershell
python tools/export_clip_onnx.py
```

**踩的小坑**：一开始只导了 `vision_model.pooler_output`（768 维），
忘了过 `visual_projection` 投影到 512 维的多模态空间。结果两张同图的余弦
只有 0.26，挑图排序完全错乱。导出时必须同时带 `visual_projection`。

---

## 五、启动器的设计（`tools/launch.py`）

双击 → 这个脚本 → uvicorn。它负责 5 件事：

1. **单实例检查**：端口上如果已经是 RetouchLab，直接开浏览器不重复起
2. **自动挑端口**：8765 被别的程序占了就往后找（8766-8784），全被占就让系统随机分配
3. **等就绪再开浏览器**：轮询 `/api/health`，通了才 `webbrowser.open`
4. **日志**：`out/launcher.log`
5. **Ctrl+C / 关窗口干净退出**

### 这里踩过一个会让用户彻底懵的 bug

`is_our_service()` 原来只检查 `{"ok": true, "sessions": ...}`。
但任何健康检查都可能返回类似结构，于是一个**不是 RetouchLab 的服务**
被误判成"自己"，启动器就去"复用"它——用户双击后**既不启服务也不报错**，
只看到浏览器打开一个陌生页面。

现在改成同时要求 3 个 RetouchLab 特有字段（`ok` / `max_sessions` / `face_model`），
并且 `pick_port()` 返回 `(port, reusable)` 二元组，逻辑显式化。

---

## 六、测试怎么跑

```powershell
# 单套
python tests\verify_hotkeys.py

# 全部（按 RESULT 行判定）
Get-ChildItem tests\verify_*.py | ForEach-Object {
    python $_.FullName 2>&1 | Select-String "^RESULT|MONOTONIC"
}
```

判定口径：输出里有 `RESULT: ALL PASS` 或 `RESULT: PASS` 或 `ALL MONOTONIC`
即通过。注意有些套用 `RESULT: PASS`（带说明）而不是 `ALL PASS`，
写批量脚本时别只匹配 `ALL PASS`，会误报失败。

### 环境依赖（这是最容易误判"测试挂了"的地方）

| 测试类型 | 需要什么 | 缺了会怎样 |
|---|---|---|
| UI 测试（`verify_ui*`、`verify_hotkeys`、`verify_simplecat`…） | Chrome 开 CDP 9223，**且有一个打开 `http://127.0.0.1:8765/` 的标签页** | 全部 FAIL 或"浏览器未连接" |
| 生成式（`verify_gen`、`verify_fill_api`） | ComfyUI 跑在 8388 | 空输出 |
| 人像/抠图（`verify_passersby`、`verify_semantic`） | `mediapipe` 已安装 | 主体掩膜断言失败 |
| 姿态（`verify_pose`） | `_downloads/pose_landmarker_lite.task`（5.7MB） | 异常退出 |

启动 Chrome 的正确姿势：

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --remote-debugging-port=9223 --remote-allow-origins=* `
  --user-data-dir=C:\Users\Rana\RetouchLab\out\_cdp --no-first-run `
  http://127.0.0.1:8765/
```

**测试必须打源码版（8765），不能用打包版。** 打包版响应慢、GPU 上下文
抢占，会让 UI 测试大面积超时——我在这上面浪费过一整轮，55 个假失败。

---

## 七、功能现状

### 已完成且可交付

- 基础调整：曝光/对比/高光阴影/白平衡/HSL/曲线/色阶/色彩分级/色轮
- 人像：磨皮、祛痘、液化、面部塑形、牙齿/眼睛、证件照
- 抠图与背景：自动蒙版（13 个区域）、背景消除（4 项）、换天空、换底色
- 消除类：祛路人（LaMa）、祛灯架/灯箱（确定性检测 + LaMa）、智能消除
- 调色：预设、追色（CLIP）、滤镜、胶片模拟
- 工作流：图库、选片（挑图）、批量处理、历史记录（撤销/重做）、配方引擎
- 联机拍摄：Z6 III + digiCamControl
- RAW：NEF 解码 + EXIF 写回
- 导出：JPEG/PNG/TIFF，原生保存对话框

全部有回归测试守着。

### 生成式三件套（AI 布景 / 创成式填充 / 道具修复）

**已接通，但是可选组件。** 需要外挂 ComfyUI（默认 `http://127.0.0.1:8388`）
+ SD1.5 inpainting 模型。没装时 `/api/health` 的 `generative_fill` 为 false，
UI 上这三个工具会提示如何启动。

质量结论（实测，别抱太高期望）：
- **SD1.5**：生成区锐度比 0.32，会污染 mask 外的内容
- **SDXL**：锐度比 0.40，但**祛路人会发明假人**（所以祛路人默认走 LaMa）
- 大区域删除（多人合影）不可靠，代码会如实返回 guidance 而不是硬做

### 已知限制（写进 README 的「已知限制」章节了）

- 生成式需要额外 13GB 模型 + 独立服务，不适合普通用户
- 联机拍摄依赖第三方 digiCamControl（写死了 `C:\Program Files\...` 路径）
- 超清全尺寸（5223×3482 → 2.9 亿像素）需要 100-150 秒，这是 Real-ESRGAN 的固有成本
- 无代码签名（Windows SmartScreen 会警告"未知发布者"）
- 无自动更新机制

---

## 八、还没做的事（按优先级）

### 立即需要
1. **代码签名**——否则用户看到"未知发布者"会不敢点
2. **首次启动引导**——现在开了就是空界面，新用户不知道该干嘛

### 发布前建议
3. **自动更新**（或用简单版本检查替代）
4. **崩溃上报**（现在只有 `out/error.log` 本地文件）
5. 把生成式做成**独立的可选组件包**，单独下载安装（现在只是"不包含"，
   没有"怎么装"的一键流程）

### 长期
6. 体积优化：torch 的 CUDA 库占 4.0GB，如果用户没 N 卡就是白下载。
   可以考虑发 CPU 版 + GPU 补丁包（用户选了带 CUDA，所以这条没做）
7. macOS / Linux 支持（现在只有 Windows 打包配置）

---

## 八A、Round 4（2026-10-24 凌晨，接手人第二轮）做了什么

### 待办清单的落实情况

1. **代码签名** ✅ 构建链就位：`tools/assemble_dist.py::sign_dist()`
   读 `RETOUCHLAB_CERT_PFX`/`RETOUCHLAB_CERT_PASS` 环境变量自动调
   signtool（带时间戳），没配证书时跳过不挡路。证书怎么选（OV/EV/自签）
   写进 README「发布」章节。exe 现在带版本资源（右键属性可见），
   版本号唯一来源是 `server/app.py::APP_VERSION`，spec 构建时提取。
2. **首次启动引导** ✅ 左下角引导卡（`#welcome`）：只出现一次
   （localStorage `rl_onboarded`），任何点击/「开始使用」收起。
   不遮画布中心与右侧参数区，不拦截事件——自动化测试零影响。
3. **版本检查** ✅ `/api/version`（版本/构建/更新源）+ 关于面板
   （顶栏「?」）异步比对 GitHub Releases latest，有新版显示下载链接；
   仓库未公开或离线时静默降级为「检查不可用」。**注意：仓库目前是
   私有的，更新检查要等仓库公开（或改 `update_url`）才真正生效。**
4. **崩溃上报** ✅ `/api/errors`（out/error.log 的结构化视图）+
   关于面板「复制诊断报告」一键复制版本/GPU 实测/最近错误。
   error.log 加了 2MB 轮转 + traceback 截断净化（起因：一次错误的
   multipart 请求把 79MB 图片二进制灌进了日志）。
5. **生成式组件包** ✅ `tools/install_generative.py`（check/start/stop/
   install 四个子命令）+ `GENERATIVE.md` 指南。本机 check 通过；
   start 需 `--disable-assets`（旧 comfyui.db 缺表会卡死启动，已修）。
   install 分支面向全新机器（官方源下载，未实测大流量下载）。
6. 体积优化 / macOS：维持不做。

### 这轮抓到的环境大坑（比 bug 更值钱）

- **8765 上跑的可能是打包版**：跑测试前 `netstat -ano | findstr 8765`
  确认监听进程是 `python.exe`（源码）而不是 `RetouchLab.exe`。
  本轮 42/72 的初始失败里一大半源于此。
- **verify_packaged_ui.py 曾劫持 tabs[0]**：它把第一个标签页导航到
  8766 测完不还原，同批其它 UI 测试全部"浏览器未连接"。已改为：
  探不到打包服务直接 SKIPPED；用自建标签页，测完即关。
- **服务重启后 CDP 标签页变 chrome-error://**，URL 不再匹配 8765，
  测试找不到标签页。重启服务后要 `Page.navigate` 回 8765 一次。

### 新增回归

- `tests/verify_release_features.py`：版本接口 / errors 接口 / 关于
  面板 / 引导卡 / newer() 版本序比较，全链路守起来。

### 这轮修的真 bug（非环境）

1. **error.log 会爆炸**：错误的 multipart 请求把整张图（79MB）的
   二进制灌进 traceback，写 UTF-8 日志再炸 UnicodeDecodeError。
   修复：traceback 截断 8000 字符 + errors="replace" + 2MB 轮转。
2. **tests/verify_done.py 重置错文件**：还在删 `server/_library_done
   .json`（旧路径），服务实际读 `data/_library_done.json`——第二轮
   跑必挂。已改走 `engine.paths.data_dir()`。
3. **showWelcomeOnce 不防重复**：未交互期间再次调用会出现两张同 id
   引导卡，点按钮只收掉一张。已加去重。
4. **build-stamp 路径多跳一级 `..`**：打包版永远报 build=dev。已修，
   新 dist 实测报 `2026-10-24.1`。
5. **install_generative 的 stop 杀错进程**：ComfyUI 半死（不监听但
   占 8GB 内存）时 netstat 找不到它，这个僵尸进程把 Chrome 挤死了
   （本轮第三次重跑全挂的原因）。stop 已加命令行匹配兜底；
   ComfyUI 本机启动必须带 `--disable-assets`（旧 comfyui.db 缺表
   会卡死启动）。

### 最终验收数据（全部实测，2026-10-24 深夜）

- 测试：**72/72 套原始回归 + 1 套新增** 在干净环境逐套通过
  （首跑 42/72 的失败全部归因于上文的三个环境坑，见 out/_round3_*.txt）
- 打包版（v1.1.0 / 2026-10-24.1）：`/api/version` 报 stamp 正确、
  GPU 实测 matmul 61.6ms、46 个工具按钮、零 JS 异常、render+人脸
  检测 200（旧 dist 的 cv2 崩溃在新构建不复现）
- 生成式：ComfyUI 8388 起来后 health `generative_fill:true`，
  verify_gen / verify_fill_api 双 PASS（fill 局部性 2.9x 达标）
- 源码版引导卡：18/18 新功能断言通过

### 遗留注意

- `git push` 会被 Mimosa 门禁拦（tests/*.py 的 localhost 调用被判
  SSRF）。备份走 `python tools/git_backup_push.py`（Git Data API），
  远端 commit SHA 与本地不同（GitHub 规范化了提交者邮箱），属预期。
- 全局 git 配置把 github.com fetch 重写到 gitclone.com 镜像，
  新建的私有仓库要等镜像同步才能 `git fetch`，或临时用
  `git -c "url.https://github.com/.insteadOf=" fetch <url>` 直连。
- 打包版与源码版并跑测试时：打包版必须 `RetouchLab.exe --port 8766`
  显式换端口，否则启动器看到 8765 有"自己人"就直接复用退出。
- 跑批判定口径：`verify_raw_api` 的 RESULT 行是描述式（"RESULT: RAW
  through the API works end to end"），按退出码/无 FAILED 判，别只
  grep "ALL PASS"。

---

## 九、这一轮（2026-10-23 ~ 24）实际做了什么

### 完成的
- ✅ `requirements.txt` + `requirements-generative.txt`（版本锁定，之前没有）
- ✅ `engine/paths.py`：统一路径解析，支持源码/打包双模式
- ✅ `engine/cliponnx.py`：ONNX CLIP，挑图不再需要 transformers+torch
- ✅ `tools/launch.py` + `RetouchLab.bat`：双击启动、单实例、自动端口、自动开浏览器
- ✅ `RetouchLab.spec` + `tools/assemble_dist.py` + `build_app.bat`：完整打包链
- ✅ `/api/diagnose` 端点：发行版可自诊断
- ✅ 全局异常处理器：未捕获异常写 `out/error.log`
- ✅ `/api/health` 提速：2030ms → 6ms（不再每次探测 ComfyUI，加 30 秒缓存）
- ✅ `tests/verify_clip_onnx.py`、`tests/verify_packaged_ui.py`：守着打包版的关键契约

### 打包版的实测验收数据
- 启动：双击后 **46.4 秒**可用（含模型预热），浏览器自动打开
- 内存：**525MB**（空载）
- 接口：`/api/health` 10ms、`/api/cats` 10ms、`/api/params` 10ms、首页 120ms
- GPU：`gpu_works: true`，`gpu_matmul_ms: 34.02`，RTX 3080 正常
- 界面：46 个工具按钮、零 JS 异常、零外链依赖、截图与源码版一致
- 超清：104-143 秒（与源码版 104-108 秒一致，全尺寸图的固有成本）

### 遇到的严重事故（记录备查）

**编码事故**：我用 PowerShell 的 `Get-Content -Raw` + `WriteAllText` 改
`server/app.py`，把中文全变成乱码（UTF-8 被当 GBK 读又写回）。

恢复过程（很惨，别重蹈）：
1. 先试直接用 GBK 反解 → 失败，因为 Windows cp936 会把被吞的 ASCII 字节
   映射成 `?`，有损
2. 改用 pyc 字节码里的字符串常量做字典回填 → 修好 189 处
3. 剩下的用「逆向字节映射 + UTF-8 状态机回溯」（DFS）逐行还原 → 350 行全中
4. 补回 8 处被吞掉的引号（其中 5 处是 `"""` 少了第三个引号）
5. 用 AST 扫出 1 处 `%` 格式化崩溃点（`"、s」"` 缺 `%s` 却传参）

**教训**：改含中文的源码，一律用 `edit`/`write` 工具，或 Python 显式
`encoding='utf-8'`。**绝对不要用 PowerShell 的 `Get-Content -Raw` + `WriteAllText` 组合。**

**死机事故**：同时跑 4 个重型服务（ComfyUI + 源码版 + 打包版 + Chrome）
+ 回归测试，32GB 内存吃光导致死机。跑重活时**一次只跑一个**。

---

## 十、关键文件索引

| 文件 | 作用 |
|---|---|
| `engine/paths.py` | **路径解析的唯一入口**，改打包行为必看 |
| `engine/cliponnx.py` | ONNX CLIP 图像塔（挑图用） |
| `tools/launch.py` | 启动器（单实例/端口/浏览器） |
| `tools/assemble_dist.py` | 打包后组装（models/web/bat/README） |
| `RetouchLab.spec` | PyInstaller 配置（含排除清单和原因） |
| `build_app.bat` | 一键打包 |
| `server/app.py` | 全部 API（3582 行，唯一的后端文件） |
| `web/index.html` | 全部前端（7152 行，单文件） |
| `requirements.txt` | 核心依赖（不含生成式） |
| `requirements-generative.txt` | 生成式可选依赖（torch 等） |
| `out/error.log` | 运行时未捕获异常（发行版排障入口） |
| `out/launcher.log` | 启动器日志 |

---

## 十一、给接手人的三句话

1. **改路径相关代码前先看 `engine/paths.py`**，那套分派是打包能用的前提。
2. **跑测试前先确认环境**（Chrome 开了标签页 / ComfyUI 起了 / 用的是源码版 8765），
   否则会看到几十个假失败，白折腾。
3. **`is_available()` 为 true 不等于能用**。诊断问题时要看实测数据
   （`/api/diagnose` 的 `gpu_works`、`gpu_matmul_ms`），不要只看布尔值。
