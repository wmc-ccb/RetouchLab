# -*- coding: utf-8 -*-
"""精确还原 mojibake：构建 cp936 解码的逆向映射 + UTF-8 状态机回溯。

思路：损坏 = 原始 UTF-8 字节流被 cp936 逐字节对解码成 Unicode 文本。
逆向：每个 mojibake 字符对应 1 或 2 个字节（cp936 单字节 0x00-0x7F 或
双字节），枚举所有候选，用 UTF-8 合法性做约束，DFS 找唯一路径。
"""
import ctypes
import io
import sys

CP936 = 936
k32 = ctypes.windll.kernel32


def m2w(data: bytes) -> str:
    if not data:
        return ""
    n = k32.MultiByteToWideChar(CP936, 0, data, len(data), None, 0)
    if n <= 0:
        return "\x00"
    buf = ctypes.create_string_buffer(n * 2)
    k32.MultiByteToWideChar(CP936, 0, data, len(data), buf, n)
    return buf.raw.decode("utf-16-le")


# ---- 1) 构建 字符 -> 字节序列 的候选表 ----
rev = {}
# cp936 的单字节范围是 0x00-0x7F，但 Windows 还有几个特殊单字节映射
# （0x80 -> €）。全部单字节都试一遍，取到 1 个字符的就是候选。
for b in range(0x100):
    c = m2w(bytes([b]))
    if len(c) == 1 and c != "\x00":
        rev.setdefault(c, []).append(bytes([b]))
for b1 in range(0x81, 0xFF):
    for b2 in range(0x100):
        try:
            data = bytes([b1, b2])
        except ValueError:
            continue
        c = m2w(data)
        if len(c) == 1 and c != "\x00":
            rev.setdefault(c, []).append(data)
print("逆向表字符数:", len(rev))


def utf8_need(b: int) -> int:
    """UTF-8 首字节期望的后续字节数；非法返回 -1。"""
    if b < 0x80:
        return 0
    if 0xC2 <= b <= 0xDF:
        return 1
    if 0xE0 <= b <= 0xEF:
        return 2
    if 0xF0 <= b <= 0xF4:
        return 3
    return -1


def is_cont(b: int) -> bool:
    return 0x80 <= b <= 0xBF


def reverse_text(text: str):
    """把 mojibake 文本还原成原始字节。DFS + 记忆化。"""
    N = len(text)
    # 先做候选压缩：字母表里每个字符的候选
    cands = []
    for ch in text:
        opts = rev.get(ch)
        if not opts:
            # 无法映射（正常 ASCII 或未知），当作原样字节
            opts = [ch.encode("utf-8")] if ord(ch) < 128 else []
        cands.append(opts)
        if not opts:
            return None, "无法映射字符 %r (U+%04X) 在位置 %d" % (
                ch, ord(ch), len(cands) - 1)

    # 状态：(i, need, acc_bytes_of_partial) —— 用 (i, need, partial) 记忆
    from functools import lru_cache
    sys.setrecursionlimit(10000)

    memo = {}

    def solve(i, need, partial):
        """partial: 当前已累积但未完成的 UTF-8 序列（bytes）。返回 (bytes|None)"""
        key = (i, need, partial)
        if key in memo:
            return memo[key]
        if i == N:
            ok = need == 0
            memo[key] = (b"",) if ok else None
            return memo[key]
        for opt in cands[i]:
            # 把 opt 的字节逐个喂给状态机
            st_need, st_part = need, partial
            ok = True
            for b in opt:
                if st_need == 0:
                    n2 = utf8_need(b)
                    if n2 < 0:
                        ok = False
                        break
                    st_need, st_part = n2, b""
                else:
                    if not is_cont(b):
                        ok = False
                        break
                    st_need -= 1
            if not ok:
                continue
            r = solve(i + 1, st_need, st_part)
            if r is not None:
                memo[key] = (opt + r[0],)
                return memo[key]
        memo[key] = None
        return None

    r = solve(0, 0, b"")
    if r is None:
        return None, "无解"
    return r[0], None


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    text = io.open(src, encoding="utf-8", newline="").read()
    # 只还原非 ASCII 段：逐行处理（行内 ASCII 部分原样保留更快）
    out_lines = []
    fails = 0
    for ln, line in enumerate(text.split("\n"), 1):
        if all(ord(c) < 128 for c in line):
            out_lines.append(line)
            continue
        data, err = reverse_text(line)
        if data is None:
            fails += 1
            out_lines.append(line)   # 保留原样，后面单独处理
            if fails <= 5:
                print("行 %d 还原失败: %s" % (ln, err))
            continue
        try:
            out_lines.append(data.decode("utf-8"))
        except UnicodeDecodeError as e:
            fails += 1
            out_lines.append(line)
            if fails <= 5:
                print("行 %d 解码失败: %s" % (ln, e))
    io.open(dst, "w", encoding="utf-8", newline="").write("\n".join(out_lines))
    print("写出 %s，失败行 %d" % (dst, fails))


if __name__ == "__main__":
    main()
