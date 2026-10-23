# -*- coding: utf-8 -*-
"""我的分组：存 / 载 / 去重 / 空分组自动消失."""
import os
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B = "http://127.0.0.1:8765"
FAIL = []
GF = os.path.join(ROOT, "server", "_library_groups.json")


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def clean():
    if os.path.exists(GF):
        os.remove(GF)


def main():
    clean()

    r = requests.post(B + "/api/library/group_add", json={
        "name": "客户A", "paths": ["/x/1.jpg", "/x/2.jpg"]}, timeout=60)
    check("创建分组", r.ok and r.json().get("added") == 2, str(r.json())[:80])

    r = requests.post(B + "/api/library/group_add", json={
        "name": "客户A", "paths": ["/x/2.jpg", "/x/3.jpg"]}, timeout=60)
    j = r.json()
    check("追加去重", r.ok and j.get("added") == 1 and j.get("total") == 3,
          str(j)[:80])

    r = requests.get(B + "/api/library/groups", timeout=60).json()
    check("分组列表带计数", r.get("groups", {}).get("客户A", {}).get("count") == 3,
          str(r.get("groups"))[:80])

    r = requests.get(B + "/api/library/group_members",
                     params={"name": "客户A"}, timeout=60).json()
    check("成员可取", r.get("paths") == ["/x/1.jpg", "/x/2.jpg", "/x/3.jpg"],
          str(r.get("paths"))[:80])

    r = requests.post(B + "/api/library/group_add", json={
        "name": "", "paths": ["/x/9.jpg"]}, timeout=60)
    check("空名拒绝 (400)", r.status_code == 400, str(r.status_code))
    r = requests.post(B + "/api/library/group_add", json={
        "name": "空组", "paths": []}, timeout=60)
    check("空路径拒绝 (400)", r.status_code == 400, str(r.status_code))

    r = requests.post(B + "/api/library/group_remove", json={
        "name": "客户A", "paths": ["/x/1.jpg"]}, timeout=60)
    check("移除单个成员", r.ok, str(r.json())[:60])
    r = requests.get(B + "/api/library/group_members",
                     params={"name": "客户A"}, timeout=60).json()
    check("移除后成员正确", r.get("paths") == ["/x/2.jpg", "/x/3.jpg"],
          str(r.get("paths"))[:60])

    r = requests.post(B + "/api/library/group_remove", json={
        "name": "客户A", "paths": ["/x/2.jpg", "/x/3.jpg"]}, timeout=60)
    r2 = requests.get(B + "/api/library/groups", timeout=60).json()
    check("空分组自动消失", r.ok and "客户A" not in r2.get("groups", {}))

    r = requests.post(B + "/api/library/group_rename", json={
        "old": "nope", "new": "x"}, timeout=60)
    check("重命名不存在的 (404)", r.status_code == 404, str(r.status_code))

    r = requests.post(B + "/api/library/group_add", json={
        "name": "备份", "paths": ["/y/1.jpg"]}, timeout=60)
    r = requests.post(B + "/api/library/group_rename", json={
        "old": "备份", "new": "正式"}, timeout=60)
    check("重命名成功", r.ok and r.json().get("group") == "正式",
          str(r.json())[:60])
    r = requests.get(B + "/api/library/groups", timeout=60).json()
    check("重命名后列表正确", "正式" in r.get("groups", {})
          and "备份" not in r.get("groups", {}), str(r.get("groups"))[:80])

    r = requests.post(B + "/api/library/group_delete",
                      json={"name": "正式"}, timeout=60)
    check("删除分组", r.ok, str(r.status_code))

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        clean()
