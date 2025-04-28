import os
import urllib.request

BASE_URL = "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2"
ENV_NAMES = ["halfcheetah", "hopper", "walker2d"]
VERSIONS = ["medium", "expert", "medium_expert", "medium_replay", "random", "full_replay"]

# 下载保存的路径
save_dir = os.path.expanduser("~/.d4rl/datasets")
os.makedirs(save_dir, exist_ok=True)

total = len(ENV_NAMES) * len(VERSIONS)
count = 0

for env in ENV_NAMES:
    for version in VERSIONS:
        filename = f"{env}_{version}-v2.hdf5"
        url = f"{BASE_URL}/{filename}"
        local_path = os.path.join(save_dir, filename)
        count += 1

        if os.path.exists(local_path):
            print(f"[{count}/{total}] 已存在: {filename}")
            continue

        print(f"[{count}/{total}] 正在下载: {filename} ...")
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"✅ 下载完成: {filename}")
        except Exception as e:
            print(f"❌ 下载失败: {filename}，错误信息: {e}")
