import os, urllib.request, json
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "mvtec_gt_test")

hf_api_url = "https://huggingface.co/api/datasets/foersben/mvtec-ad"
req = urllib.request.Request(hf_api_url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as resp:
    repo_data = json.loads(resp.read().decode("utf-8"))

all_files = [s["rfilename"] for s in repo_data.get("siblings", [])]
screw_files = [f for f in all_files if f.startswith("screw/test/")]

def dl(f):
    local_path = os.path.join(CACHE_DIR, f)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"https://huggingface.co/datasets/foersben/mvtec-ad/resolve/main/{f}"
    try:
        r = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(r, timeout=15) as u:
            data = u.read()
        with open(local_path, "wb") as out:
            out.write(data)
    except Exception as e:
        pass

with ThreadPoolExecutor(max_workers=32) as ex:
    list(ex.map(dl, screw_files))
print("Screw download finished!")
