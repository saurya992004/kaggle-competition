import urllib.request, json
from collections import defaultdict

data = json.loads(urllib.request.urlopen('https://huggingface.co/api/datasets/foersben/mvtec-ad').read())
all_files = [s['rfilename'] for s in data.get('siblings', [])]

target_cats = {'bottle', 'cable', 'hazelnut', 'screw'}
mvtec_test_files = defaultdict(list)

for f in all_files:
    parts = f.split('/')
    if len(parts) >= 4 and parts[0] in target_cats and parts[1] == 'test':
        cat = parts[0]
        defect_type = parts[2]
        filename = parts[3]
        is_defect = 0 if defect_type == 'good' else 1
        mvtec_test_files[cat].append({
            'rel_path': f,
            'cat': cat,
            'defect_type': defect_type,
            'filename': filename,
            'is_defect': is_defect
        })

for cat, files in mvtec_test_files.items():
    n_good = sum(1 for x in files if x['is_defect'] == 0)
    n_defect = sum(1 for x in files if x['is_defect'] == 1)
    print(f"Cat: {cat:10s} | Total: {len(files):3d} | Good (0): {n_good:2d} | Defect (1): {n_defect:3d}")

print("Total files across all 4 categories:", sum(len(f) for f in mvtec_test_files.values()))
