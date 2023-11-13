from pathlib import Path
import requests

donlot_model = 'https://huggingface.co/datasets/xontoloyoo/mymodel/resolve/main/Model-Training/'
donlot_rh = 'https://huggingface.co/datasets/xontoloyoo/mymodel/resolve/main/'

BASE_DIR = Path.cwd()

# Daftar tupel dengan nama model dan direktori tujuan
download_targets = [
    ('D32k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('D40k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('D48k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('G32k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('G40k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('G48k.pth', 'assets/pretrained_v2', 'donlot_model'),
    
    ('f0D32k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('f0D40k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('f0D48k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('f0G32k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('f0G40k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('f0G48k.pth', 'assets/pretrained_v2', 'donlot_model'),
    ('hubert_base.pt', 'assets/hubert', 'donlot_rh'),
    ('rmvpe.pt', 'assets/rmvpe', 'donlot_rh')
]

# Fungsi untuk mengunduh model ke direktori yang sesuai
def dl_model(link, model_name, dir_name):
    if not dir_name.exists():
        dir_name.mkdir(parents=True)

    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

if __name__ == '__main__':
    for model, dir_path, source_url in download_targets:
        print(f'Downloading {model}...')
        dl_model(globals()[source_url], model, BASE_DIR / dir_path)

    print('All models downloaded!')
