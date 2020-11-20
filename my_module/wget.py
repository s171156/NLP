import sys
import requests
import shutil

param = sys.argv

# verify=Falseで自己証明書でもスキップする
r = requests.get(param[2], stream=True, verify=False)
if r.status_code == 200:
    with open(param[1], 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)
