import os
import sys
from pathlib import Path
for line in Path(__file__).with_name('server.env').read_text().splitlines():
    if line and not line.startswith('#'):
        key, value = line.split('=', 1)
        os.environ[key] = value
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tt-media-server'))
import uvicorn
uvicorn.run('main:app', host=os.environ.get('HTTP_HOST', '127.0.0.1'), port=8000, workers=1, timeout_keep_alive=60)
