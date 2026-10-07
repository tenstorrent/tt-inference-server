import os
import sys
from pathlib import Path
# The deployment's env file: argv[1], or the one beside this script.
env_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('server.env')
for line in env_file.read_text().splitlines():
    if line and not line.startswith('#'):
        key, value = line.split('=', 1)
        os.environ[key] = value
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tt-media-server'))
import uvicorn
uvicorn.run('main:app', host=os.environ.get('HTTP_HOST', '127.0.0.1'),
            port=int(os.environ.get('HTTP_PORT', '8000')), workers=1, timeout_keep_alive=60)
