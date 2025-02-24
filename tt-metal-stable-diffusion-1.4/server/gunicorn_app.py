from gunicorn.app.base import BaseApplication
import os
from models.demos.wormhole.stable_diffusion.demo.web_demo.model import warmup_model
from server.flaskserver import app as flask_app, create_worker
from threading import Thread


class GunicornApp(BaseApplication):
    def __init__(self, app, port):
        self.app = app
        self.port = port
        super().__init__()

    def load(self):
        return self.app

    def load_config(self):
        config = {
            "bind": f"0.0.0.0:{self.port}",  # Specify the binding address
            "workers": 1,  # Number of Gunicorn workers
            "reload": False,
            "worker_class": "gthread",
            "threads": 16,
            "post_worker_init": self.post_worker_init,
            "timeout": 0,
        }

        # Set the configurations for Gunicorn (optional but useful)
        for key, value in config.items():
            self.cfg.set(key, value)

    def post_worker_init(self, worker):
        # all setup tasks and spinup background threads must be performed
        # here as gunicorn spawns worker processes who must be the parent
        # of all server threads
        warmup_model()

        # run the model worker in a background thread
        thread = Thread(target=create_worker)
        thread.daemon = True
        thread.start()


def test_app(port):
    # Ensure the generated images directory exists
    os.makedirs("generated_images", exist_ok=True)
    gunicorn_app = GunicornApp(flask_app, port)
    gunicorn_app.run()
