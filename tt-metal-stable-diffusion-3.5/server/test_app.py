import os
from gunicorn.app.base import BaseApplication
from server.app import app as flask_app, worker as worker_job
from server.model import warmup_model
from threading import Thread


class GunicornApp(BaseApplication):
    def __init__(self, app):
        self.app = app
        super().__init__()

    def load(self):
        return self.app

    def load_config(self):
        config = {
            "bind": f"0.0.0.0:{7000}",  # Specify the binding address
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
        thread = Thread(target=worker_job)
        thread.daemon = True
        thread.start()


def test_app():
    # Ensure the generated images directory exists
    os.makedirs("generated_images", exist_ok=True)
    app = GunicornApp(flask_app)
    app.run()
