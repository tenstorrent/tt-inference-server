from model_services.scheduler import Scheduler

# scheduler is singleton
current_scheduler_holder = None

def get_scheduler() -> Scheduler:
    global current_scheduler_holder
    if (current_scheduler_holder is None):
        current_scheduler_holder = Scheduler()
    return current_scheduler_holder