import wandb

# Modified from INM706 Lab 4 code to simplify logger
class Logger:

    def __init__(self, project='CC_INM706_1'):
        logger = wandb.init(project=project)
        self.logger = logger
        return

    def get_logger(self):
        return self.logger


