import os
import time
import wandb
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

class Logging:
    def __init__(self, args):
        self.__log = defaultdict(float)
        self.__epoch = 0
        self.__args = args

        if not os.path.isdir(args.exp_dir):
            raise ValueError("Invalid experiment directory path provided.")

        if args.wandb:
            if not args.wandb_prj:
                raise ValueError("Project name must be specified when using wandb.")
            
            args.run_name = f"{args.ds}_{args.polar}_{args.loss}_{args.model}__{int(time.time())}"
            self.__run = wandb.init(
                project=args.wandb_prj,
                config=args,
                name=args.run_name,
                force=True
            )
        else:
            self.__run = None

        if args.log:
            self.__writer = SummaryWriter(args.exp_dir)
            self.__writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )
        else:
            self.__writer = None

    def __call__(self, key, value):
        self.__log[key] += value

    def __update_wandb(self):
        if self.__run:
            for log_key, log_value in self.__log_avg.items():
                if log_key == 'loss':
                    self.__run.log({'loss': log_value}, step=self.__epoch)  # Log loss explicitly
                elif log_key == 'dsc':
                    self.__run.log({'dsc': log_value}, step=self.__epoch)  # Log DSC explicitly
                else:
                    self.__run.log({log_key: log_value}, step=self.__epoch)

    def __update_board(self):
        if self.__writer:
            for log_key, log_value in self.__log_avg.items():
                self.__writer.add_scalar(log_key, log_value, self.__epoch)

    def __reset_epoch(self):
        self.__log.clear()

    def reset(self):
        self.__reset_epoch()
        self.__epoch = 0

    def step(self, epoch):
        self.__epoch = epoch
        self.__log_avg = {}

        for log_key, log_value in self.__log.items():
            if log_key.split("/")[-1] in ["loss", "dsc"]:
                divisor = getattr(self.__args, f"num_{log_key.split('/')[0]}_batch", 0)
            else:
                divisor = getattr(self.__args, f"num_{log_key.split('/')[0]}_sample", 0)

            if divisor > 0:
                self.__log_avg[log_key] = log_value / divisor
            else:
                raise ValueError(f"Invalid divisor for key: {log_key}")

        self.__update_wandb()
        self.__update_board()
        self.__reset_epoch()

    def watch(self, model):
        if self.__run:
            self.__run.watch(models=model, log="all", log_freq=self.__args.num_train_batch, log_graph=True)

    def log_model(self):
        if self.__run:
            for suffix in ["best", "last"]:
                model_path = os.path.join(self.__args.exp_dir, f"{suffix}.pt")
                if os.path.exists(model_path):
                    self.__run.log_model(path=model_path, model_name=f"{self.__args.run_name}-{suffix}-model")

    def close(self):
        if self.__run:
            self.__run.finish()
        if self.__writer:
            self.__writer.close()

    @property
    def log(self):
        return dict(self.__log)

    @property
    def log_avg(self):
        return getattr(self, "_Logging__log_avg", {})

    @property
    def epoch(self):
        return self.__epoch

    @property
    def args(self):
        return self.__args
