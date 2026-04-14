from __future__ import annotations

from config import get_config, refresh_config
from eval import construct_problem_set, get_epoch_dict
from trainer import Trainer


def main():
    seed = 0
    config = get_config()
    config.dataset = "slice-50x20-two"
    config.dim = 50
    config = refresh_config(config)
    train_set, test_set = construct_problem_set(dataset=config.dataset)

    task_epoch_dict = get_epoch_dict(config.dataset)
    config.train_agent = "RLEPSO_Agent"
    config.train_optimizer = "RLEPSO_Optimizer"
    config.train_epoch = task_epoch_dict[config.train_agent[:-6]]
    config.fitness_mode = "z-score"
    config.in_task_agg = "np.mean"
    config.out_task_agg = "np.mean"

    trainer = Trainer(config, train_set, test_set, seed)
    result = trainer.train(trival=True)
    print(config.train_agent)
    print(result)


if __name__ == "__main__":
    main()
