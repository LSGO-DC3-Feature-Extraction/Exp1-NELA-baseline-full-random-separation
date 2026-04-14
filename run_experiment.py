import argparse
import datetime

import numpy as np

from train_metabbo_with_feat_extractor import train
from gen_problem import gen_slice_save, load_sliced_problem, load_full_problem


class Config:
    def __init__(self):
        curr_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

        self.total_dim = 1000
        self.sliced_dim = 20
        self.problem_id = 1
        self.checkpoint = "./feat_extractor/checkpoint.pkl"
        self.save_path = f"./records/{curr_time}"



        self.train_set_path = "./records/26-04-14-01-31"
        self.test_set_path = "./records/26-04-14-14-44"
        self.algo_name = "LDE"
        self.train_epoch = 2
        self.max_fes = 2000
        self.max_learning_step = 10
        self.use_ray = True
        self.ray_num_cpus = 4

        self.seed = 676
        self.use_feat_extractor = True
    

def test_func():
    for i in range(20):
        func = load_sliced_problem("./records/26-04-14-01-31", i)
        print(f"sliced id: {i}")
        print(f"1 value: {func.eval(np.ones(shape=(1, 20)))}")
        print(f"zero value: {func.eval(np.zeros(shape=(1, 20)))}")
    
    func = load_full_problem("./records/26-04-14-01-31")
    print(func.eval(np.zeros(shape=(2, 1000))))

def main():
    cfg = Config()
    def build_problem_set(save_path, count):
        return [load_sliced_problem(save_path, i) for i in range(count)]
    build_problem_set
    train_set = build_problem_set(cfg.train_set_path, count=50)
    test_set = build_problem_set(cfg.test_set_path, count=10)
    
    print("Dataset build finished")
    result = train(
        train_set,
        test_set,
        cfg.algo_name,
        checkpoint_path=cfg.checkpoint,
        seed=cfg.seed,
        use_feat_extractor=cfg.use_feat_extractor,
        train_epoch=cfg.train_epoch,
        max_fes=cfg.max_fes,
        max_learning_step=cfg.max_learning_step,
        use_ray=cfg.use_ray,
        ray_num_cpus=cfg.ray_num_cpus,
    )

    print(result["resolved_agent_name"])
    print(result["train_history"][-1])
    print(result["test_results"])    


if __name__ == "__main__":
    main()
