from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prob_instance_slicer import load_slice_corpus


@dataclass(frozen=True)
class MetaBBOCompatibleSliceProblem:
    corpus_name: str
    problem_index: int
    slice_index: int
    high_dim_instance_id: str
    family_id: str
    dim: int
    lb: float
    ub: float
    optimum: float | None
    _slice_problem: object
    _slice_task: object

    def eval(self, x):
        return self._slice_task.evaluate(self._slice_problem, x)

    def func(self, x):
        return self.eval(x)

    def reset(self):
        return None

    def __str__(self) -> str:
        return f"{self.corpus_name}/p{self.problem_index:03d}/s{self.slice_index:02d}"


class SliceProblemSet:
    def __init__(self, problems: list[MetaBBOCompatibleSliceProblem]) -> None:
        self._problems = list(problems)
        self.N = len(self._problems)

    def __iter__(self):
        return iter(self._problems)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, index: int) -> MetaBBOCompatibleSliceProblem:
        return self._problems[index]


def construct_slice_problem_set(
    corpus_root: str | Path,
    *,
    selected_count: int = 2,
) -> tuple[SliceProblemSet, SliceProblemSet]:
    corpus_root = Path(corpus_root)
    corpus = load_slice_corpus(corpus_root)
    selected: list[MetaBBOCompatibleSliceProblem] = []
    corpus_name = corpus_root.name

    for problem_index, bundle in corpus.items():
        lower_bound, upper_bound = bundle.problem.bounds
        for task in bundle.tasks:
            optimum = float(task.evaluate(bundle.problem, task.optimum))
            selected.append(
                MetaBBOCompatibleSliceProblem(
                    corpus_name=corpus_name,
                    problem_index=problem_index,
                    slice_index=task.slice_id,
                    high_dim_instance_id=bundle.problem.instance_id,
                    family_id=bundle.problem.family_id,
                    dim=int(task.indices.size),
                    lb=float(lower_bound),
                    ub=float(upper_bound),
                    optimum=optimum,
                    _slice_problem=bundle.problem,
                    _slice_task=task,
                )
            )
            if len(selected) >= selected_count:
                dataset = SliceProblemSet(selected)
                # For the current smoke-test route, use the same small set for train/test.
                return dataset, SliceProblemSet(list(selected))

    raise ValueError(
        f"Requested {selected_count} slice problems from {corpus_root}, "
        f"but only found {len(selected)}."
    )


def construct_slice_problem_set_from_specs(
    corpus_root: str | Path,
    specs: list[tuple[int, int]],
) -> SliceProblemSet:
    corpus_root = Path(corpus_root)
    corpus = load_slice_corpus(corpus_root)
    corpus_name = corpus_root.name
    selected: list[MetaBBOCompatibleSliceProblem] = []
    for problem_index, slice_index in specs:
        bundle = corpus[problem_index]
        task = next((item for item in bundle.tasks if item.slice_id == slice_index), None)
        if task is None:
            raise ValueError(f"Slice id {slice_index} not found under problem {problem_index} in {corpus_root}.")
        lower_bound, upper_bound = bundle.problem.bounds
        optimum = float(task.evaluate(bundle.problem, task.optimum))
        selected.append(
            MetaBBOCompatibleSliceProblem(
                corpus_name=corpus_name,
                problem_index=problem_index,
                slice_index=task.slice_id,
                high_dim_instance_id=bundle.problem.instance_id,
                family_id=bundle.problem.family_id,
                dim=int(task.indices.size),
                lb=float(lower_bound),
                ub=float(upper_bound),
                optimum=optimum,
                _slice_problem=bundle.problem,
                _slice_task=task,
            )
        )
    return SliceProblemSet(selected)


def construct_all_slices_for_problem(
    corpus_root: str | Path,
    *,
    problem_index: int | None = None,
    high_dim_instance_id: str | None = None,
) -> SliceProblemSet:
    corpus_root = Path(corpus_root)
    corpus = load_slice_corpus(corpus_root)
    corpus_name = corpus_root.name
    bundle = None
    resolved_problem_index = problem_index
    if problem_index is not None:
        bundle = corpus[problem_index]
    else:
        for candidate_index, candidate in corpus.items():
            if candidate.problem.instance_id == high_dim_instance_id:
                bundle = candidate
                resolved_problem_index = candidate_index
                break
    if bundle is None:
        raise ValueError(
            f"Unable to resolve high-dimensional problem in {corpus_root} "
            f"for problem_index={problem_index}, high_dim_instance_id={high_dim_instance_id}."
        )
    lower_bound, upper_bound = bundle.problem.bounds
    selected: list[MetaBBOCompatibleSliceProblem] = []
    for task in bundle.tasks:
        optimum = float(task.evaluate(bundle.problem, task.optimum))
        selected.append(
            MetaBBOCompatibleSliceProblem(
                corpus_name=corpus_name,
                problem_index=int(resolved_problem_index),
                slice_index=task.slice_id,
                high_dim_instance_id=bundle.problem.instance_id,
                family_id=bundle.problem.family_id,
                dim=int(task.indices.size),
                lb=float(lower_bound),
                ub=float(upper_bound),
                optimum=optimum,
                _slice_problem=bundle.problem,
                _slice_task=task,
            )
        )
    return SliceProblemSet(selected)
