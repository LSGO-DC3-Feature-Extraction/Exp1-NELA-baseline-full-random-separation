class PBO_Env:
    """
    An environment with a problem and an optimizer.
    """
    def __init__(self,
                 problem,
                 optimizer,
                 ):
        self.problem = problem
        self.optimizer = optimizer

    def _emit_history(self):
        writer = getattr(self.optimizer, "history_writer", None)
        if writer is None:
            return
        if not hasattr(self.optimizer, "get_best_position"):
            return
        if not hasattr(self.optimizer, "get_best_value"):
            return
        writer.write(
            fes=self.optimizer.fes,
            position=self.optimizer.get_best_position(),
            value=self.optimizer.get_best_value(),
        )

    def reset(self):
        state = self.optimizer.init_population(self.problem)
        self._emit_history()
        return state

    def step(self, action):
        out = self.optimizer.update(action, self.problem)
        self._emit_history()
        return out
