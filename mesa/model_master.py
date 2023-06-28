"""
The parallel model master class for Mesa framework.

Core Objects: ModelMaster"""
# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

import random

# mypy
from typing import Any

from mesa.datacollection import DataCollector, ParallelDataCollector
from mesa.model import Model

class ModelMaster:
    """A parallel worker managing computation and messaging with other workers"""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create a new model object and instantiate its RNG automatically."""
        obj = object.__new__(cls)
        obj._seed = kwargs.get("seed", None)
        obj.random = random.Random(obj._seed)
        return obj

    def __init__(self, n_workers: int, WorkerModel: type, *args, **kwargs: Any) -> None:
        """Create a new model. Overload this method with the actual code to
        start the model.

        Attributes:
            schedule: schedule object
            running: a bool indicating if the model should continue running
        """

        self.running = True
        self.schedule = None
        self.current_id = 0
        self._model_workers = {i: WorkerModel(*args, **kwargs) for i in range(n_workers)}
        self.schedule_allocation = {}

    def run_model(self) -> None:
        """Run the model until the end condition is reached. Overload as
        needed.
        """
        import ipdb; ipdb.set_trace()
        while self.running:
            self.step()

    def step(self) -> None:
        """Step each worker node. Fill in here."""
        model_statuses = []
        for i, model in self._model_workers.items():
            # This won't work as-is - need all processes to send an OK signal
            status = model.advance()
            model_statuses.append(status)

        if all(model_statuses):
            # Allow the models to resolve things amongst themselves
            for i, model in self._model_workers.items():
                #print(f"Model {i} working")
                model.step()
        print(self.grid._agent_points)

    def next_id(self) -> int:
        """Return the next unique ID for agents, increment current_id"""
        self.current_id += 1
        return self.current_id

    def reset_randomizer(self, seed: int | None = None) -> None:
        """Reset the model random number generator.

        Args:
            seed: A new seed for the RNG; if None, reset using the current seed
        """

        if seed is None:
            seed = self._seed
        self.random.seed(seed)
        self._seed = seed

    def initialize_data_collector(
        self,
        model_reporters=None,
        agent_reporters=None,
        tables=None,
        exclude_none_values=False,
    ) -> None:
        if not hasattr(self, "schedule") or self.schedule is None:
            raise RuntimeError(
                "You must initialize the scheduler (self.schedule) before initializing the data collector."
            )
        if (self.schedule.get_agent_count() == 0
            and all(model.schedule.get_agent_count() == 0 for model in self._model_workers.values())
            ):
            raise RuntimeError(
                "You must add agents to the scheduler before initializing the data collector."
            )
        self.datacollector = ParallelDataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
            tables=tables,
            exclude_none_values=exclude_none_values,
        )
        for key, model_worker in self._model_workers.items():
            model_worker.initialize_data_collector(
                model_reporters=model_reporters,
                agent_reporters=agent_reporters,
                tables=tables,
                exclude_none_values=exclude_none_values
            )
        # Collect data for the first time during initialization.
        self.datacollector.collect(self)

    def assign_scheduled_agents_to_workers(self, random=True):
        """Assign agents to each of the workers."""
        agents_to_cleanup = []
        for key, agent in self.schedule._agents.items():
            k = self.random.randint(0, len(self._model_workers)-1)
            self._model_workers[k].schedule.add(agent)
            agent.model = self._model_workers[k]
            self.schedule_allocation[key] = {'worker': k}
            agents_to_cleanup.append(agent)

        for agent in agents_to_cleanup:
            self.schedule.remove(agent)

    def link_grid_to_workers(self):
        """Link the grid to each of the workers"""
        grid = self.grid
        for key, worker_model in self._model_workers.items():
            # TODO: remove dependency on grid private attributes _xxx...
            worker_model.link_shared_memory_grid(grid)

