"""
The parallel model master class for Mesa framework.

Core Objects: ModelMaster"""
# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

import random
import multiprocessing
import socket
import time

from typing import Any

from mesa.datacollection import DataCollector, ParallelDataCollector
from mesa.model import Model
from mesa.server import model_worker_server, communicate_message

class WorkerException(Exception):

    """Docstring for WorkerException. """

    def __init__(self, *args, **kwargs):
        """TODO: to be defined. """
        Exception.__init__(self, *args, **kwargs)

        


class ModelMaster:
    """A parallel worker managing computation and messaging with other child_models"""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create a new model object and instantiate its RNG automatically."""
        obj = object.__new__(cls)
        obj._seed = kwargs.get("seed", None)
        obj.random = random.Random(obj._seed)
        return obj

    def __init__(self, n_workers: int, WorkerModel: type, port=None, *args, **kwargs: Any) -> None:
        """Create a new model. Overload this method with the actual code to
        start the model.

        Attributes:
            schedule: schedule object
            running: a bool indicating if the model should continue running
        """

        self.running = True
        self.schedule = None
        self.current_id = 0
        self._port = 8000 if port is None else port
        self._n_workers = n_workers
        self._child_models = {i: WorkerModel(*args, **kwargs) for i in range(n_workers)}
        self._model_workers = {}
        self.schedule_allocation = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for _, model in self._model_workers.items():
            model['connection'].close()
            model['process'].terminate()

        if hasattr(self, 'grid'):
            self.grid.__exit__(exc_type, exc_val, exc_tb)
        return True


    def initialize_worker_processes(self) -> None:
        """ Initialize worker processes
        """
        ports = [self._port + i for i in range(self._n_workers)]
        for i in range(self._n_workers):
            port = ports[i]
            process = multiprocessing.Process(target=model_worker_server, args=(port,self._child_models[i]))
            process.start()
            time.sleep(2)
            server_address = ('localhost', port)
            print(server_address)
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(server_address)
            self._model_workers[i] = {'address': server_address, 'process': process, 'connection': client_socket}


    def run_model(self) -> None:
        """Run the model until the end condition is reached. Overload as
        needed.
        """
        while self.running:
            self.step()

    def step(self) -> None:
        """Step each worker node. Fill in here."""
        model_statuses = []
        for i, worker in self._model_workers.items():
            response = communicate_message(worker, ('advance', tuple()))
            # need to use threading to async get all 
            model_statuses.append(response is None)

        if all(model_statuses):
            # Allow the models to resolve things amongst themselves
            for i, model in self._model_workers.items():
                #print(f"Model {i} working")
                # need to use threading to async get all 
                response = communicate_message(worker, ('step',tuple()))
        else:
            raise WorkerException("Not all worker models returned success")

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
            and all(model.schedule.get_agent_count() == 0 for model in self._child_models.values())
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
        for key, model_worker in self._child_models.items():
            model_worker.initialize_data_collector(
                model_reporters=model_reporters,
                agent_reporters=agent_reporters,
                tables=tables,
                exclude_none_values=exclude_none_values
            )
        # Collect data for the first time during initialization.
        self.datacollector.collect(self)

    def assign_scheduled_agents_to_child_models(self, random=True):
        """Assign agents to each of the child_models."""
        agents_to_cleanup = []
        for key, agent in self.schedule._agents.items():
            k = self.random.randint(0, len(self._child_models)-1)
            self._child_models[k].schedule.add(agent)
            agent.model = self._child_models[k]
            self.schedule_allocation[key] = {'worker': k}
            agents_to_cleanup.append(agent)

        for agent in agents_to_cleanup:
            self.schedule.remove(agent)

    def link_grid_to_child_models(self):
        """Link the grid to each of the child_models"""
        grid = self.grid
        for key, worker_model in self._child_models.items():
            # TODO: remove dependency on grid private attributes _xxx...
            worker_model.link_shared_memory_grid(grid)

