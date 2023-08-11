"""
The parallel model master class for Mesa framework.

Core Objects: ModelMaster"""
# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

import random
import concurrent.futures
import time
import pickle

from typing import Any

from mesa.datacollection import DataCollector, ParallelDataCollector
from mesa.model import Model
from mesa.server import initialize_worker, shutdown_worker, communicate_message
from mesa.space import SharedMemoryContinuousSpace
from mesa.attribute import SharedMemoryAttributeCollection
from tqdm import tqdm
import logging as log

class ModelMaster:
    """A parallel worker managing computation and messaging with other child_models"""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create a new model object and instantiate its RNG automatically."""
        obj = object.__new__(cls)
        obj._seed = kwargs.get("seed", None)
        obj.random = random.Random(obj._seed)
        return obj

    def __init__(self, n_workers: int, WorkerModel: type, dump_on_keyboard_interrupt: bool = True, *args, **kwargs: Any) -> None:
        """Create a new model. Overload this method with the actual code to
        start the model.

        Attributes:
            schedule: schedule object
            running: a bool indicating if the model should continue running
        """

        self.running = True
        self.schedule = None
        self.current_id = 0
        self._n_workers = n_workers
        self._child_models = {i: WorkerModel(*args, **kwargs) for i in range(n_workers)}
        self._model_workers = {}
        self.schedule_allocation = {}
        self.grid = None
        self._dump_on_interrupt = dump_on_keyboard_interrupt

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_type, KeyboardInterrupt) and self._dump_on_keyboard_interrupt:
            log.info("Keyboard interrupt detected: dumping data")
            df = self.datacollector.get_agent_vars_dataframe()
            df.to_csv("agent_data.csv")
        for worker in tqdm(self._model_workers.values(), total=len(self._model_workers), desc="Shutting down workers"):
            shutdown_worker(worker)

        if hasattr(self, 'grid'):
            self.grid.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is not None:
            return False

        if hasattr(self, '_shared_attributes'):
            for attribute_collection in self._shared_attributes.values():
                attribute_collection.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is not None:
            return False
        return True

    def __getstate__(self):
        """Method to customize pickling behaviour
        :returns: state

        """
        state = {
            key: value for key, value in self.__dict__.items() if key not in ('_child_models', '_model_workers')
        }
        if '_model_workers' in state:
            # Record this information instead
            ...
            ...
            state['model_worker_data'] = {}
        return state

    def __setstate__(self, state):
        """Method to customize unpickling behaviour
        :returns: state

        """
        if '_model_workers' in state:
            model_worker_data = state.pop('model_worker_data')
            state['_model_workers'] = self._start_model_workers(model_worker_data)
        else:
            state['_model_workers'] = {}
        self.__dict__.update(state)

    def to_pickle(self, path):
        log.info(f"Saving to {path}")
        print(f"Saving to {path}")
        child_models = self.__dict__.pop('_child_models')
        print(f'{len(child_models)} popped')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            pickle.dump(child_models,f)
        self._child_models = child_models

    @classmethod
    def from_pickle(cls, path):
        log.info(f"Loading from {path}")
        print(f"Loading from {path}")
        with open(path, 'rb') as f:
            model_master = pickle.load(f)
            child_models = pickle.load(f)
            model_master._child_models = child_models
        return model_master
            

    def _start_model_workers(self, model_worker_data):
        """Restart model workers after serializations

        :model_worker_data: TODO

        """
        # refactor to use initialize_workers or combine?
        pass

    def initialize_workers(self) -> None:
        """ Initialize worker processes
        """
        for i in tqdm(range(self._n_workers), total=self._n_workers, desc="Initializing worker processes"):
            worker = initialize_worker(self._child_models[i])
            self._model_workers[i] = worker

    def run_model(self) -> None:
        """Run the model until the end condition is reached. Overload as
        needed.
        """
        while self.running:
            self.step()

    def step(self) -> None:
        """Step each worker node. Fill in here."""
        self.schedule.step() # Possibly the below should be included in schedule.step?

        # Advance
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            future_to_responses = {executor.submit(communicate_message, worker, ('advance', tuple())): worker for worker in self._model_workers.values()}
            for future in concurrent.futures.as_completed(future_to_responses):
                worker = future_to_responses[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (worker, exc))

        # Step
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            future_to_responses = {executor.submit(communicate_message, worker, ('step', tuple())): worker for worker in self._model_workers.values()}
            for future in concurrent.futures.as_completed(future_to_responses):
                worker = future_to_responses[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (worker, exc))

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
        # Would be better if this passed through data collectors/reporters to the workers
        self.datacollector.collect(self)

    def assign_scheduled_agents_to_child_models(self, allocator=None):
        """Assign agents to each of the child_models.
           allocator: function agents to workers: (agent, n_workers) |-> {0, 1, ..., n_workers-1}."""
        agents_to_cleanup = []
        for key, agent in tqdm(self.schedule._agents.items(), total=len(self.schedule._agents), desc="Assigning agents to workers"):
            if allocator is not None:
                k = allocator(agent, len(self._child_models))
            else:
                k = self.random.randint(0, len(self._child_models)-1)
            self._child_models[k].schedule.add(agent)
            agent.model = self._child_models[k]
            self.schedule_allocation[key] = {'worker': k}
            agents_to_cleanup.append(agent)

        for agent in agents_to_cleanup:
            self.schedule.remove(agent)

    def link_grid_to_child_models(self):
        # Move this to city implementation or vice versa?
        """Link the grid to each of the child_models"""
        grid = self.grid
        for worker_model in self._child_models.values():
            # TODO: remove dependency on grid private attributes _xxx...
            worker_model.link_shared_memory_grid(grid)

    def link_shared_attributes_to_child_models(self):
        """Link the grid to each of the child_models"""
        attrs = self._shared_attributes
        for worker_model in self._child_models.values():
            worker_model.link_shared_attributes(attrs)

        
             
        

class ParallelWorkerModel(Model):

    """ A model with additional capabilities to support parallelization of computation"""

    def link_shared_memory_grid(self, grid):
        self.grid = SharedMemoryContinuousSpace(grid.x_max,grid.y_max,x_min=grid.x_min, y_min=grid.y_min,  name=grid.name, create=False, owner=False, torus=grid.torus)
        self.grid._index_to_agent = grid._index_to_agent
        self.grid._agent_to_index = grid._agent_to_index
        self.grid.initialize_array()

    def link_shared_attributes(self, attrs):
        self._shared_attributes = {}
        for cls, attr_collection in attrs.items():
            attrs = SharedMemoryAttributeCollection(attr_collection.size)
            attrs.set_indexes(agent_to_index=attr_collection._agent_to_index)
            for name, metadata in attr_collection.attributes.items():
                new_metadata = metadata.copy()
                del new_metadata['array']
                new_metadata['owner'] = False
                attrs.add_attribute(name, new_metadata)
            self._shared_attributes[cls] = attrs


