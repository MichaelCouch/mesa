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
import multiprocessing
from typing import Any

from mesa.datacollection import DataCollector, ParallelDataCollector
from mesa.model import Model
from mesa.server import initialize_worker, shutdown_worker, communicate_message, listener_api, read_listener
from mesa.space import SharedMemoryContinuousSpace
from mesa.attribute import SharedMemoryAttributeCollection
from tqdm import tqdm
import logging as log
import os

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

        if hasattr(self, '_listener_process'):
            self._listener_process.terminate()

        if hasattr(self, 'grid'):
            self.grid.__exit__(exc_type, exc_val, exc_tb)

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

    def to_pickle(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        log.info(f"Saving to {dir_path}")

        child_models = self.__dict__.pop('_child_models')
        self._child_models = {id_: None for id_ in child_models}
        model_workers = self.__dict__.pop('_model_workers')
        self._model_workers = {id_: None for id_ in model_workers}
        if hasattr(self, '_listener_process'):
            listener_process, listener_queue = self.__dict__.pop('_listener_process'), self.__dict__.pop('_listener_queue')
        else: 
            listener_process, listener_queue = None, None

        self._model_workers = {id_: None for id_ in model_workers}

        # TODO: handle the child-model/worker-model dual existance better
        with open(os.path.join(dir_path, 'model_master.pkl'), 'wb') as f:
            pickle.dump(self, f)
        self._child_models = child_models
        self._model_workers = model_workers
        if listener_queue is not None:
            self._listener_process = listener_process
            self._listener_queue = listener_queue

        for id_, child in self._child_models.items():
            if child is not None:
                child.to_pickle(os.path.join(dir_path, f'child_model_{id_}.pkl'))

        args = {id_: (os.path.join(dir_path, f'child_model_{id_}.pkl'),)
                for id_ in self._model_workers}
        self.instruct_workers('to_pickle', args)


    def instruct_workers(self, operation, args):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            future_to_responses = {executor.submit(
                    communicate_message,
                    worker,
                    (operation, args[id_])
                ): worker for id_, worker in self._model_workers.items()
            }
            for future in concurrent.futures.as_completed(future_to_responses):
                worker = future_to_responses[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (worker, exc))
                    raise exc


    @staticmethod
    def from_pickle(dir_path, with_children=False):
        log.info(f"Loading from {dir_path}")
        with open(os.path.join(dir_path,'model_master.pkl'), 'rb') as f:
            model_master = pickle.load(f)
        if with_children:
            for id_ in list(model_master._child_models):
                path = os.path.join(dir_path, f'child_model_{id_}.pkl')
                child_model = ParallelWorkerModel.from_pickle(path)
                model_master._child_models[id_] = child_model
        return model_master


    def _start_model_workers(self, model_worker_data):
        """Restart model workers after serializations

        :model_worker_data: TODO

        """
        # refactor to use initialize_workers or combine?
        pass

    def initialize_workers(self, from_pickle=None) -> None:
        """ Initialize worker processes
        """
        if not hasattr(self, '_child_models'):
            self._child_models = {i: None for i in range(self._n_workers)}
        child_model_ids = list(self._child_models.keys())
        for id_ in tqdm(child_model_ids, total=self._n_workers, desc="Initializing worker processes"):
            worker = initialize_worker(
                self._child_models[id_],
                pickle_path=os.path.join(from_pickle, f'child_model_{id_}.pkl')
            )
            self._model_workers[id_] = worker

    def start_listener(self):
        # move this to server.py?
        listener_queue = multiprocessing.Queue()
        listener_process = multiprocessing.Process(target=listener_api, args=(listener_queue,))
        listener_process.start()
        self._listener_queue = listener_queue
        self._listener_process = listener_process

    def read_from_listener(self):
        return read_listener(self._listener_queue)


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
        args = {id_: tuple() for id_ in self._model_workers}
        self.instruct_workers('advance', args)

        # Step
        args = {id_: tuple() for id_ in self._model_workers}
        self.instruct_workers('step', args)

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

    def to_pickle(self, path):
        "serialize model as a pickle"
        log.debug(f"Saving to {path}")
        with open(path, 'wb') as f:
            pickle.dump(self,f)

    @staticmethod
    def from_pickle(path):
        "deserialize pickle"
        log.debug(f"Loading from {path}")
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model



