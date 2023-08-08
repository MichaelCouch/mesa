"""
Mesa Attribute Module
=================

modules providing shared-memory atrributes enabling worker parallelism
"""
# Instruction for PyLint to suppress variable name errors, since we have a
# good reason to use one-character variable names for x and y.
# pylint: disable=invalid-name

# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations
import random
import string
from multiprocessing.shared_memory import SharedMemory

import numpy as np

class AttributeCollection:
    # pylint: disable=too-many-instance-attributes
    """Attribute taking a ctype value.

    This class uses a numpy array internally in shared memory to store agents data in order
    """
    _shared = False

    def __init__(
        self, size, attributes={}
    ) -> None:
        """Create a floating point attribute state space.

        Args:
            size: integer length for the space. Fixed.
            attributes: dict of {name: {'dtype':dtype}: dict of attributes and their dtypes
        """
        self.size = size
        self._cursor = 0
        self._agent_to_index: dict[str, int | None] = {}
        self._index_to_agent: dict[int, str] = {}
        self.attributes = {}
        self._initialize_attribute_arrays(attributes)

    def _initialize_attribute_arrays(self, attributes):
        """Create the numpy array used to store the attributes
        :returns: TODO

        """
        for name, metadata in attributes.items():
            self.add_attribute(name, metadata)

    def add_attribute(self, name, metadata):
        assert name not in self.attributes, f'attribute {name} already registered'
        metadata = metadata.copy()  # shallow copy only
        dtype = metadata['dtype']
        metadata['array'] = np.ndarray(shape=(self.size,), dtype=dtype)
        self.attributes[name] = metadata


    def __setitem__(self, agent_id_attr: (str, str), value: float) -> None:
        """Place a new agent in the space.

        Args:
            agent_id_attr: id, attribute name of agent object to set value of.
            value: value of agent attribute.
        """
        # Not process-safe - need to put the _agent_to_index in a
        # Manager to allow updates to be communicated?
        agent_id, attr_name = agent_id_attr
        try:
            idx = self._agent_to_index[agent_id]
        except KeyError:
            # New agent
            idx = self._cursor
            if idx >= self.size:
                space = self.size - len(self._agent_to_index)
                if space == 0:
                    raise IndexError("Attempted to add agent but no space in array")
                raise IndexError(f"Attempted to add agent but no space left at end of array. You could reindex and recover {space} slots.")
            self._cursor += 1
            self._agent_to_index[agent_id] = idx
            self._index_to_agent[idx] = agent_id

        self.attributes[attr_name]['array'][idx] = value


    def __getitem__(self, agent_id_attr: str) -> float:
        """Place a new agent in the space.

        Args:
            agent_id: id of agent object to get value of.
        """
        agent_id, attr_name = (agent_id_attr, None) if len(agent_id_attr) == 1 else agent_id_attr
        idx = self._agent_to_index[agent_id]
        if attr_name is None:
            return {name: data['array'][idx] for name, data in self.attributes.items()}
        return self.attributes[attr_name]['array'][idx]

    def __delitem__(self, agent_id: str) -> None:
        """Remove an agent from the attribute.

        Args:
            agent_id: The agent_id of the attribute to remove
        """
        idx = self._agent_to_index[agent_id]
        del self._index_to_agent[idx]
        del self._agent_to_index[agent_id]
        # Don't bother to unset the values - though this will mean a gap in the attribute array
        # Use reindex attribute method to recover this lost space

    def items(self):
        """ generator for iterating over agents in order that they are
        indexed in the array"""

        iter_cursor = 0
        counter = 0
        while counter < len(self._index_to_agent):
            try:
                agent_id = self._index_to_agent[iter_cursor]
            except KeyError:
                pass
            else:
                counter += 1
                yield agent_id, {name: data['array'][iter_cursor] for name, data in self.attributes.items()}
            iter_cursor += 1

    def reindex_attribute_array(self):
        """ reindex the attribute array and recover space lost space due to deletes
        """
        offset = 0
        for old_index in range(self._cursor):
            new_idx = old_index - offset
            try:
                agent_id = self._index_to_agent[old_index]
            except KeyError:
                offset += 1
            else:
                if new_idx != old_index: # save some time if no reindexing is needed
                    # update data values
                    for data in self.attributes.values():
                        value = data['array'][old_index]
                        data['array'][new_idx] = value
                    # update agent_id index
                    self._agent_to_index[agent_id] = new_idx
                    # update index to agent
                    self._index_to_agent[new_idx] = agent_id
                    del self._index_to_agent[old_index]
        self._cursor = max(self._index_to_agent.keys())


class SharedMemoryAttributeCollection(AttributeCollection):

    """Float attribute stored in shared memory - accessible by parallel workers"""
    _shared = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for arr in self.attributes.values():
            try:
                shm = arr['shm']
            except KeyError:
                pass
            else:
                shm.close()
                if arr['owner']:
                    shm.unlink()
        return True

    def add_attribute(self, name, metadata):
        assert name not in self.attributes, f'attribute {name} already registered'
        metadata = metadata.copy()  # shallow copy only
        if 'handle' not in metadata:
            metadata['handle'] = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
            metadata['owner'] = True
        else:
            metadata['owner'] = False
        dtype, handle, create = metadata['dtype'], metadata['handle'], metadata['owner']
        byte_array_size = self.size * np.dtype(dtype).itemsize
        metadata['shm'] = SharedMemory(name=handle, size=byte_array_size, create=create)
        buffer = metadata['shm'].buf
        metadata['array'] = np.ndarray(shape=(self.size,), dtype=dtype, buffer=buffer)
        self.attributes[name] = metadata

    def __exit__(self, exc_type, exc_val, exc_tb):
        for attribute in self.attributes.values():
            array = attribute['shm']
            array.close()
            if attribute['owner']:
                array.unlink()
        return True
