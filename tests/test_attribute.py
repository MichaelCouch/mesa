"""
Test the Grid objects.
"""
import random
import unittest
from unittest.mock import Mock, patch
import numpy as np
from mesa.attribute import AttributeCollection, SharedMemoryAttributeCollection

class TestAttributeCollection(unittest.TestCase):
    """
    Testing an attribute.
    """

    def setUp(self):
        """
        Set up attributes
        """
        self.agents = ['a', 'b', 'c', 'd', 'e']
        self.float_data = [1.9, 2, 17.1, -1, 0]
        self.integer_data = [-2, 5, 6, 0, 12]
        attributes = {
            'float': {'dtype': np.float64},
            'int': {'dtype': np.int32},
        }

        self.attributes = AttributeCollection(len(self.agents) + 5, attributes)

        for agent, flt, intg in zip(self.agents, self.float_data, self.integer_data):
            self.attributes[agent,'float'] = flt
            self.attributes[agent,'int'] = intg

    def test_getitem(self):

        """
        Ensure that the agents have correct data.
        """
        for agent, flt, intg in zip(self.agents, self.float_data, self.integer_data):
            assert self.attributes[agent, 'float'] == flt
            assert self.attributes[agent, 'int'] == intg

    def test_setitem(self):

        """
        Ensure that the agents have correct data.
        """
        i = 3
        agent, flt, intg = self.agents[i], 22.2, 666
        self.float_data[i] = flt
        self.integer_data[i] = intg

        self.attributes[agent, 'float'] = flt
        self.attributes[agent, 'int'] = intg

        for agent2, flt2, intg2 in zip(self.agents, self.float_data, self.integer_data):
            assert self.attributes[agent2, 'float'] == flt2
            assert self.attributes[agent2, 'int'] == intg2

    def test_deleteitem(self):

        """
        Ensure that the agents have correct data.
        """
        i = 4
        agent, flt, intg = self.agents[i], self.float_data[i], self.integer_data[i]
        del self.agents[i]
        del self.float_data[i]
        del self.integer_data[i]

        del self.attributes[agent]
 
        with self.assertRaises(KeyError):
            self.attributes[agent, 'float']
        with self.assertRaises(KeyError):
            self.attributes[agent, 'int']
        for agent2, flt2, intg2 in zip(self.agents, self.float_data, self.integer_data):
            assert self.attributes[agent2, 'float'] == flt2
            assert self.attributes[agent2, 'int'] == intg2
        # Cleanup after test
        self.attributes[agent, 'float'] = flt
        self.attributes[agent, 'int'] = intg
        self.agents.append(agent)
        self.float_data.append(flt)
        self.integer_data.append(intg2)


    def test_reindex_array(self):
        pass
        #not sure if we should test this

        i = 2
        agent, flt, intg = self.agents[i], self.float_data[i], self.integer_data[i]
        del self.agents[i]
        del self.float_data[i]
        del self.integer_data[i]

        del self.attributes[agent]
        self.attributes.reindex_attribute_array()
        self.assertEqual(len(self.attributes._agent_to_index),len(self.integer_data))
        self.assertEqual(max(self.attributes._agent_to_index.values())+1, len(self.attributes._agent_to_index))
        self.assertEqual(max(self.attributes._index_to_agent.keys())+1, len(self.attributes._agent_to_index))
        for agent2, flt2, intg2 in zip(self.agents, self.float_data, self.integer_data):
            self.assertEqual(self.attributes[agent2, 'float'], flt2)
            self.assertEqual(self.attributes[agent2, 'int'],intg2)
        # Cleanup after test
        self.attributes[agent, 'float'] = flt
        self.attributes[agent, 'int'] = intg
        self.agents.append(agent)
        self.float_data.append(flt)
        self.integer_data.append(intg2)


class TestSharedMemoryAttributeCollection(TestAttributeCollection):


    def setUp(self):
        """
        Set up attributes
        """
        self.agents = ['a', 'b', 'c', 'd', 'e']
        self.float_data = [1.9, 2, 17.1, -1, 0]
        self.integer_data = [-2, 5, 6, 0, 12]
        attributes = {
            'float': {'dtype': np.float64},
            'int': {'dtype': np.int32},
        }

        self.attributes = SharedMemoryAttributeCollection(len(self.agents) + 5, attributes)

        for agent, flt, intg in zip(self.agents, self.float_data, self.integer_data):
            self.attributes[agent,'float'] = flt
            self.attributes[agent,'int'] = intg

    def tearDown(self):
        self.attributes.__exit__(None,None,None)

if __name__ == '__main__':
    unittest.main()
