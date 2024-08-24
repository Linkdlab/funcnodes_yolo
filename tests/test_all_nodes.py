from all_nodes_test_base import TestAllNodesBase
import funcnodes as fn
import funcnodes_yolo as fnmodule
import dataclasses


class TestAllNodes(TestAllNodesBase):
    ### in this test class all nodes should be triggered at least once to mark them as testing
    async def test_first_node(self):
        node = fnmodule.FirstNode()
        node.inputs["x"].value = "foo"
        await node
        self.assertEqual(node.get_output("out").value, "bar")
