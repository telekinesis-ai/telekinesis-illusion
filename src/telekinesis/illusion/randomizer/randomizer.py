"""
Defines the Randomizer class that is responsible for executing ranodmizer nodes
on the context to generate scenes.
"""

from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from telekinesis.illusion.randomizer.randomizer_node import (
    RandomizerNode,
    NodeConfig,
    STAGE_DEFAULT,
)
from telekinesis.illusion.core.context import Context


@dataclass(slots=True)
class EdgeConfig:
    """
    A data class for providing additional edge configs for the Randomizer
    """

    enabled: bool = True


""""""


class Randomizer:
    """
    Class for executing the randomizer nodes on the context. The Randomizer is
    implemented as a Directed Acyclic Graph (Graph) consisitng of
    RandomizerNodes. The graphs internal data structures are based on an
    adjacenjy list representation using Python dictionary datastructures similar
    to NetworkX (https://pelegm-networkx.readthedocs.io/en/latest/index.html).
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize internal states.
        """
        self.graph = {}  # Dicitonary for graph attributes
        self._node = {}  # Dictinoary for node attributes
        self._succ = {}  # Dictionary for successors of nodes
        self._pred = {}  # Dictionary for predecessors of nodes

        # Currently, the graph is a simple chain of RandomizterNodes which
        # are added successively. In the later versions this should be replaced
        # with more advanced graph connections.
        self.start_node_name = None
        self.end_node_name = None

    def add_node(
        self,
        randomizer_node: RandomizerNode,
        name: str | None = None,
        node_config: NodeConfig | None = None,
    ) -> None:
        """Adds a randomizer node to the DAG.

        Notes:
            For now, 'name' must be provided and unique. (Later you can auto-generate
            a name/id if 'name is None'.)

        Args:
            name: str
                Unique node identifier.
            randomizer_node: RandomizerNode or None
                The randomizer callable/object executed for this node.
            node_config:  NodeConfig or None
                Optional execution/config metadata for the node.

        Raises:
            ValueError:
                - If 'name' is None, empty, or already exists.
                - If 'randomizer_node' is None.
        """
        if name is None:
            raise ValueError(
                "'name' must be provided (auto-generated names not implemented yet)."
            )
        if not name:
            raise ValueError("'name' must be a non-empty string.")
        if randomizer_node is None:
            raise ValueError("'randomizer_node' must be provided.")

        if name in self._node:
            raise ValueError(f"Node {name!r} already exists.")

        self._node[name] = {
            "randomizer": randomizer_node,
            "config": node_config if node_config is not None else NodeConfig(),
        }
        self._succ[name] = {}
        self._pred[name] = {}

    def add_edge(
        self,
        from_node_name: str,
        to_node_name: str,
        edge_config: EdgeConfig | None = None,
    ) -> None:
        """
        Adds a directed edge 'from_node_name' -> 'to_node_name'.

        Notes:
            For now, 'name' must be provided and unique. (Later you can auto-generate
            a name/id if 'name is None'.)

        Args:
            from_node_name: str
                Upstream node name.
            to_node_name: str
                Downstream node name.
            edge_config: EdgeConfig or None
                Optional configuration/metadata for this edge.

        Raises:
            ValueError:
                - If either node does not exist.
        """
        if from_node_name not in self._node:
            raise ValueError(
                f"Randomizer node {from_node_name!r} does not exist."
            )
        if to_node_name not in self._node:
            raise ValueError(
                f"Randomizer node {to_node_name!r} does not exist."
            )

        # Share the same edge-attr dict from both sides (succ/pred), like NetworkX.
        edge_attrs = self._succ[from_node_name].get(to_node_name)
        if edge_attrs is None:
            edge_attrs = {}
            self._succ[from_node_name][to_node_name] = edge_attrs
            self._pred[to_node_name][from_node_name] = edge_attrs

        # Store config in a consistent key so you can add more edge metadata later.
        edge_attrs["config"] = (
            edge_config if edge_config is not None else EdgeConfig()
        )

    def add_randomizer(
        self,
        randomizer_node: RandomizerNode,
        node_name: str,
        node_config: NodeConfig | None = None,
        edge_config: EdgeConfig | None = None,
    ) -> None:
        """
        Append a RandomizerNode to a linear chain:
            - If this is the first node, it becomes the 'start' (and 'end').
            - Otherwise, it is appended as successor of the current 'end'.

        Notes:
            For now this method simple creates a chain of RandomizerNodes. In
            later versions it should allow to define explicit dependencies such
            as prioirity in execution. Also it should allow autogenerated names.

        Args:
            randomizer_node: RandomizerNode
                A RandomizerNode to be appended to the DAG.
            node_name: str
                The unique name of the RandomizerNode
            node_config: NodeConfig or None
                Optional configurations for the node.
            edge_config: EdgeConfig or None
                Optional configurations for the edge.
        """

        self.add_node(
            randomizer_node=randomizer_node,
            name=node_name,
            node_config=node_config,
        )

        if self.start_node_name is None:
            # First node in the chain
            self.start_node_name = node_name
            self.end_node_name = node_name
            return

        # Append: end -> new
        self.add_edge(self.end_node_name, node_name, edge_config=edge_config)
        self.end_node_name = node_name

    def get_randomizer_node(self, node_name: str) -> RandomizerNode | None:
        """
        Get the RandomizerNode registered under 'node_name', or None if no
        such node exists.

        Args:
            node_name: str
                The unique name of the RandomizerNode.

        Returns:
            The RandomizerNode, or None when the name is not registered.
        """
        entry = self._node.get(node_name)
        return entry["randomizer"] if entry else None

    def replace_randomizer(
        self,
        node_name: str,
        randomizer_node: RandomizerNode,
    ) -> None:
        """
        Swap the RandomizerNode registered under 'node_name' for another one,
        leaving the DAG topology and the node's config untouched.

        Intended for interactive tools (e.g. the Blender spec editor) that
        re-tune a randomizer's parameters between runs without paying for a
        full worker rebuild. 'randomize()' recomputes the execution order on
        every call, so the replacement takes effect immediately.

        Args:
            node_name: str
                The unique name of the RandomizerNode to replace.
            randomizer_node: RandomizerNode
                The replacement RandomizerNode.

        Raises:
            KeyError: When no node is registered under 'node_name'.
        """
        if node_name not in self._node:
            raise KeyError(f"No randomizer node named '{node_name}'.")
        self._node[node_name]["randomizer"] = randomizer_node

    def _topological_order(self) -> list[str]:
        """
        Return a stable topological execution order for the DAG.

        Uses Kahn's algorithm. If multiple nodes are ready at the same time,
        tie-breaking follows insertion order of 'self._node'.

        Returns:
            List of node names in execution order.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        indegree: dict[str, int] = {
            n: len(self._pred.get(n, {})) for n in self._node
        }

        # Stable tie-break: iterate in insertion order of self._node.
        ready: list[str] = [n for n in self._node if indegree[n] == 0]
        order: list[str] = []

        while ready:
            n = ready.pop(0)  # FIFO => stable
            order.append(n)

            for child in self._succ.get(n, {}):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)

        if len(order) != len(self._node):
            raise ValueError("Cycle detected: Randomizer DAG must be acyclic.")

        return order

    def get_node_stages(self) -> dict[str, str]:
        """
        Get the stage each registered node belongs to, keyed by node name.

        Returns:
            Mapping of node name to its NodeConfig.stage.
        """
        return {
            name: entry["config"].stage for name, entry in self._node.items()
        }

    def randomize(
        self,
        context: Context,
        stages: AbstractSet[str] | None = None,
    ):
        """
        Execute the RandomizerNodes in dependency order.

        Each node must be stored under 'self._node[name]["randomizer"]' and
        expose a '.randomize(...)' method.

        Args:
            context: Context
                A shared context with assets.
            stages: set[str] or None
                When given, only nodes whose NodeConfig.stage is in this set are
                executed. Nodes left at STAGE_DEFAULT always run, so a node that
                doesn't opt into a stage is never skipped by accident. Passing
                None (the default) runs everything - this is what a real
                generation run does.
        """
        order = self._topological_order()
        for node_name in order:
            node_entry = self._node[node_name]
            node_config = node_entry["config"]

            if not node_config.enabled:
                continue

            if (
                stages is not None
                and node_config.stage != STAGE_DEFAULT
                and node_config.stage not in stages
            ):
                continue

            randomizer_node = node_entry["randomizer"]
            randomizer_node.randomize(context)
