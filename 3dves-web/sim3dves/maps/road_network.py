"""
sim3dves.world.road_network
============================
Directed road graph with A* pathfinding and grid-network factory.

Design Pattern: Builder — ``RoadNetwork.build_grid()`` is a static factory
that constructs rectangular grid networks from high-level parameters.

All path costs are Euclidean distance (metres); speed limits are stored
per-edge for future use by the vehicle kinematic model.

NF-CE-001: PEP8 compliant.
NF-CE-002: Full type annotations.
NF-CE-003: NumPy-format docstrings.
NF-CE-004: Builder pattern applied.
NF-CE-005: Requirement IDs cited inline.
Implements: ENV-006 (road network), VEH-003 (A* pathfinding).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ### Data primitives ###

@dataclass(frozen=True)
class RoadNode:
    """
    Intersection or endpoint in the road graph (ENV-006).

    Parameters
    ----------
    node_id : str
        Unique string identifier.
    position : np.ndarray
        2-D ENU position [x, y] in metres.
    """

    node_id: str
    position: np.ndarray  # shape (2,)

    # numpy arrays are not hashable; equality is by ID only.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RoadNode):
            return NotImplemented
        return self.node_id == other.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)


@dataclass(frozen=True)
class RoadEdge:
    """
    Directed road segment connecting two nodes (ENV-006).

    Parameters
    ----------
    from_id : str
        Source node ID.
    to_id : str
        Destination node ID.
    speed_limit_mps : float
        Speed limit in m/s (stored; used by future speed-zone enforcement).
    bidirectional : bool
        If True, the reverse edge (to_id → from_id) is also added at the
        same speed limit.
    """

    from_id: str
    to_id: str
    speed_limit_mps: float = 13.9   # ~50 km/h default
    bidirectional: bool = True


# ### RoadNetwork ###

class RoadNetwork:
    """
    Directed road graph supporting A* pathfinding (ENV-006, VEH-003).

    Internally maintained as an adjacency dictionary keyed by node ID::

        _adjacency[from_id] = {to_id: distance_m, ...}

    Path costs are Euclidean distances (metres).  Speed limits are stored
    separately for future velocity constraints; they do not currently
    affect pathfinding heuristics.

    Examples
    --------
    >>> rn = RoadNetwork.build_grid(rows=3, cols=3, spacing_m=100.0,
    ...                             origin=np.array([0.0, 0.0]))
    >>> path = rn.find_path("n_0_0", "n_2_2")
    >>> len(path) >= 2
    True
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, RoadNode] = {}
        # {from_id: {to_id: distance_m}}
        self._adjacency: Dict[str, Dict[str, float]] = {}
        # {(from_id, to_id): speed_limit_mps}
        self._speed_limits: Dict[Tuple[str, str], float] = {}

    # ### Mutation ###

    def add_node(self, node: RoadNode) -> None:
        """
        Register a node.  Silently replaces an existing node with the
        same ID (useful for configuration-driven setup).

        Parameters
        ----------
        node : RoadNode
            Node to register.
        """
        self._nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, {})

    def add_edge(self, edge: RoadEdge) -> None:
        """
        Add a directed road segment.

        Parameters
        ----------
        edge : RoadEdge
            Edge to add.  If ``edge.bidirectional`` is True, the reverse
            edge is also registered.

        Raises
        ------
        ValueError
            If either endpoint node has not been registered.
        """
        for nid in (edge.from_id, edge.to_id):
            if nid not in self._nodes:
                raise ValueError(
                    f"Unknown node '{nid}' — add_node() before add_edge()."
                )

        dist = float(np.linalg.norm(
            self._nodes[edge.to_id].position
            - self._nodes[edge.from_id].position
        ))

        # Forward edge
        self._adjacency[edge.from_id][edge.to_id] = dist
        self._speed_limits[(edge.from_id, edge.to_id)] = edge.speed_limit_mps

        # Reverse edge (bidirectional)
        if edge.bidirectional:
            self._adjacency.setdefault(edge.to_id, {})[edge.from_id] = dist
            self._speed_limits[(edge.to_id, edge.from_id)] = edge.speed_limit_mps

    # ### Query ###

    def find_path(self, from_id: str, to_id: str) -> List[str]:
        """
        A* shortest path between two nodes (VEH-003).

        Heuristic: Euclidean distance to goal.
        Cost:      Cumulative Euclidean edge lengths.

        Parameters
        ----------
        from_id : str
            Source node ID.
        to_id : str
            Destination node ID.

        Returns
        -------
        list of str
            Ordered list of node IDs from source to destination.
            Returns ``[]`` if no path exists or either node is unknown.
            Returns ``[from_id]`` if ``from_id == to_id``.
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            return []
        if from_id == to_id:
            return [from_id]

        goal_pos: np.ndarray = self._nodes[to_id].position

        # Priority queue: (f_score, node_id)
        open_heap: List[Tuple[float, str]] = [(0.0, from_id)]
        came_from: Dict[str, Optional[str]] = {from_id: None}
        g_score: Dict[str, float] = {from_id: 0.0}
        visited: set = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current in visited:
                continue
            visited.add(current)

            if current == to_id:
                return self._reconstruct_path(came_from, current)

            for neighbour_id, edge_dist in self._adjacency.get(current, {}).items():
                if neighbour_id in visited:
                    continue
                tentative_g = g_score[current] + edge_dist
                if tentative_g < g_score.get(neighbour_id, math.inf):
                    g_score[neighbour_id] = tentative_g
                    h = float(np.linalg.norm(
                        self._nodes[neighbour_id].position - goal_pos
                    ))
                    heapq.heappush(open_heap, (tentative_g + h, neighbour_id))
                    came_from[neighbour_id] = current

        return []  # No path found

    def nearest_node(self, xy: np.ndarray) -> Optional[str]:
        """
        Return the ID of the node whose position is closest to *xy*.

        Parameters
        ----------
        xy : np.ndarray
            Query position [x, y] in metres.

        Returns
        -------
        str or None
            Nearest node ID, or ``None`` if the network is empty.
        """
        if not self._nodes:
            return None
        return min(
            self._nodes,
            key=lambda nid: float(
                np.linalg.norm(self._nodes[nid].position - xy)
            ),
        )

    def node_position(self, node_id: str) -> np.ndarray:
        """
        Return the [x, y] position of *node_id*.

        Parameters
        ----------
        node_id : str
            A registered node ID.

        Returns
        -------
        np.ndarray
            Position vector [x, y] in metres.

        Raises
        ------
        KeyError
            If *node_id* is not registered.
        """
        return self._nodes[node_id].position

    def node_ids(self) -> List[str]:
        """Return a list of all registered node IDs."""
        return list(self._nodes.keys())

    def speed_limit(self, from_id: str, to_id: str) -> float:
        """
        Return the speed limit on edge (from_id → to_id) in m/s.

        Returns the configured default if the edge has no explicit limit.
        """
        from sim3dves.config.defaults import SimDefaults
        return self._speed_limits.get(
            (from_id, to_id), SimDefaults().ROAD_SPEED_LIMIT_MPS
        )

    def __len__(self) -> int:
        """Return the number of registered nodes."""
        return len(self._nodes)

    # ### Private helpers ###

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[str, Optional[str]], current: str
    ) -> List[str]:
        """Walk the came_from chain to reconstruct the A* path."""
        path: List[str] = []
        node: Optional[str] = current
        while node is not None:
            path.append(node)
            node = came_from[node]
        return list(reversed(path))

    # ### Factory ###

    @staticmethod
    def build_grid(
        rows: int,
        cols: int,
        spacing_m: float,
        origin: np.ndarray,
        speed_limit_mps: float = 13.9,
    ) -> "RoadNetwork":
        """
        Build a rectangular grid road network (Builder pattern).

        Nodes are named ``"n_{row}_{col}"`` (zero-indexed).
        Horizontal and vertical edges are bidirectional.

        Parameters
        ----------
        rows : int
            Number of node rows.
        cols : int
            Number of node columns.
        spacing_m : float
            Euclidean distance between adjacent nodes (m).
        origin : np.ndarray
            [x, y] position of the bottom-left node (n_0_0).
        speed_limit_mps : float
            Speed limit applied to every edge.

        Returns
        -------
        RoadNetwork
            Fully connected grid network.
        """
        rn = RoadNetwork()

        def _nid(r: int, c: int) -> str:
            return f"n_{r}_{c}"

        # Register all nodes
        for r in range(rows):
            for c in range(cols):
                pos = origin + np.array([c * spacing_m, r * spacing_m], dtype=float)
                rn.add_node(RoadNode(_nid(r, c), pos))

        # Horizontal edges (left → right)
        for r in range(rows):
            for c in range(cols - 1):
                rn.add_edge(RoadEdge(_nid(r, c), _nid(r, c + 1), speed_limit_mps))

        # Vertical edges (bottom → top)
        for r in range(rows - 1):
            for c in range(cols):
                rn.add_edge(RoadEdge(_nid(r, c), _nid(r + 1, c), speed_limit_mps))

        return rn
