'''
    Coupling Map Graph and Graph Node objects
    Builds the patches for the patched measurement calibration from a
     coupling map and a set of edges
'''
import copy
from PatchedMeasCal.utils import Progressbar, vprint, vProgressbar, vtick

class GraphNode():
    '''
    GraphNode
    Node object for a coupling map graph
    '''
    def __init__(self, adjacent_nodes=None, label=''):
        if adjacent_nodes is None:
            self.adjacent_nodes = set()
        else:
            self.adjacent_nodes=set(adjacent_nodes)

        self.visited = False
        self.label = label
        self.n_edges = 0

    def add_edge(self, graph_node):
        '''
            Ads an edge to the Node
        '''
        self.adjacent_nodes.add(graph_node)
        self.n_edges += 1

    def __repr__(self):
        '''
        repr for the node, returns the node's label
        '''
        return str(self.label)


class CouplingMapGraph():
    '''
    CouplingMapGraph
    Graph of Node objects for constructing the measurement patches
    '''
    def __init__(self, coupling_map):
        if len(coupling_map) < 1:
            raise Exception("Empty Coupling Map")

        # Clear duplicates from the coupling_map
        self.coupling_map = []
        self.cmap_no_duplicate_edges(coupling_map)
        self.graph = self.coupling_map_to_graph(self.coupling_map)

    def __call__(self, *args, **kwargs) -> list:
        return self.edge_patches(*args, **kwargs)

    def __repr__(self):
        return str(self.coupling_map)

    def __getitem__(self, i):
        return self.graph.__getitem__(i)

    def __setitem__(self, i):
        return self.graph.__setitem__(i)

    def cmap_no_duplicate_edges(self, coupling_map):
        '''
            Culls duplicate edges from the coupling map
            Before saving the new culled map to the class's instance
            of the coupling map
        '''
        for edge in coupling_map:
            if edge not in self.coupling_map and edge[::-1] not in self.coupling_map:
                self.coupling_map.append(edge)


    def edge_patches(self, edges=None, distance=2, participating_qubits=None, verbose=False) -> list:
        '''
            Constructs the edge measurement patches 
        '''
        if edges is None:
            edges = self.coupling_map

        patches = []
        patched_edges = copy.deepcopy(edges)

        pb = vProgressbar(verbose, 20, len(edges), "\tBuilding Edge Patches")
        while len(patched_edges) > 0:
            for node in self.graph:
                node.visited = False

            if pb is not None:
                pb.invoked = len(edges) - len(patched_edges)
            vtick(verbose, pb)

            # Pick an initial edge
            initial_edge = patched_edges[0]
            patched_edges.remove(initial_edge)
            boundary = [self.graph[i] for i in initial_edge]
            patches.append([initial_edge])

                # Set initials
            for node in boundary:
                node.visited = True

            new_edge = True
            while new_edge:
                # Dead zone from previous patches
                boundary = self.update_boundary(boundary, distance=distance)

                # Find new edge
                new_edge = False
                for node in boundary:
                    if not node.visited:
                        edge = self.new_pair(node)
                        # If an edge is found
                        if edge is not None:
                            # Remove it from the patched edges
                            patch_edge = False
                            if edge in patched_edges:
                                patched_edges.remove(edge)
                                patches[-1].append(edge)
                                patch_edge = True
                            elif edge[::-1] in patched_edges:
                                patched_edges.remove(edge[::-1])
                                patches[-1].append(edge[::-1])
                                patch_edge = True

                            if patch_edge:
                                tmp_boundary = [self.graph[i] for i in edge]
                                boundary.remove(node)
                                # Push out boundary from newly added patch
                                # This will update the visited flags
                                # without extending the current boundary
                                self.update_boundary(tmp_boundary, distance=distance)

                # If there are still unvisited nodes on the boundary then keep going
                for node in boundary:
                    if node.visited is False:
                        new_edge = True
                    node.visited = True
        return patches

    @staticmethod
    def coupling_map_to_graph(coupling_map:list):
        '''
            Builds a graph of GraphNodes from a coupling map
        '''
        num_nodes = max(max(coupling_map, key=lambda x: max(x)))
        graph = [GraphNode(label=i) for i in range(num_nodes + 1)]
        for edge in coupling_map:
            graph[edge[0]].add_edge(graph[edge[1]])
            graph[edge[1]].add_edge(graph[edge[0]])
        return graph

    @staticmethod
    def new_pair(node):
        '''
            Finds the next unvisited node from a given starting node, forming an edge
        '''
        for adj_node in node.adjacent_nodes:
            if adj_node.visited is False:
                return [node.label, adj_node.label]
        return None

    @staticmethod
    def update_boundary(boundary, distance=2):
        '''
           Updates the boundary of edges that may be added to the list of
        '''
        next_edges = set([])
        for i in range(distance):
            for node in boundary:
                for adj_node in node.adjacent_nodes:
                    next_edges.add(adj_node)
                    if i < distance - 1:
                        adj_node.visited = True

            boundary = list(next_edges)
            next_edges = set([])

        # Clear previously visited nodes from the boundary
        for node in boundary:
            if node.visited is True:
                boundary.remove(node)

        return boundary
