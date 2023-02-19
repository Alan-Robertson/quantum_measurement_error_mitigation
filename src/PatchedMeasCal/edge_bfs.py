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
            Adds an edge to the Node
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


def cmap_djikstra(cmap, n_qubits, root = 0):
        
    distances = [{i:0} for i in range(n_qubits)]
    
    cmap = copy.deepcopy(cmap)
    traversed = [root]
    edges_used = []
    nodes_found = []
    
    for c in cmap:
        distances[c[0]][c[1]] = 1
    
    while len(cmap) > 0:
        for t in traversed:
            for c in cmap:
                if c[0] == t:
                    edges_used.append(c)
                    nodes_found.append(c[1])

                    # Join
                    distances_t = distances[t]
                    distances_e = distances[c[1]]
                    for d_t in distances_t:
                        if d_t not in distances_e:
                            distances_e[d_t] = distances_t[d_t] + 1
                        else:
                            distances_e[d_t] = min(distances_e[d_t], distances_t[d_t] + 1)
                            distances_t[d_t] = min(distances_e[d_t] + 1, distances_t[d_t])
        traversed = nodes_found
        nodes_found = []
        
        for e in edges_used:
            if e in cmap:
                cmap.remove(e)
            if e[::-1] in cmap:
                cmap.remove(e[::-1])
        edges_used = []
            
    # Symmetric Cleanup
    for i, d in enumerate(distances):
        for j in range(n_qubits):
            if j not in d and i in distances[j]:
                d[j] = distances[j][i]
            if j not in d and i not in distances[j]:   
                #Grown at the same time, hence not present in either
                dist = float('inf')
                if i not in distances[j]:
                    for k in distances[j]:
                        if i in distances[k] and j != k:
                            dist = min(dist, distances[j][k] + distances[k][i])
                distances[i][j] = dist
                distances[j][i] = dist

    return distances

def djikstra_tree(coupling_map, n_qubits, root = 0):
    traversed = []
    front_nodes = [root]

    tree_cmap = []
    coupling_map = copy.deepcopy(coupling_map)


    while len(traversed) < n_qubits:
        next_front = []
        for t in front_nodes:
            for c in coupling_map:
                if c[0] == t:
                    if c[1] not in traversed and c[1] not in front_nodes and c[1] not in next_front:
                        next_front.append(c[1])
                        tree_cmap.append(c)
                        tree_cmap.append(c[::-1])
                        coupling_map.remove(c)
                        coupling_map.remove(c[::-1])

        traversed += front_nodes
        front_nodes = next_front

    return tree_cmap

# This would be much faster with a proper graph structure
def cmap_shortest_path(start, end, distance_map, cmap):
    distance = distance_map[start][end]
    path = [start]
    curr_node = start
    while curr_node != end:
        next_node = None
        for i in distance_map[curr_node]:
            if [curr_node, i] in cmap or [i, curr_node] in cmap:
                if distance_map[i][end] == distance_map[curr_node][end] - 1:
                    next_node = i
                    break
        path.append(next_node)
        curr_node = next_node
    return path