# 6.0002 Problem Set 5
# Graph optimization
# Name:
# Collaborators:
# Time:

#
# Finding shortest paths through MIT buildings
#
import random
from sys import path
import unittest

from numpy import true_divide
from graph import Digraph, Node, WeightedEdge, Edge
import time
from map import PolarLocation
from RandomWalks import Field, Location
import matplotlib.pyplot as plt

#
# Problem 2: Building up the Campus Map
#
# Problem 2a: Designing your graph
#
# What do the graph's nodes represent in this problem? What
# do the graph's edges represent? Where are the distances
# represented?
#
# Answer:
#

# Problem 2b: Implementing load_map
def load_map(map_filename):
    """
    Parses the map file and constructs a directed graph

    Parameters:
        map_filename : name of the map file

    Assumes:
        Each entry in the map file consists of the following four positive
        integers, separated by a blank space:
            From To TotalDistance DistanceOutdoors
        e.g.
            32 76 54 23
        This entry would become an edge from 32 to 76.

    Returns:
        a Digraph representing the map
    """
    

    g = Digraph()
    print("Loading map from file...")
    NewList = []
    with open(map_filename, 'r') as f:
        s = f.read()
    splitted= s.split('\n')
    for e in splitted:
        splittedInside = e.split(' ')
        NewList.append(splittedInside)

    NewList.pop()
    for index in range(len(NewList)):
        src, dest, TotalDist, OutsideDistance = NewList[index]
        if not g.has_node(Node(src)):
            g.add_node(Node(src))
        if not g.has_node(Node(dest)):
            g.add_node(Node(dest))
        g.add_edge(WeightedEdge(Node(src), Node(dest), TotalDist, OutsideDistance))
        #print(g.has_node(SourceBuilding))
    return g  

# Problem 2c: Testing load_map
# Include the lines used to test load_map below, but comment them out


# Problem 3: Finding the Shorest Path using Optimized Search Method

# Problem 3a: Objective function

# What is the objective function for this problem? What are the constraints?

# Answer:


"""
Finds the shortest path between buildings subject to constraints.

Parameters:
    digraph: Digraph instance
        The graph on which to carry out the search
    start: string
        Building number at which to start
    end: string
        Building number at which to end
    path: list composed of [[list of strings], int1, int]
        Represents the current path of nodes being traversed. Contains
        a list of node names, total distance traveled, and total
        distance outdoors.
    max_dist_outdoors: int
        Maximum distance spent outdoors on a path
    best_dist: int
        The smallest distance between the original start and end node
        for the initial problem that you are trying to solve
    best_path: list of strings
        The shortest path found so far between the original start
        and end node.

Returns:
    A tuple with the shortest-path from start to end, represented by
    a list of building numbers (in strings), [n_1, n_2, ..., n_k],
    where there exists an edge from n_i to n_(i+1) in digraph,
    for all 1 <= i < k and the distance of that path.

    If there exists no path that satisfies max_total_dist and
    max_dist_outdoors constraints, then return None.
"""

def get_best_path(digraph, start, end, path, max_dist_outdoors, best_dist, best_path):
    if start != end:
        path[0].append(start)
    if start == end:
        tmp = path[:] 
        tmp[0].append(start)
        return tmp
    if not digraph.has_node(start) or not digraph.has_node(end):
        raise KeyError  
    if digraph.has_node(start) and digraph.has_node(end):
        for edges in digraph.get_edges_for_node(start):
            if edges.get_destination() not in path[0]:
                if path[2] + edges.get_outdoor_distance() <= max_dist_outdoors:
                    if best_path == None or best_dist == 0 or path[1] + edges.get_total_distance() < best_dist:
                        TotalDistance = edges.get_total_distance()
                        OutDistance = edges.get_outdoor_distance()
                        tmp = path[0][:] #Bunu ekledim çözüldü.
                        ConstructedPath = [tmp, 0, 0]
                        ConstructedPath[1] = path[1] + TotalDistance
                        ConstructedPath[2] = path[2] + OutDistance
                        newPath = get_best_path(digraph, edges.get_destination(), end, ConstructedPath, max_dist_outdoors, best_dist, best_path)
                        if newPath != None:
                            best_path = newPath[0]
                            best_dist = newPath[1]
        return (best_path, best_dist, path[2])
      
# Problem 3c: Implement directed_dfs
def directed_dfs(digraph, start, end, max_total_dist, max_dist_outdoors):
    """
    Finds the shortest path from start to end using a directed depth-first
    search. The total distance traveled on the path must not
    exceed max_total_dist, and the distance spent outdoors on this path must
    not exceed max_dist_outdoors.

    Parameters:
        digraph: Digraph instance
            The graph on which to carry out the search
        start: string
            Building number at which to start
        end: string
            Building number at which to end
        max_total_dist: int
            Maximum total distance on a path
        max_dist_outdoors: int
            Maximum distance spent outdoors on a path

    Returns:
        The shortest-path from start to end, represented by
        a list of building numbers (in strings), [n_1, n_2, ..., n_k],
        where there exists an edge from n_i to n_(i+1) in digraph,
        for all 1 <= i < k

        If there exists no path that satisfies max_total_dist and
        max_dist_outdoors constraints, then raises a ValueError.
    """
    Shortest_Path, DistanceSpend, DistanceOutside = get_best_path(digraph, Node(start), Node(end), [[], 0, 0], max_dist_outdoors, 0, None)
    if Shortest_Path == None:
        raise ValueError('None')
    if DistanceSpend <= max_total_dist and DistanceOutside <= max_dist_outdoors:
        for index in range(len(Shortest_Path)):
            Shortest_Path[index] = str(Shortest_Path[index])
        return Shortest_Path
    else: 
        raise ValueError

def Breadth_First_Search(digraph, start, end, path):
    random.seed(0)
    """
    Digraph: graph 
    queue: nodes to be searched 
    path : [[local path], distanceSpendTotal, distanceSpendOutside] until that node
    """
    queue = {}
    print(f'queue:{queue}')
    path[0].append(start)
    print(path)
    for edges in digraph.get_edges_for_node(start):
        print(edges)
        print('girdim')
        dest = edges.get_destination()
        queue[dest] = [edges.get_total_distance(), edges.get_outdoor_distance()]    
    print(queue)
    for nodes in queue:
        if nodes == end:
            print('girdim ben buraya')
            return path
    for nodes in queue:
        print('son kale')
        path[1] += queue[nodes][0]
        path[2] += queue[nodes][1]
        shortestPath = Breadth_First_Search(digraph, nodes, end, path) 
        print(f'shortestPath: {shortestPath}')
    if shortestPath != None:
        return shortestPath
    print(queue)
    # tmp = path[:]
    # tmp += dest
def BFS(digraph, start, end):
    path = Breadth_First_Search(digraph, Node(start), Node(end), [[], 0, 0])      
    return path

def load_map_asMap(map_filename):
    g = Digraph()
    f = Field()
    print("Loading map from file...")
    NewList = []
    with open(map_filename, 'r') as field:
        s = field.read()
    splitted= s.split('\n')
    for e in splitted:
        splittedInside = e.split(' ')
        NewList.append(splittedInside)
    NewList.pop()
    maxEdges = {}
    for index in range(len(NewList)):
        src, dest, TotalDist, OutsideDistance = NewList[index]
        if not g.has_node(Node(src)):
            g.add_node(Node(src))
        if not g.has_node(Node(dest)):
            g.add_node(Node(dest))
        g.add_edge(WeightedEdge(Node(src), Node(dest), TotalDist, OutsideDistance))
        maxEdges[src] = 0
        #print(g.has_node(SourceBuilding))
    for index in range(len(NewList)):
        src = NewList[index][0]
        maxEdges[src] += 1
    InitialNode = max(maxEdges, key= maxEdges.get)
    print(NewList)
    print(InitialNode)
    print(type(f))
    f.addDrunks(Node(InitialNode), PolarLocation(0))
    plt.plot(f.getLoc(Node(InitialNode)).getX(), f.getLoc(Node(InitialNode)).getY(), 'mo') 
    for index in range(len(g.get_nodes())):
        currentNode = g.get_nodes()[index]
        for edge in g.get_edges_for_node(currentNode):
            print('okkk')
            print(currentNode)
            print(edge)
            print(edge.get_source())
            print(f.FirstDrunk())
            if index == 0:
                if edge.get_source() == f.FirstDrunk():
                    print('buna da girdim')
                    angle = random.randint(0,360)
                    print('angle', str(angle))
                    f.addDrunks(edge.get_destination(), PolarLocation(5 * edge.get_total_distance(), random.randint(0,360)))
                    print(f.getLoc(edge.get_destination()))
                    plt.plot(f.getLoc(edge.get_destination()).getX(), f.getLoc(edge.get_destination()).getY(), 'mo') 
            else:
                if edge.get_source() == currentNode and not edge.get_destination() in f.getDrunks():
                    print('buna da girdim')
                    angle = random.randint(0,360)
                    print('angle', str(angle))
                    f.addDrunks(edge.get_destination(), PolarLocation(5 * edge.get_total_distance(), random.randint(0,360)))
                    print(f.getLoc(edge.get_destination()))
                    plt.plot(f.getLoc(edge.get_destination()).getX(), f.getLoc(edge.get_destination()).getY(), 'k+' )
    index = 0
    for x in f.getDrunks():
        index += 1
    print('asdasdasdasd', index)
    plt.show() 


print(load_map_asMap(r"C:\Users\arda_\Desktop\VS Code 2\PS2 (1)\mit_map.txt"))

# graph = load_map(r"C:\Users\arda_\Desktop\VS Code 2\PS2 (1)\mit_map.txt")
# shrt= directed_dfs(graph, '32', '56', 9999,0)
# print(shrt)
# ================================================================
# Begin tests -- you do not need to modify anything below this line
# ================================================================

# graph = load_map(r"C:\Users\arda_\Desktop\VS Code 2\PS2 (1)\mit_map.txt")
# X = directed_dfs(graph, '32', '56', 9999, 9999)
# print(X)

# class Ps2Test(unittest.TestCase):
#     LARGE_DIST = 99999

#     def setUp(self):
#         self.graph = load_map(r"C:\Users\arda_\Desktop\VS Code 2\PS2 (1)\mit_map.txt")
#         print(self.graph)
#     def test_load_map_basic(self):
#         self.assertTrue(isinstance(self.graph, Digraph))
#         self.assertEqual(len(self.graph.nodes), 37)
#         all_edges = []
#         for _, edges in self.graph.edges.items():
#             all_edges += edges  # edges must be dict of node -> list of edges
#         all_edges = set(all_edges)
#         self.assertEqual(len(all_edges), 129)

#     def _print_path_description(self, start, end, total_dist, outdoor_dist):
#         constraint = ""
#         if outdoor_dist != Ps2Test.LARGE_DIST:
#             constraint = "without walking more than {}m outdoors".format(
#                 outdoor_dist)
#         if total_dist != Ps2Test.LARGE_DIST:
#             if constraint:
#                 constraint += ' or {}m total'.format(total_dist)
#             else:
#                 constraint = "without walking more than {}m total".format(
#                     total_dist)

#         print("------------------------")
#         print("Shortest path from Building {} to {} {}".format(
#             start, end, constraint))

#     def _test_path(self,
#                    expectedPath,
#                    total_dist=LARGE_DIST,
#                    outdoor_dist=LARGE_DIST):
#         start, end = expectedPath[0], expectedPath[-1]
#         self._print_path_description(start, end, total_dist, outdoor_dist)
#         dfsPath = directed_dfs(self.graph, start, end, total_dist, outdoor_dist)
#         print("Expected: ", expectedPath)
#         print("DFS: ", dfsPath)
#         self.assertEqual(expectedPath, dfsPath)

#     def _test_impossible_path(self,
#                               start,
#                               end,
#                               total_dist=LARGE_DIST,
#                               outdoor_dist=LARGE_DIST):
#         self._print_path_description(start, end, total_dist, outdoor_dist)
#         with self.assertRaises(ValueError):
#             directed_dfs(self.graph, start, end, total_dist, outdoor_dist)

#     def test_path_one_step(self):
#         self._test_path(expectedPath=['32', '56'])

#     def test_path_no_outdoors(self):
#         self._test_path(
#             expectedPath=['32', '36', '26', '16', '56'], outdoor_dist=0)

#     def test_path_multi_step(self):
#         self._test_path(expectedPath=['2', '3', '7', '9'])

#     def test_path_multi_step_no_outdoors(self):
#         self._test_path(
#             expectedPath=['2', '4', '10', '13', '9'], outdoor_dist=0)

#     def test_path_multi_step2(self):
#         self._test_path(expectedPath=['1', '4', '12', '32'])

#     def test_path_multi_step_no_outdoors2(self):
#         self._test_path(
#             expectedPath=['1', '3', '10', '4', '12', '24', '34', '36', '32'],
#             outdoor_dist=0)

#     def test_impossible_path1(self):
#         self._test_impossible_path('8', '50', outdoor_dist=0)

#     def test_impossible_path2(self):
#         self._test_impossible_path('10', '32', total_dist=100)


# if __name__ == "__main__":
#     unittest.main()