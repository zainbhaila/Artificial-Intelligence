"""
File: fsearch.py
Author: Dana Nau <nau@cs.umd.edu>
Last updated: Sept 5, 2019

This file provides a Python implementation of the "Graph-Search-Redo" algorithm.
For information on how to use it, see the docstring for fsearch.main.
"""

import sys                              # We need flush() and readline()


class Node():
    """
    Each node includes ID#, state, parent node, g-value, h-value, and list of children.
    """
    def __init__(self,state,parent,cost,h_value):
        """
        Args: current state, parent node, cost of transition from parent state
        to current state, and h(current state)
        """
        global node_count
        self.state = state
        self.parent = parent
        node_count += 1                      # total number of nodes
        self.id = node_count                 # this node's ID number
        if parent: 
            parent.children.append(self)
            self.depth = parent.depth + 1    # depth in the search tree
            self.g = parent.g + cost         # total accumulated cost 
        else:
            self.depth = 0
            self.g = cost
        self.children = []
        self.h = h_value

def getpath(y):
    """Return the path from the root to y"""
    path = [y]
    while y.parent:
        y = y.parent
        path.append(y)
    path.reverse()
    return path
    
def finish(x, node_count, prunes, frontier, explored, verbose, draw_edges):
    """called after a successful search, to print info and/or draw the solution"""
    path = getpath(x)
    if verbose >= 1:
        # Path length = number of actions = number of nodes - 1
        print('==> Path length {}, cost {}.'.format(len(path)-1,x.g), \
            'Generated {}, pruned {}, explored {}, frontier {}.'.format( \
            node_count, prunes, len(explored), len(frontier)))
    if draw_edges:  
        draw_edges([(x.parent.state[0],x.state[0]) for x in path if x.parent], 'solution')
    return [p.state for p in path]

def get_edges(nodes):
    return [(x.parent.state[0],x.state[0]) for x in nodes if x.parent]

def expand(x, next_states, h, frontier, explored, strategy, verbose, draw_edges):
    """
    expand returns six lists of nodes: new nodes, nodes pruned from new, frontier
    nodes, nodes pruned from frontier, explored nodes, and nodes pruned from explored
    """
    (key_name, key_func, template) = ('h',   lambda x: x.h,     '#{0}: h {4:.2f}, d {1}, g {3:.2f}, state {5}')
    new = [Node(s, x, cost, h(s)) for (s,cost) in next_states(x.state)]

    # make a list of dominated new nodes, then prune them
    n_prune = [m for m in new if
        [n for n in explored if m.state == n.state and key_func(m) >= key_func(n)] or
        [n for n in frontier if m.state == n.state and key_func(m) >= key_func(n)] or
        [n for n in new if m.state == n.state and key_func(m) > key_func(n)] or
        [n for n in new if m.state == n.state and key_func(m) == key_func(n) and m.id > n.id]]
    new = [m for m in new if not m in n_prune]

    # make a list of dominated frontier nodes, then prune them
    f_prune = [m for m in frontier if
        [n for n in new if m.state == n.state and key_func(m) > key_func(n)]]
    frontier = [m for m in frontier if not m in f_prune]

    # make a list of dominated explored nodes, then prune them
    e_prune = [m for m in explored if
        [n for n in new if m.state == n.state and key_func(m) > key_func(n)]]
    explored = [m for m in explored if not m in e_prune]

    frontier.extend(new)
    frontier.sort(key=key_func)

    return (new, n_prune, frontier, f_prune, explored, e_prune)


def searching(s0, next_states, goal_test, strategy, h=None, verbose=2, draw_edges=None):
    """
    Do a "graph-search-redo" search starting at state s0, looking for a path
    from s0 to a state that satisfies a user-supplied goal test. The arguments are:
    - s0 is the starting state.
    - next_states(s) is a user-supplied function to return the children of state s.
    - goal_test(s) is a user-supplied predicate to tell whether s is a goal state.
    - strategy is 'bf', 'df', 'uc', 'gbf', or 'a*'. 
    - h(s) is a user-supplied heuristic function.
    - verbose is a numeric argument; here are its possible values and their meanings:
          0 - run silently.
          1 - print some statistics at the end of the search.
          2 - print the above, plus some info at each iteration. 
          3 - print the above, plus some additional info.
          4 - print the above, and pause at each iteration.
    - draw_edges(edges,status) is a user-supplied function to draw edges in the search
      graph. It should take the following arguments:
          edges - a list of edges to draw;
          status - one of the following strings, to tell what kind of edge to draw:
          'expand', 'add', 'discard', 'frontier_prune', 'explored_prune', or 'solution'.
    """
    global node_count
    node_count = 0      # total number of generated nodes
    prunes = 0          # total number of pruned nodes
    explored = []       # all nodes that have been expanded
    (key_name, key_func, template) = ('h',   lambda x: x.h,     '#{0}: h {4:.2f}, d {1}, g {3:.2f}, state {5}')
    # Below, thes 2nd arg is None because the node has no parent.
    if h: frontier = [Node(s0,None,0,h(s0))]
    else: frontier = [Node(s0,None,0,None)]
    iteration = 0
    while frontier: 
        iteration += 1                  # keep track of how many iterations we've done
        x = frontier.pop(0)             # inefficient
        explored.append(x)
        if goal_test(x.state):
            return finish(x, node_count, prunes, frontier, explored, verbose, draw_edges)
        (new, n_prune, frontier, f_prune, explored, e_prune) = expand( \
            x, next_states, h, frontier, explored, strategy, verbose, draw_edges)
        prunes += len(n_prune) + len(f_prune) + len(e_prune)
    return False
