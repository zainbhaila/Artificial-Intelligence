"""
File: proj1.py
Author: Zain Bhaila <zainb@terpmail.umd.edu>
Last updated: Sept 29, 2019

Project 1 - Racetrack Heuristics

I pledge on my honor that I have not given or received
any unauthorized assistance on this project.
Zain Bhaila
"""

import racetrack as rt
import math
import sys

def h_edist(state, fline, walls):
    """Euclidean distance from state to fline, ignoring walls."""
    ((x,y),(u,v)) = state
    ((x1,y1),(x2,y2)) = fline
    
    # find the smallest and largest coordinates
    xmin = min(x1,x2); xmax = max(x1,x2)
    ymin = min(y1,y2); ymax = max(y1,y2)

    return min([math.sqrt((xx-x)**2 + (yy-y)**2)
        for xx in range(xmin,xmax+1) for yy in range(ymin,ymax+1)])


def h_esdist(state, fline, walls):
    """
    h_edist modified to include an estimate of how long it will take to stop.
    """
    ((x,y),(u,v)) = state
    ((x1,y1),(x2,y2)) = fline
    if ((x==x1 and y==y1)or (x==x2 and y==y2)) and u == 0 and v == 0:
        return 0
    m = math.sqrt(u**2 + v**2)
    stop_dist = m*(m-1)/2.0 + 1
    return max(h_edist(state, fline, walls)+stop_dist/10.0,stop_dist)*3


# Global variables for h_walldist

infinity = float('inf')     # alternatively, we could import math.inf

g_fline = False
g_walls = False
grid = []


def h_proj1(state, fline, walls):
    """
    The first time this function is called, for each gridpoint that's not inside a wall
    it will cache a rough estimate of the length of the shortest path to the finish line.
    The computation is done by a breadth-first search going backwards from the finish 
    line, one gridpoint at a time.
    
    On all subsequent calls, this function will retrieve the cached value and add an
    estimate of how long it will take to stop. 
    """
    global g_fline, g_walls
    ((x,y),(u,v)) = state
    
    # if there are no walls between state and finish, use h_esdist
    if edistw_to_finish((x,y), fline, walls) != infinity:
        return h_esdist(state, fline, walls)
    # update cache if necessary
    if fline != g_fline or walls != g_walls or grid == []:
        edist_grid(fline, walls)
    hval = float(grid[x][y])
    
    # add a small penalty to favor short stopping distances
    au = abs(u); av = abs(v); 
    sdu = au*(au-1)/2.0
    sdv = av*(av-1)/2.0
    sd = max(sdu,sdv)
    penalty = sd/10.0

    # compute location after fastest stop, and add a penalty if it goes through a wall
    if u < 0: sdu = -sdu
    if v < 0: sdv = -sdv
    sx = x + sdu
    sy = y + sdv
    if rt.crash([(x,y),(sx,sy)],walls):
        penalty += math.sqrt(au**2 + av**2)
    hval = max(hval+penalty,sd)
    return hval*3

def edist_grid(fline,walls):
    """
    This functions creates a grid to cache values in the graph. Walls and 
    unreachable nodes are stored as infinity. The function uses a BFS
    to traverse the grid and find distance values to each node from the
    finish line.
    """  
    global grid, g_fline, g_walls, xmax, ymax
    xmax = max([max(x,x1) for ((x,y),(x1,y1)) in walls])
    ymax = max([max(y,y1) for ((x,y),(x1,y1)) in walls])
    visited = []
    # initialize grid
    grid = [[infinity for y in range(ymax+1)] for x in range(xmax+1)]
    # get all reachable points from finish and mark as visited
    for x in range(xmax+1):
        for y in range(ymax+1):
            grid[x][y] = edistw_to_finish((x,y), fline, walls)
            if grid[x][y] != infinity:
                visited.append((x,y))
    queue = visited[:]
    while queue:
        (x,y) = queue.pop(0)
        # for each neighbor of the first node in queue
        for y1 in range(max(0,y-1),min(ymax+1,y+2)):
            for x1 in range(max(0,x-1),min(xmax+1,x+2)):
                # if a neighbor is not a wall and not visited
                # add it to queue and mark as visited
                # then update grid with new value for (x, y)
                if not rt.crash(((x,y),(x1,y1)),walls):
                    if (x1, y1) not in visited:
                        queue.append((x1,y1))
                        visited.append((x1, y1))
                    if x == x1 or y == y1:
                        d = grid[x1][y1] + 1
                    else:
                        # In principle, it seems like a taxicab metric should be just as
                        # good, but Euclidean seems to work a little better in my tests.
                        d = grid[x1][y1] + 1.4142135623730951
                    if d < grid[x][y]:
                        grid[x][y] = d
                        flag = True
    g_fline = fline
    g_walls = walls
    return grid


def edistw_to_finish(point, fline, walls):
    """
    straight-line distance from (x,y) to the finish line ((x1,y1),(x2,y2)).
    Return infinity if there's no way to do it without intersecting a wall
    """  
    (x,y) = point
    ds = infinity
    ((x1,y1),(x2,y2)) = fline
    # make a list of distances to each reachable point in fline
    if x1 == x2:               # fline is vertical, so iterate over y
        for y3 in range(min(y1,y2),max(y1,y2)+1):
            if not rt.crash(((x,y),(x1,y3)), walls):
                ds = min(ds, math.sqrt((x1-x)**2 + (y3-y)**2))
    else:                      # fline is horizontal, so iterate over x
        for x3 in range(min(x1,x2),max(x1,x2)+1):
            if not rt.crash(((x,y),(x3,y1)), walls):
                ds = min(ds, math.sqrt((x3-x)**2 + (y1-y)**2))
                
    return ds
