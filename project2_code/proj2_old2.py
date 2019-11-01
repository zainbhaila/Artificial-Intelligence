"""
File: proj2_example.py -- Dana Nau, Oct 16, 2019
A simple-minded proj2 program. It ignores what the opponent might do, and chooses its
next move by running a modified version of racetrack.py. It can sometimes win if it's 
lucky, but in most cases it will eventually crash.

To run this program in the supervisor, rename the file to proj2.py .
"""

import math
import sys
import json
import random

# Global variable for h_walldist
infinity = float('inf')     # same as math.inf

def main(state, finish, walls):
    # Retrieve the grid data that the "initialize" function stored in data.txt
    data_file = open('data.txt', 'r')
    grid_using = False
    try:
        grid = json.load(data_file)
        grid_using = True
    except:
        print("Failed to load json")
    data_file.close()
    choices_file = open('choices.txt', 'w')
    
    if grid_using:
        h = lambda state,fline,walls: h_walldist(state,fline,walls,grid)
    else:
        h = lambda state,fline,walls: h_esdist(state,fline,walls)
    
    ((x,y),(u,v)) = state
    limit = 0
    if abs(u) <= 2 and abs(v) <= 2:
        print((0,0),file=choices_file,flush=True)
    if edist_to_line((x,y),finish) <= 1 and abs(u) <= 2 and abs(v) <= 2:
        print((0,0),file=choices_file,flush=True)
    else:
        best_val = infinity
        while True:
            (states, val) = search(state, state, finish, walls, limit, h)
            if val < best_val:
                print(states[1])
                print(states[1],file=choices_file,flush=True)
            limit = limit+1
       
def search(state, errstate, finish, walls, limit, h):
    if limit == 0 or h(state, finish, walls) == 0:
        possible = next_states(state, finish, walls)
        best = (state, infinity)
        
        for (u,i) in possible:
            v = h(i, finish, walls)
            vu = h(u, finish, walls)
            if v < best[1]:
                best = (i,v)
        
        return best
    else:
        possible = next_states(errstate, finish, walls)
        
        best = (state, infinity)
        
        for (u,i) in possible:
            (s, v) = search(u, i, finish, walls, limit - 1, h)
            if v < best[1]:
                best = (i,v)
            if v == best[1] and random.randint(1,2) == 2:
                best = (i,v)
                
        return best

def edist_to_line(point, edge):
    """
    Euclidean distance from (x,y) to the line ((x1,y1),(x2,y2)).
    """
    (x,y) = point
    ((x1,y1),(x2,y2)) = edge
    if x1 == x2:
        ds = [math.sqrt((x1-x)**2 + (y3-y)**2) \
            for y3 in range(min(y1,y2),max(y1,y2)+1)]
    else:
        ds = [math.sqrt((x3-x)**2 + (y1-y)**2) \
            for x3 in range(min(x1,x2),max(x1,x2)+1)]
    return min(ds)
    
def initialize(state,fline,walls):    
    """
    Call edist_grid to initialize the grid for h_walldist, then write the data, in
    json format, to the file "data.txt" so it won't be lost when the process exits
    """
    data_file = open('data.txt', 'w')
    data_file.close()
    edist_grid(fline,walls)
    data_file = open('data.txt', 'w')
    json.dump(grid,data_file)
    data_file.close()


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
    if ((x==x1 and y==y1) or (x==x2 and y==y2)) and u == 0 and v == 0:
        return 0
    m = math.sqrt(u**2 + v**2)
    stop_dist = m*(m-1)/2.0 + 1
    return max(h_edist(state, fline, walls)+stop_dist/10.0,stop_dist)

def h_walldist(state, fline, walls, grid):
    """
    The new version of h_walldist no longer calls edist_grid, but instead takes
    the grid as a fourth argument. It retrieves the current position's grid value,
    and adds an estimate of how long it will take to stop. 
    """
    ((x,y),(u,v)) = state
    
    # if there are no walls between state and finish, use h_esdist
    #if edistw_to_finish((x,y), fline, walls) != infinity and grid[x][y] != infinity:
    #    return h_esdist(state, fline, walls)
        
    hval = float(grid[x][y])
    
    # add a small penalty to favor short stopping distances
    (ex, ey) = opponent1((x,y),(u,v),fline,walls)
    au = abs(u)+abs(ex); av = abs(v)+abs(ey); 
    sdu = au*(au-1)/2.0
    sdv = av*(av-1)/2.0
    sd = max(sdu,sdv)
    penalty = 0

    # compute location after fastest stop, and add a penalty if it goes through a wall
    if u < 0: sdu = -sdu
    if v < 0: sdv = -sdv
    sx = x + sdu
    sy = y + sdv
    if crash([(x,y),(sx,sy)],walls) or crash([(x,y),(sx+ex,sy+ey)],walls):
        penalty += infinity
    hval = max(hval+penalty,sd)
    return hval
    
def edist_grid(fline,walls):
    """
    This functions creates a grid to cache values in the graph. Walls andcrash
    unreachable nodes are stored as infinity. The function uses a BFS
    to traverse the grid and find distance values to each node from the
    finish line.
    """  
    global grid
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
    inifinity_states = []
    while queue:
        (x,y) = queue.pop(0)
        # for each neighbor of the first node in queue
        infinity_flag = False
        for y1 in range(max(0,y-1),min(ymax+1,y+2)):
            for x1 in range(max(0,x-1),min(xmax+1,x+2)):
                # if a neighbor is not a wall and not visited
                # add it to queue and mark as visited
                # then update grid with new value for (x, y)
                if not crash(((x,y),(x1,y1)),walls):
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
                else:
                    infinity_flag = True
        if infinity_flag:
            inifinity_states.append((x,y))
    
    # set all wall neighbors to infinity
    #for (x,y) in inifinity_states:
    #   grid[x][y] = infinity
    return grid

def edistw_to_finish(point, fline, walls):
    """
    straight-line distance from (x,y) to the finish line ((x1,y1),(x2,y2)).
    Return infinity if there's no way to do it without intersecting a wall
    """
#   if min(x1,x2) <= x <= max(x1,x2) and  min(y1,y2) <= y <= max(y1,y2):
#       return 0
    (x,y) = point
    ((x1,y1),(x2,y2)) = fline
    # make a list of distances to each reachable point in fline
    if x1 == x2:           # fline is vertical, so iterate over y
        ds = [math.sqrt((x1-x)**2 + (y3-y)**2) \
            for y3 in range(min(y1,y2),max(y1,y2)+1) \
            if not crash(((x,y),(x1,y3)), walls)]
    else:                  # fline is horizontal, so iterate over x
        ds = [math.sqrt((x3-x)**2 + (y1-y)**2) \
            for x3 in range(min(x1,x2),max(x1,x2)+1) \
            if not crash(((x,y),(x3,y1)), walls)]
    ds.append(infinity)    # for the case where ds is empty
    return min(ds)
        
def edistf_to_line(point, edge, f_line):
    """
    straight-line distance from (x,y) to the line ((x1,y1),(x2,y2)).
    Return infinity if there's no way to do it without intersecting f_line
    """
#    if min(x1,x2) <= x <= max(x1,x2) and  min(y1,y2) <= y <= max(y1,y2):
#        return 0
    (x,y) = point
    ((x1,y1),(x2,y2)) = edge
    if x1 == x2:
        ds = [math.sqrt((x1-x)**2 + (yy-y)**2) \
            for yy in range(min(y1,y2),max(y1,y2)+1) \
            if not intersect([(x,y),(x1,yy)], f_line)]
    else:
        ds = [math.sqrt((xx-x)**2 + (y1-y)**2) \
            for xx in range(min(x1,x2),max(x1,x2)+1) \
            if not intersect([(x,y),(xx,y1)], f_line)]
    ds.append(infinity)
    return min(ds)

def crash(move,walls):
    """Test whether move intersects a wall in walls"""
    for wall in walls:
        if intersect(move,wall): return True
    return False
    
def opponent1(p, z, finish, walls):
    """
    p is the current location; z is the new velocity chosen by the user.
    finish and walls are the finish line and walls.
    If possible, find an error (q,r) that will cause a crash. Otherwise, choose
    an error (q,r) that will put the user as close to a wall as possible.
    """
    if z == (0,0):
        # velocity is 0, so there isn't any error
        return (0,0)
    # calculate the position we'd go to if there were no error
    x = p[0] + z[0]
    y = p[1] + z[1]
    ebest = None                # best error found so far
    dbest = infinity            # min. distance to wall if we use error ebest
    for q in range(-1,2):           # i.e., q = -1, 0, 1    
        for r in range(-1,2):       # i.e., r = -1, 0, 1
            xe = x + q
            ye = y + r
            if crash((p, (xe,ye)), walls):
                return (q,r)
            for w in walls:
                # how close will wall w be if the error is (xe,ye)?
                d = edistf_to_line((xe,ye), w, finish)
                if d < dbest:
                    dbest = d
                    ebest = (q,r)
    return ebest
    
def intersect(e1,e2):
    """Test whether edges e1 and e2 intersect"""       
    
    # First, grab all the coordinates
    ((x1a,y1a), (x1b,y1b)) = e1
    ((x2a,y2a), (x2b,y2b)) = e2
    dx1 = x1a-x1b
    dy1 = y1a-y1b
    dx2 = x2a-x2b
    dy2 = y2a-y2b
    
    if (dx1 == 0) and (dx2 == 0):        # both lines vertical
        if x1a != x2a: return False
        else:     # the lines are collinear
            return collinear_point_in_edge((x1a,y1a),e2) \
                or collinear_point_in_edge((x1b,y1b),e2) \
                or collinear_point_in_edge((x2a,y2a),e1) \
                or collinear_point_in_edge((x2b,y2b),e1)
    if (dx2 == 0):        # e2 is vertical (so m2 = infty), but e1 isn't vertical
        x = x2a
        # compute y = m1 * x + b1, but minimize roundoff error
        y = (x2a-x1a)*dy1/float(dx1) + y1a
        return collinear_point_in_edge((x,y),e1) and collinear_point_in_edge((x,y),e2) 
    elif (dx1 == 0):        # e1 is vertical (so m1 = infty), but e2 isn't vertical
        x = x1a
        # compute y = m2 * x + b2, but minimize roundoff error
        y = (x1a-x2a)*dy2/float(dx2) + y2a
        return collinear_point_in_edge((x,y),e1) and collinear_point_in_edge((x,y),e2) 
    else:        # neither line is vertical
        # check m1 = m2, without roundoff error:
        if dy1*dx2 == dx1*dy2:        # same slope, so either parallel or collinear
            # check b1 != b2, without roundoff error:
            if dx2*dx1*(y2a-y1a) != dy2*dx1*x2a - dy1*dx2*x1a:    # not collinear
                return False
            # collinear
            return collinear_point_in_edge((x1a,y1a),e2) \
                or collinear_point_in_edge((x1b,y1b),e2) \
                or collinear_point_in_edge((x2a,y2a),e1) \
                or collinear_point_in_edge((x2b,y2b),e1)
        # compute x = (b2-b1)/(m1-m2) but minimize roundoff error:
        x = (dx2*dx1*(y2a-y1a) - dy2*dx1*x2a + dy1*dx2*x1a)/float(dx2*dy1 - dy2*dx1)
        # compute y = m1*x + b1 but minimize roundoff error
        y = (dy2*dy1*(x2a-x1a) - dx2*dy1*y2a + dx1*dy2*y1a)/float(dy2*dx1 - dx2*dy1)
    return collinear_point_in_edge((x,y),e1) and collinear_point_in_edge((x,y),e2) 


def collinear_point_in_edge(point, edge):
    """
    Helper function for intersect, to test whether a point is in an edge,
    assuming the point and edge are already known to be collinear.
    """
    (x,y) = point
    ((xa,ya),(xb,yb)) = edge
    # point is in edge if (i) x is between xa and xb, inclusive, and (ii) y is between
    # ya and yb, inclusive. The test of y is redundant unless the edge is vertical.
    if ((xa <= x <= xb) or (xb <= x <= xa)) and ((ya <= y <= yb) or (yb <= y <= ya)):
       return True
    return False
    
def next_states(state, f_line, walls):
    """Return a list of states we can go to from state"""
    states = []
    (loc,(vx,vy)) = state
    for dx in [0,-1,1,-2,2]:
        if abs(dx+vx) <= 4:
            for dy in [0,-1,1,-2,2]:
                if abs(dy+vy) <= 4:
                    (wx,wy) = (vx+dx,vy+dy)
                    newloc = (loc[0]+wx,loc[1]+wy)
                    err = opponent1(loc, (wx, wy), f_line, walls)
                    errloc = (newloc[0] + err[0], newloc[1] + err[1])
                    if (not crash((loc,errloc),walls) and
                    not crash((loc,newloc),walls) and
                    not ((wx, wy) != 0 and (errloc == loc))):
                        states.append(((newloc, (wx,wy)),(errloc,(wx,wy))))
    return states
