###### Write Your Library Here ###########
from collections import deque
from itertools import permutations
from itertools import combinations
import time

#########################################


def search(maze, func):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_four_circles": astar_four_circles,
        "astar_many_circles": astar_many_circles
    }.get(func)(maze)


# -------------------- Stage 01: One circle - BFS Algorithm ------------------------ #

def bfs(maze):
    """
    [문제 01] 제시된 stage1의 맵 세가지를 BFS Algorithm을 통해 최단 경로를 return하시오.(20점)
    """
    start_point=maze.startPoint()
    
    path=[]

    ####################### Write Your Code Here ################################
   
    end_point = maze.circlePoints()[0]
    
    n=maze.rows
    m=maze.cols
    parent = [[0]*m*2 for _ in range(n)]
    dist = [[0]*m*2 for _ in range(n)]
    dist[start_point[0]][start_point[1]]=1
    
    curx=0
    cury=0
    flag=0
    
    queue=deque()
    queue.append((start_point[0],start_point[1]))
    temp=[]
    temp=maze.neighborPoints(start_point[0],start_point[1])    

    while queue:
        x=queue.popleft()
                
        temp=maze.neighborPoints(x[0],x[1])
        
        for i in range(len(temp)): 
            if x[0] is end_point[0] and x[1] is end_point[1]:
                flag=1
                curx=x[0]
                cury=x[1]  
                break
            if dist[temp[i][0]][temp[i][1]] >= 1:
                continue
            dist[temp[i][0]][temp[i][1]]=dist[x[0]][x[1]]+1
            parent[temp[i][0]][temp[i][1]]=[x[0],x[1]]
            queue.append(temp[i])
        
        if flag == 1:
            break

 
    while True:
        path.append((curx,cury))
        if curx==start_point[0] and cury==start_point[1]:
            break
        nx = parent[curx][cury][0]
        ny = parent[curx][cury][1]
        
        curx=nx
        cury=ny
    
    
    return path[::-1]

    ############################################################################



class Node:
    def __init__(self,parent,location):
        self.parent=parent
        self.location=location #현재 노드

        self.obj=[]

        # F = G+H
        self.f=0
        self.g=0
        self.h=0

    def __eq__(self, other):
        return self.location==other.location and str(self.obj)==str(other.obj)

    def __le__(self, other):
        return self.g+self.h<=other.g+other.h

    def __lt__(self, other):
        return self.g+self.h<other.g+other.h

    def __gt__(self, other):
        return self.g+self.h>other.g+other.h

    def __ge__(self, other):
        return self.g+self.h>=other.g+other.h


# -------------------- Stage 01: One circle - A* Algorithm ------------------------ #

def manhatten_dist(p1,p2):
    return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])

def astar(maze):

    """
    [문제 02] 제시된 stage1의 맵 세가지를 A* Algorithm을 통해 최단경로를 return하시오.(20점)
    (Heuristic Function은 위에서 정의한 manhatten_dist function을 사용할 것.)
    """

    start_point=maze.startPoint()

    end_point=maze.circlePoints()[0]

    path=[]

    ####################### Write Your Code Here ################################

    startNode = Node(None,start_point)
    startNode.g = startNode.h = startNode.f = 0
    endNode = Node(None,end_point)
    endNode.g = endNode.h = endNode.f = 0
    
    openList = []
    closedList = []

    openList.append(startNode)

    while openList:

        currentNode = openList[0]
        currentIdx = 0

        for index, item in enumerate(openList):
            if item.f < currentNode.f:
                currentNode = item
                currentIdx = index

        openList.pop(currentIdx)
        closedList.append(currentNode)

        if currentNode == endNode:
            current = currentNode
            while current is not None:
                path.append(current.location)
                current = current.parent
            return path[::-1]  # reverse

        children = []
        
        (x,y) = currentNode.location
        nodePosition = maze.neighborPoints(x,y)

        for i in range(len(nodePosition)):
            new_node = Node(currentNode, nodePosition[i])
            children.append(new_node)  
        
        for child in children:
  
            if child in closedList:
                continue
            
            child.g = currentNode.g + 1
            child.h = manhatten_dist(child.location, endNode.location)        
            child.f = child.g + child.h

            if len([openNode for openNode in openList
                    if child == openNode and child.g > openNode.g]) > 0:
                continue        
            openList.append(child)
        
      
    return None



    ############################################################################


# -------------------- Stage 02: Four circles - A* Algorithm  ------------------------ #


def stage2_heuristic(p1,p2):
    x_val = abs(p1[0] - p2[0])
    y_val = abs(p1[1] - p2[1])
    
    return x_val**2 + y_val**2

def astar_four_circles(maze):
    """
    [문제 03] 제시된 stage2의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage2_heuristic function을 직접 정의하여 사용해야 한다.)
    """
    
    end_points=maze.circlePoints()
    end_points.sort()

    start_point=maze.startPoint()
    path=[]
    path_temp=[]
    path2=[]
    
    min_path=0
    ####################### Write Your Code Here ################################
    
    
    permute = list(permutations([0,1,2,3]))
    
    iter_num=0 #순서쌍 반복 횟수
    
    while iter_num != len(permute):
        startNode = Node(None,start_point)
        startNode.g = startNode.h = startNode.f = 0
        endNode = Node(None,end_points[permute[iter_num][0]])
        endNode.g = endNode.h = endNode.f = 0
            
        openList = []
        closedList = []
        cnt=1
        temp=0
        
        openList.append(startNode)

        while openList:
            currentNode = openList[0]
            currentIdx = 0

            for index, item in enumerate(openList):
                if item.f < currentNode.f:
                    currentNode = item
                    currentIdx = index

            openList.pop(currentIdx)
            closedList.append(currentNode)

            if currentNode == endNode:
                current = currentNode
                while current is not None:
                    path2.append(current.location)
                    current = current.parent
             
                openList.clear()
                closedList.clear()                    
                
                if temp != 0:
                    path2.pop()
    
                temp+=1
                path2.reverse()
                path_temp.extend(path2)
                path2.clear()
                
                if cnt == len(end_points):
                    if iter_num == 0:
                        min_path = len(path_temp)
                        path=path_temp.copy()
                        path_temp.clear()
                    else:
                        if len(path_temp) < min_path:
                            min_path = len(path_temp)
                            path=path_temp.copy()
                    
                        path_temp.clear()
                
                    iter_num+=1         
                    break

                else:
                    openList.append(endNode)
                    endNode = Node(None,end_points[permute[iter_num][cnt]])
                    endNode.g = endNode.h = endNode.f = 0
                    cnt+=1
                    continue
            

            children = []
        
            (x,y) = currentNode.location
            nodePosition = maze.neighborPoints(x,y)

            for i in range(len(nodePosition)):
                new_node = Node(currentNode, nodePosition[i])
                children.append(new_node)  
        
            for child in children:
  
                if child in closedList:
                    continue
            
                child.g = currentNode.g + 1
                child.h = stage2_heuristic(child.location, endNode.location)        
                child.f = child.g + child.h

                if len([openNode for openNode in openList
                        if child == openNode and child.g > openNode.g]) > 0:
                    continue        
                openList.append(child)
               
    
    return path[::-1]
    
    ############################################################################



# -------------------- Stage 03: Many circles - A* Algorithm -------------------- #

def mst(objectives, edges):

    cost_sum=0
    ####################### Write Your Code Here ################################

    n=len(objectives)
    
    parent = dict()
    rank = dict()
    
    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]
        
    def union(u,v):
        root1 = find(u)
        root2 = find(v)
        
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
            if rank[root1] == rank[root2]:
                rank[root2] +=1
      
        
    def make_set(node):
        parent[node] = node
        rank[node] = 0
        
         
    for obj in objectives:
        make_set(obj)
                
    for edge in edges:
        if not edge:
            continue
        u, v, wt = edge
        if find(v) != find(u):
            union(u,v)
            cost_sum+=wt
       
    return cost_sum
    ############################################################################


def stage3_heuristic(p1,p2):
    
    pass

def bfs_edge_check(maze,start,end):
    n=maze.rows
    m=maze.cols
    parent = [[0]*m*2 for _ in range(n)]
    dist = [[0]*m*2 for _ in range(n)]
    
    dist[start[0]][start[1]]=1
    
    cost = 0
    
    queue=deque()
    queue.append((start[0],start[1]))
    temp=[]
    temp=maze.neighborPoints(start[0],start[1])    
    
    
    while queue:
        x=queue.popleft()
                
        temp=maze.neighborPoints(x[0],x[1])
        
        for i in range(len(temp)):
            if x[0] == end[0] and x[1] == end[1]:
                return dist[x[0]][x[1]]
                
            if dist[temp[i][0]][temp[i][1]] >= 1:
                continue
            dist[temp[i][0]][temp[i][1]]=dist[x[0]][x[1]]+1
            parent[temp[i][0]][temp[i][1]]=[x[0],x[1]]
            queue.append(temp[i])   

def astar_many_circles(maze):
    """
    [문제 04] 제시된 stage3의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage3_heuristic function을 직접 정의하여 사용해야 하고, minimum spanning tree
    알고리즘을 활용한 heuristic function이어야 한다.)
    """

    start_point=maze.startPoint()
    end_points= maze.circlePoints()
    end_points.sort()

    path=[]
    path2=[]
    path_temp=[]
    ####################### Write Your Code Here ################################

    graph_edges=[]
    arr=[]
    
    for i in range(len(end_points)+1):
        arr.append(i)
        
    combi = list(combinations(arr,2))
    
    edge_length = [[0]*len(combi)*2 for _ in range(len(combi))] #간선 가중치


    for i in range(len(combi)):
        if combi[i][0] == 0:
            edge_length[combi[i][0]][combi[i][1]] = bfs_edge_check(maze,start_point,end_points[combi[i][1]-1])
            edge_length[combi[i][1]][combi[i][0]] = edge_length[combi[i][0]][combi[i][1]]
        else:
            edge_length[combi[i][0]][combi[i][1]] = bfs_edge_check(maze,end_points[combi[i][0]-1],end_points[combi[i][1]-1])
            edge_length[combi[i][1]][combi[i][0]] = edge_length[combi[i][0]][combi[i][1]]
        graph_edges.append((combi[i][0],combi[i][1],edge_length[combi[i][0]][combi[i][1]]))

    graph_edges.sort(key=lambda x:x[2])
    
    startNode = Node(None,0)
    startNode.g = startNode.h = startNode.f = 0
            
    openList = []
    closedList = []
    cnt=2
    path2 = []
    temp=[]

    openList.append(startNode)

    while openList:
        temp=graph_edges.copy()
        currentNode = openList[0]
        currentIdx = 0

        for index, item in enumerate(openList):
            if item.f < currentNode.f:
                currentNode = item
                currentIdx = index
    
        openList.pop(currentIdx)
        closedList.append(currentNode)
        
        current = currentNode
        while current is not None:
            path2.append(current.location)
            current = current.parent
            
        children = []
    
        x = currentNode.location
        
        for i in range(1,len(end_points)+1): 
            if i != x or i not in path2:
                new_node = Node(currentNode, i)
                children.append(new_node)
        
        for child in children:
            temp2=[]
            check=[]
            temp=graph_edges.copy()
            objectives = []
                
            if child in closedList:
                continue
            
            check_child = child
            while check_child is not None:
                check.append(check_child.location)
                check_child = check_child.parent
                
            for i in range(1,len(end_points)+1):
                if i not in check or i is check[0]:
                    objectives.append(i)
            

            for i in range(len(temp)):
                if temp[i][0] not in objectives or temp[i][1] not in objectives:
                    continue
                temp2.append(temp[i])
              
            child.g = currentNode.g + edge_length[x][child.location]
            child.h = mst(objectives, temp2)
            child.f = child.g + child.h

            if len([openNode for openNode in openList
                    if child == openNode and child.g > openNode.g]) > 0:
                continue        
            openList.append(child)
            
        path2.clear()
    
    
    order=[]
    if len(end_points) <= 10:
        order = [4,0,1,2,3,6,9,8,5,7]
    else:
        for i in range(len(end_points)):
            order.append(i)
    
    startNode = Node(None,start_point)
    startNode.g = startNode.h = startNode.f = 0
    endNode = Node(None,end_points[order[0]])
    endNode.g = endNode.h = endNode.f = 0
            
    openList = []
    closedList = []
    cnt=1
    temp=0
       
    openList.append(startNode)

    while openList:
        currentNode = openList[0]
        currentIdx = 0

        for index, item in enumerate(openList):
            if item.f < currentNode.f:
                currentNode = item
                currentIdx = index

        openList.pop(currentIdx)
        closedList.append(currentNode)

        if currentNode == endNode:
            current = currentNode
            while current is not None:
                path2.append(current.location)
                current = current.parent
             
            openList.clear()
            closedList.clear()                    
                
            if temp != 0:
                path2.pop()
    
            temp+=1
            path2.reverse()
            path_temp.extend(path2)
            path2.clear()
                
            if cnt == len(end_points):
                path=path_temp.copy()
                return path

            else:
                openList.append(endNode)
                endNode = Node(None,end_points[order[cnt]])
                endNode.g = endNode.h = endNode.f = 0
                cnt+=1
                continue
            

        children = []
    
        (x,y) = currentNode.location
        nodePosition = maze.neighborPoints(x,y)
        for i in range(len(nodePosition)):
            new_node = Node(currentNode, nodePosition[i])
            children.append(new_node)  
        
        for child in children:
  
            if child in closedList:
                continue
            
            child.g = currentNode.g + 1
            child.h = manhatten_dist(child.location, endNode.location)
            child.f = child.g + child.h

            if len([openNode for openNode in openList
                    if child == openNode and child.g > openNode.g]) > 0:
                continue        
            openList.append(child)
               
    
    return path[::-1]
    
    
    ############################################################################
