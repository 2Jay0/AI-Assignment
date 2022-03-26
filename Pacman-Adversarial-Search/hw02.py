from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

## Example Agent
class ReflexAgent(Agent):

  def Action(self, gameState):

    move_candidate = gameState.getLegalActions()

    scores = [self.reflex_agent_evaluationFunc(gameState, action) for action in move_candidate]
    bestScore = max(scores)
    Index = [index for index in range(len(scores)) if scores[index] == bestScore]
    get_index = random.choice(Index)

    return move_candidate[get_index]

  def reflex_agent_evaluationFunc(self, currentGameState, action):

    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()



def scoreEvalFunc(currentGameState):

  return currentGameState.getScore()

class AdversialSearchAgent(Agent):

  def __init__(self, getFunc ='scoreEvalFunc', depth ='2'):
    self.index = 0
    self.evaluationFunction = util.lookup(getFunc, globals())

    self.depth = int(depth)

######################################################################################
        
class MinimaxAgent(AdversialSearchAgent):
  """
    [문제 01] MiniMax의 Action을 구현하시오. (20점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  
  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    
    def Utility(depth,gameState):
        if depth>self.depth or gameState.isWin() or gameState.isLose():
            return True
        else:
            return False
    
    def minmax(depth, agentIndex, gameState):
        if Utility(depth,gameState) is True:
            return self.evaluationFunction(gameState)

        value = [] 
        actions = gameState.getLegalActions(agentIndex) 
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (agentIndex+1) >= gameState.getNumAgents():
                value += [minmax(depth+1, 0, successor)]
            else:
                value += [minmax(depth, agentIndex+1, successor)]

            
            
        if agentIndex > 0:
            result = min(value)
            
        elif agentIndex == 0:
            if depth == 1:
                max_val = max(value)
                for i in range(len(value)):
                    if (value[i] == max_val):
                        return actions[i]
            else:
                result = max(value)

        return result
        
        
    pacman = 0
    return minmax(1, pacman, gameState)

    
    raise Exception("Not implemented yet")

    ############################################################################




class AlphaBetaAgent(AdversialSearchAgent):
  """
    [문제 02] AlphaBeta의 Action을 구현하시오. (25점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
                       
  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    
    def Utility(depth,gameState):
        if depth>self.depth or gameState.isWin() or gameState.isLose():
            return True
        else:
            return False
            
    def AlphaBeta(depth, agentIndex, gameState, a, b):
        alpha = a
        beta = b

        if Utility(depth,gameState) is True:
            return self.evaluationFunction(gameState)

        store_value = []  
        actions = gameState.getLegalActions(agentIndex)  
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (agentIndex+1) >= gameState.getNumAgents():
                value = AlphaBeta(depth+1, 0, successor, alpha, beta)
            else:
                value = AlphaBeta(depth, agentIndex+1, successor, alpha, beta)

            if (agentIndex == 0 and value > beta):
                return value
            if (agentIndex > 0 and value < alpha):
                return value

            if(agentIndex == 0):
                alpha = max(alpha,value)

            if(agentIndex > 0):
                beta = min(beta,value)

            store_value += [value]
            
            

        if agentIndex > 0:
            result = min(store_value)
            
        elif agentIndex == 0:
            if(depth == 1): 
                max_val = max(store_value)
                for i in range(len(store_value)):
                    if (store_value[i] == max_val):
                        return actions[i]
            else:
                result = max(store_value)


        return result
        
    pacman = 0
    a = -math.inf
    b = math.inf
    return AlphaBeta(1, pacman, gameState, a, b)

    
    raise Exception("Not implemented yet")

    ############################################################################

class ExpectimaxAgent(AdversialSearchAgent):
  """
    [문제 03] Expectimax의 Action을 구현하시오. (25점)
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """

  def Action(self, gameState):
    ####################### Write Your Code Here ################################
    def Utility(depth,gameState):
        if depth>self.depth or gameState.isWin() or gameState.isLose():
            return True
        else:
            return False
            
    def expectimax(depth, agentIndex, gameState):
        if Utility(depth,gameState) is True:
            return self.evaluationFunction(gameState)

        value = []  
        actions = gameState.getLegalActions(agentIndex) 
        if Directions.STOP in actions:
            actions.remove(Directions.STOP) 

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if (agentIndex+1) >= gameState.getNumAgents():
                value += [expectimax(depth+1, 0, successor)]
            else:
                value += [expectimax(depth, agentIndex+1, successor)]


        if agentIndex != 0:
            result = float(sum(value)/len(value))
            
        elif agentIndex == 0:
            if depth == 1: 
                max_val = max(value)
                for i in range(len(value)):
                    if value[i] == max_val:
                        return actions[i]
            else:
                result = max(value)


        return result
        
    return expectimax(1, 0, gameState)
    
    raise Exception("Not implemented yet")
   
    ############################################################################
