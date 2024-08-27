# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            
            if agentIndex == 0:  # Pacman's turn (Maximizer)
                value = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    value = max(value, alphaBeta(state.generateSuccessor(agentIndex, action), depth, 1, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts' turn (Minimizer)
                value = float("inf")
                nextAgent = agentIndex + 1
                if nextAgent >= state.getNumAgents():  # Last ghost, reduce depth
                    nextAgent = 0
                    depth -= 1
                
                for action in state.getLegalActions(agentIndex):
                    value = min(value, alphaBeta(state.generateSuccessor(agentIndex, action), depth, nextAgent, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        # Initial call to the alphaBeta function
        bestAction = None
        alpha = float("-inf")
        beta = float("inf")
        value = float("-inf")

        for action in gameState.getLegalActions(0):
            actionValue = alphaBeta(gameState.generateSuccessor(0, action), self.depth, 1, alpha, beta)
            if actionValue > value:
                value = actionValue
                bestAction = action
            alpha = max(alpha, value)

        return bestAction


class GraphGameTreeTest:
    def _init_(self):
        self.num_agents = 2
        self.depth = 4

    def getLegalActions(self, state):
        actions = {
            'a': ['Left', 'Right'],
            'b1': ['Left', 'Right'],
            'b2': ['Left', 'Right'],
            'c1': ['Left', 'Right'],
            'c2': ['Left', 'Right'],
            'c3': ['Left', 'Right'],
            'c4': ['Left', 'Right'],
            'd1': ['Left', 'Right'],
            'd2': ['Left', 'Right'],
            'd3': ['Left', 'Right'],
            'd4': ['Left', 'Right'],
            'd5': ['Left', 'Right'],
            'd6': ['Left', 'Right'],
            'd7': ['Left', 'Right'],
            'd8': ['Left', 'Right'],
        }
        return actions.get(state, [])

    def generateSuccessor(self, state, action):
        successors = {
            'a': {'Left': 'b1', 'Right': 'b2'},
            'b1': {'Left': 'c1', 'Right': 'c2'},
            'b2': {'Left': 'c3', 'Right': 'c4'},
            'c1': {'Left': 'd1', 'Right': 'd2'},
            'c2': {'Left': 'd3', 'Right': 'd4'},
            'c3': {'Left': 'd5', 'Right': 'd6'},
            'c4': {'Left': 'd7', 'Right': 'd8'},
            'd1': {'Left': 'A', 'Right': 'B'},
            'd2': {'Left': 'C', 'Right': 'D'},
            'd3': {'Left': 'E', 'Right': 'F'},
            'd4': {'Left': 'G', 'Right': 'H'},
            'd5': {'Left': 'I', 'Right': 'J'},
            'd6': {'Left': 'K', 'Right': 'L'},
            'd7': {'Left': 'M', 'Right': 'N'},
            'd8': {'Left': 'O', 'Right': 'P'},
        }
        return successors[state][action]

    def getEvaluation(self, state):
        evaluations = {
            'A': 3.0, 'B': 13.0, 'C': 5.0, 'D': 9.0,
            'E': 10.0, 'F': 11.0, 'G': 6.0, 'H': 8.0,
            'I': 1.0, 'J': 0.0, 'K': 4.0, 'L': 7.0,
            'M': 12.0, 'N': 15.0, 'O': 2.0, 'P': 14.0,
        }
        return evaluations.get(state, 0)

    def isWin(self, state):
        return state in {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'}

    def isLose(self, state):
        return False

    def getNumAgents(self):
        return self.num_agents


# Initialize the game and the agent
test_game = GraphGameTreeTest()
alpha_beta_agent = AlphaBetaAgent(depth=4)

# Get the best action from the start state 'a'
best_action = alpha_beta_agent.getAction(test_game)

print(f"Best action from the start state 'a': {best_action}")

def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

