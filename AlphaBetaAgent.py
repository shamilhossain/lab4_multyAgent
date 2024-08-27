from pacman import Directions
from game import Agent
import random

class AlphaBetaAgent(Agent):
    def getAction(self, gameState):
        """
        Returns the best action using alpha-beta pruning algorithm.
        """
        def alphaBeta(state, depth, alpha, beta, agentIndex):
            """
            Recursive alpha-beta pruning function.
            """
            # If the game is over or the depth limit is reached
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            # Check if it's the last agent (Pacman)
            if agentIndex == state.getNumAgents() - 1:
                # Pacman's turn: Maximize
                bestValue = float('-inf')
                bestAction = None
                legalActions = state.getLegalActions(agentIndex)
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = alphaBeta(successor, depth - 1, alpha, beta, 0)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                    alpha = max(alpha, bestValue)
                    if beta <= alpha:
                        break  # Beta cut-off
                return bestValue, bestAction

            else:
                # Ghosts' turn: Minimize
                bestValue = float('inf')
                bestAction = None
                legalActions = state.getLegalActions(agentIndex)
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
                    value, _ = alphaBeta(successor, depth, alpha, beta, nextAgentIndex)
                    if value < bestValue:
                        bestValue = value
                        bestAction = action
                    beta = min(beta, bestValue)
                    if beta <= alpha:
                        break  # Alpha cut-off
                return bestValue, bestAction

        _, action = alphaBeta(gameState, self.depth, float('-inf'), float('inf'), 0)
        return action

    def evaluationFunction(self, gameState):
        """
        Evaluation function for the state.
        You might want to use a more complex evaluation function here.
        """
        return gameState.getScore()
