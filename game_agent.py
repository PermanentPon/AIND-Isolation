"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import logging
from math import sqrt
import time

logging.basicConfig(level=logging.INFO)

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def move_score(game, player):
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return own_moves - opp_moves

def next_move_score(game, player):
    #3rd level own_moves - opp_moves heuristic
    own_3rd_move = 0.
    opp_3rd_move = 0.

    blank_spaces = game.get_blank_spaces()

    directions = [(-4, -2), (-4, 0), (-4, 2),
                  (-3, -3), (-3, -1), (-3, 1), (-3, 3),
                  (-2, -4), (-2, 0), (-2, 4),
                  (-1, -3), (-1, -1), (-1, 1),  (-1, 3),
                  (0, -4), (0, -2), (0, 2), (0, 4),
                  (1, -3), (1, -1), (1, 1),  (1, 3),
                  (2, -4),  (2, 0), (2, 4),
                  (3, -3), (3, -1), (3, 1), (3, 3),
                  (4, -2), (4, 0), (4, 2)]
    my_pos = game.get_player_location(player)
    opp_pos = game.get_player_location(game.get_opponent(player))
    for dh, dv in directions:
        if (my_pos[0] + dh, my_pos[1] + dv) in blank_spaces:
            own_3rd_move += 1.0
        if (opp_pos[0] + dh, opp_pos[1] + dv) in blank_spaces:
            opp_3rd_move += 1.0
    return own_3rd_move - opp_3rd_move

def blank_spaces_score(game):
    return len(game.get_blank_spaces())

def close_to_center_score(game, player):
    own_loc = game.get_player_location(player)
    opp_loc  = game.get_player_location(game.get_opponent(player))
    me_from_center = sqrt((own_loc[0] - game.width/2)**2 + (own_loc[1] - game.height/2)**2)
    opp_from_center = sqrt((opp_loc[0] - game.width / 2) ** 2 + (opp_loc[1] - game.height / 2) ** 2)
    max_from_center = sqrt((game.width / 2) ** 2 + (game.height / 2) ** 2)
    own_center_score = max_from_center - me_from_center
    opp_center_score = max_from_center - opp_from_center
    return own_center_score - opp_center_score


def average_distance_between_blank_spaces_score(game):
    #average distance between blank_spaces heuristic
    score = 0.0
    blank_spaces = game.get_blank_spaces()
    for blank in blank_spaces:
        for blank2 in game.get_blank_spaces():
            score += sqrt(((blank[0] - blank2[0]) ** 2 + (blank[1] - blank2[1]) ** 2))
    score /= len(blank_spaces)**2
    return score


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if own_moves == 0:
        return float("-inf")

    if opp_moves == 0:
        return float("inf")

    '''
    #close my blank spaces - opponent's
    own_close_moves = 0.
    opp_close_moves = 0.

    blank_spaces = game.get_blank_spaces()

    directions = [(-2, -1), (-2, 0), (-2, 1),
                  (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
                  (0, -2), (0, -1), (0, 1), (0, 2),
                  (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
                  (2, -1), (2, 0), (2, 1)]
    my_pos = game.get_player_location(player)
    opp_pos = game.get_player_location(game.get_opponent(player))
    for dh, dv in directions:
        if (my_pos[0] + dh, my_pos[1] + dv) in blank_spaces:
            own_close_moves += 1.0
        if (opp_pos[0] + dh, opp_pos[1] + dv) in blank_spaces:
            opp_close_moves += 1.0
    '''

    #opp_from_center = sqrt((opp_loc[0] - game.width / 2) ** 2 + (abs(opp_loc[1] - game.height / 2)) ** 2)
    #between_players = sqrt(((own_loc[0] - opp_loc[0])**2 + (own_loc[1] - opp_loc[1])**2))
    #print("Between players: ", between_players)
    #return float(own_moves/(random.uniform(0, 2)*opp_moves+0.01) + between_players + 2/me_from_center) 71%
    #if (len(game.get_blank_spaces()) > 0.5*game.height*game.width):
    #if game.move_count < 20:
    #    return float(score)
    #return float((8.0 + own_moves - opp_moves)/16.0 + 0.5*(32.0 + own_3rd_move - opp_3rd_move)/64.0 + own_center_score/max_from_center)
    #move_score_value = move_score(game, player)
    close_to_center_score_value = close_to_center_score(game, player)
    blank_spaces_score_value = blank_spaces_score(game)
    average_distance_between_blank_spaces_score_value = average_distance_between_blank_spaces_score(game)
    next_move_score_value = next_move_score(game, player)
    #logging.info("custom_score called")
    return player.aggr((own_moves-opp_moves),next_move_score_value, close_to_center_score_value, average_distance_between_blank_spaces_score_value)
    #else:
    #    return float(own_moves-opp_moves)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3,score_fn=custom_score, aggr = None,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.aggr = aggr
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        best_move = (-1, -1)

        if len(legal_moves) == 0:
            return best_move

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            clone = game.copy()
            if self.iterative:
                depth = 1

                while True:
                    if self.method == 'minimax':
                        score, best_move = self.minimax(clone, depth)
                    else:
                        score, best_move = self.alphabeta(clone, depth)
                    depth += 1
                    if score == float("-inf") or score == float("inf"):
                        break

            else:
                depth = self.search_depth
                if self.method == 'minimax':
                    score, best_move = self.minimax(clone, self.search_depth)
                else:
                    score, best_move = self.alphabeta(clone, self.search_depth)

        except Timeout:
            #Handle any actions required at timeout, if necessary
            #print('Depth inside except: ', depth)
            pass

        # Return the best move from the last completed search iteration
        #print('Board: ')
        #print(game.to_string())
        #print('Depth: ', depth)
        #print('Best move: ', best_move)
        #print('Score: ', score)
        return best_move

    def minimax(self, game, depth, maximizing_player=True, count = 0):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        moves = game.get_legal_moves()
        if count == depth:
            temp_score = self.score(game, self)
            return temp_score, (-1, -1)
        if len(moves) == 0:
            return self.score(game, self), (-1, -1)
        # if count == 0:
        best_move = moves[0]
        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")
        for move in moves:
            clone = game.forecast_move(move)
            if maximizing_player:
                score, temp = self.minimax(clone, depth, False, count + 1)
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                score, temp = self.minimax(clone, depth, True, count + 1)
                if score < best_score:
                    best_score = score
                    best_move = move
        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True, count = 0):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        moves = game.get_legal_moves()
        if count == depth:
            temp_score = self.score(game, self)
            return temp_score, (-1, -1)
        if len(moves) == 0:
            return game.utility(self), (-1, -1)
        #if count == 0:
        best_move = moves[0]
        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")
        for move in moves:
            clone = game.forecast_move(move)
            if maximizing_player:
                score, temp = self.alphabeta(clone, depth, alpha, beta, False, count + 1)
                if score > best_score:
                    best_score = score
                    best_move = move
                if score >= beta: return best_score, best_move
                alpha = max(alpha, score)

            else:
                score, temp = self.alphabeta(clone, depth, alpha, beta, True, count + 1)
                if score < best_score:
                    best_score = score
                    best_move = move
                if score <= alpha: return best_score, best_move
                beta = min(beta, score)
        return best_score, best_move
