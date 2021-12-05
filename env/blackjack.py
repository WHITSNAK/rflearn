"""
Blackjack Enviornment
"""
import random
from itertools import product
from collections import namedtuple
from .base import FiniteMDPENV


SUITS = ('C', 'S', 'H', 'D')
RANKS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K')
VALUES = {'A':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':10, 'Q':10, 'K':10}
BJS = namedtuple('BJState', ['ace', 'pv', 'fdv'])


class Card:
    """
    Simple class that represents a card in poker

    parameter
    ---------
    suit: str, the suit of the card
        {'C', 'S', 'H', 'D'}
    rank: str, the rank of the card
        {'A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'}
    """
    def __init__(self, suit, rank):
        if (suit in SUITS) and (rank in RANKS):
            self.suit = suit
            self.rank = rank
        else:
            raise ValueError('Invalid card suit or rank')

    def __str__(self):
        return self.suit + self.rank
    
    def __repr__(self):
        return self.__str__()

    def get_suit(self):
        return self.suit

    def get_rank(self):
        return self.rank
        
        
class Hand:
    """
    Simple class that represents a player's hand in BlackJack
        - it can contains arbitary number of cards
        - it also has some nice methods to work with BlackJack
    """
    def __init__(self):
        self.cards = []
        self.has_ace = False
        self.nomial_value = 0

    def __str__(self):
        card_str = ' '.join([c.__str__() for c in self.cards])
        ace_str = 'T' if self.has_ace else 'F'
        return f'Hand[A:{ace_str} V:{self.nomial_value}] <{card_str}>'
    
    def add_card(self, card):
        """Update state with new card"""
        self.cards.append(card)

        c_rank = card.get_rank()
        self.nomial_value += VALUES[c_rank]

        if c_rank == 'A':
            self.has_ace = True
    
    def usable_ace(self):
        """Flag for showing whether the hand has an usable ace"""
        if not self.has_ace:
            return False

        if self.nomial_value + 10 <= 21:
            return True
        return False
    
    def get_value(self):
        """Get the actual value that takes in account of useable ace"""
        # count aces as 1, if the hand has an ace
        # then add 10 to hand value if it doesn't bust
        val = self.nomial_value
        if self.usable_ace():
            val += 10
        return val

    def is_busted(self):
        """Busted flag"""
        return self.get_value() > 21
 
        
class Deck:
    """Represents a deck of card"""
    def __init__(self):
        # generate a 52 cards deck list
        cards = []
        for suit in SUITS:
            for rank in RANKS:
                cards.append(Card(suit, rank))
                
        self.cards = cards

    def __str__(self):
        string = ' '.join([c.__str__() for c in self.cards])
        return f'Deck<{string}>'

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self):
        return self.cards.pop()



class BlackJack(FiniteMDPENV):
    """
    BlackJack Game Simulator with reduced state support
    - win the game with larger hand than the dealer
    - at the same time do not go busted, value > 21
    - winning reward = 1, lossing reward = -1, draw = 0

    Avaliable actions: {hit, stand}
    Avaliable states: {0, 1, ...., 199}
        - In total 200 states as follow
        - Whether player has usable Ace or not [x2], *ace*
        - Player always deal when hand is <= 11, max card is 10
          thus {12, 13, ..., 21} [x10], *pv*
        - Only the first card dealer has is visible
          thus {1, 2, 3, ..., 10} in nominal value [x10], *fdv*
    System transition: invisible
    """
    def __init__(self):
        self.score = 0

        self.__states = []
        gen = product(
            [True, False],
            range(12, 31+1),
            range(1, 10+1),
        )
        for state in gen:
            self.__states.append(BJS(*state))
        self.reset()
    
    def reset(self):
        self.in_play = False
        self.deck = None
        self.dealer_hand = None
        self.player_hand = None
        self.s0 = None
    
    @property
    def A(self):
        return ['hit', 'stand']

    @property
    def S(self):
        return self.__states
    
    def start(self):
        """Start the game fresh"""
        self._deal()
        self.s0 = self._get_nominal_state()
        return self.s0

    def step(self, action):
        """Take action/step forward"""
        reward = self.__getattribute__('_'+action)()
        self.s0 = self._get_nominal_state()
        return self.s0, reward
    
    def is_terminal(self):
        """Terminal state flag"""
        return not self.in_play

    def _get_nominal_state(self):
        """The Actual Game Explicit State Information"""
        return BJS(
            self.player_hand.usable_ace(),
            self.player_hand.get_value(),
            VALUES[self.dealer_hand.cards[0].get_rank()]
        )
    
    def _deal(self):
        """Starts the game with new hands and new deck of cards"""
        self.reset()
        self.in_play = True
        self.deck = Deck()
        self.dealer_hand = Hand()
        self.player_hand = Hand()
        
        self.deck.shuffle()
        for _ in range(2):  # 2 cards each hand
            self.player_hand.add_card(self.deck.deal_card())
            self.dealer_hand.add_card(self.deck.deal_card())
        
        # decrease unnesscary states
        # you should alwasy hit when value is below 12
        while self.player_hand.get_value() < 12:
            self.player_hand.add_card(self.deck.deal_card())
        
    def _hit(self):
        """Palyer hit get new card"""
        if not self.in_play:
            return 0
        
        # deal new card to player
        self.player_hand.add_card(self.deck.deal_card())
        
        # check if busted
        if self.player_hand.is_busted():
            self.in_play = False
            self.score -= 1
            return -1  # loss
        return 0
        
    def _stand(self):
        """Player stand, dealer draw then compare"""
        if not self.in_play:
            return 0
        
        # dealer draws cards until 17 or more
        # and she must stand if the value is >= 17
        while self.dealer_hand.get_value() < 17:
            self.dealer_hand.add_card(self.deck.deal_card())
        
        reward = 0
        pv, dv = self.player_hand.get_value(), self.dealer_hand.get_value()
        if dv > 21:  # dealer busted
            self.score += 1
            reward += 1
        else:
            if pv > dv:  # win
                self.score += 1
                reward += 1
            elif pv < dv:   # loss
                self.score -= 1
                reward -= 1
            else:  # draw
                reward += 0
        
        self.in_play = False
        return reward
