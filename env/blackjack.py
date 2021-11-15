"""
Blackjack Enviornment
"""
import random
from .base import FiniteMDPENV


# define globals for cards
SUITS = ('C', 'S', 'H', 'D')
RANKS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K')
VALUES = {'A':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':10, 'Q':10, 'K':10}


# define card class
class Card:
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
        
        
# define hand class
class Hand:
    def __init__(self):
        self.cards = []
        self.has_ace = False

    def __str__(self):
        card_str = ' '.join([c.__str__() for c in self.cards])
        return 'Hand contains %s' % card_str
    
    def add_card(self, card):
        self.cards.append(card)
        if card.get_rank() == 'A':
            self.has_ace = True
    
    def get_card_value(self, card):
        return VALUES[card.get_rank()]
    
    def get_nomial_value(self):
        v = 0
        for c in self.cards:
            v += self.get_card_value(c)
        return v

    def get_value(self):
        # count aces as 1, if the hand has an ace, then add 10 to hand value if it doesn't bust
        v = self.get_nomial_value()
            
        if self.has_ace and v+10 <= 21:
            v += 10
        return v
    
    def usable_ace(self):
        if not self.has_ace:
            return False

        v = self.get_nomial_value()

        if v+10 <= 21:
            return True
        return False
    
    def is_busted(self):
        return self.get_value() > 21
 
        
# define deck class 
class Deck:
    def __init__(self):
        cards = []
        for suit in SUITS:
            for rank in RANKS:
                cards.append(Card(suit, rank))
                
        self.cards = cards

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self):
        return self.cards.pop()
    
    def __str__(self):
        s = ' '.join([c.__str__() for c in self.cards])
        return 'Deck contains %s' % s


def blackjack_states():
    states = []
    for has_ace in [True, False]:
        for player_v in range(12, 22):
            for dealer_v in range(1, 11):
                states.append((has_ace, player_v, dealer_v))
    return states

class BlackJack(FiniteMDPENV):
    SMAPPER = {s:i for i, s in enumerate(blackjack_states())}
    NMAPPER = {i:s for s, i in SMAPPER.items()}

    def __init__(self, debug=False):
        self.debug = debug
        self.score = 0
        self.reset()
    
    def reset(self):
        self.in_play = False
        self.outcome = "game outcome"
        self.deck = None
        self.dealer_hand = None
        self.player_hand = None
    
    @property
    def A(self):
        return ['hit', 'stand']

    @property
    def S(self):
        return list(range(200))
    
    def start(self):
        self.deal()
        return self.get_state()
    
    def is_terminal(self):
        return not self.in_play

    def step(self, action):
        if action == 'hit':
            reward = self.hit()
        elif action == 'stand':
            reward = self.stand()
        return self.get_state(), reward
    
    def get_state(self):
        cnt = 0
        if not self.player_hand.usable_ace():
            cnt += 100
        
        pv = self.player_hand.get_value()
        fdv = self.dealer_hand.get_card_value(self.dealer_hand.cards[0])
        cnt += (pv-12) * 10 + (fdv-1+1) - 1
        return cnt
    
    def _to_state(self, state):
        return self.NMAPPER[state]
    
    def _to_idx(self, state):
        return self.SMAPPER[state]


    def deal(self):
        self.reset()
        
        self.in_play = True
        self.outcome = 'Hit or Stand?'
        if self.debug:
            print(self.outcome)
        self.deck = Deck()
        self.dealer_hand = Hand()
        self.player_hand = Hand()
        
        self.deck.shuffle()
        for _ in range(2): # 2 cards each hand
            self.player_hand.add_card(self.deck.deal_card())
            self.dealer_hand.add_card(self.deck.deal_card())
        
        # decrease unnesscary states
        # you should alwasy hit when value is below 12
        while self.player_hand.get_value() < 12:
            self.player_hand.add_card(self.deck.deal_card())
        
        if self.debug:
            print('\nNew deal,', 'current score = {}'.format(self.score))
            print('Dealer', self.dealer_hand, '=', self.dealer_hand.get_value())
            print('Player', self.player_hand, '=', self.player_hand.get_value())
               

    def hit(self):
        if not self.in_play:
            return

        self.player_hand.add_card(self.deck.deal_card())
        
        if self.debug:
            print('Hit!')
            print('Player', self.player_hand, '=', self.player_hand.get_value())
        
        if self.player_hand.is_busted():
            self.outcome = 'You have busted, new deal?'
            if self.debug:
                print(self.outcome)
            self.in_play = False
            self.score -= 1
            return -1
        return 0

        
    def stand(self):
        if not self.in_play:
            return
        
        if self.debug:
            print('Dealer is drawing cards')

        while self.dealer_hand.get_value() < 17:
            self.dealer_hand.add_card(self.deck.deal_card())
        
        reward = 0
        pv, dv = self.player_hand.get_value(), self.dealer_hand.get_value()
        if dv > 21:
            self.score += 1
            reward += 1
            self.outcome = 'Dealer is busted, you win! New deal?'
            if self.debug:
                print(self.outcome)
            
            if self.debug:
                print('Dealer =', self.dealer_hand, dv)
                print('Dealer =', dv, 'Player =', pv)
        else:
            if pv > dv:
                self.score += 1
                reward += 1
                self.outcome = 'Your value is higher, you win! New deal?'
            elif pv < dv:
                self.score -= 1
                reward -= 1
                self.outcome = 'Dealer value is higher, you loss! New deal?'
            else:
                self.outcome = 'Draw!'
            
            if self.debug:
                print('Dealer', self.dealer_hand, '=', self.dealer_hand.get_value())
                print('Player', self.player_hand, '=', self.player_hand.get_value())
                print(self.outcome)
        
        self.in_play = False
        return reward
