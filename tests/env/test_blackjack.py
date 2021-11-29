import pytest
import random
from unittest.mock import patch
from rflearn.env import blackjack as bj


@pytest.mark.parametrize(
    'suit, rank, repr, error_flag',
    [
        ('A', 'Q', None, True),
        ('S', 'Z', None, True),
        ('C', 'A', 'CA', False),
        ('H', '4', 'H4', False),
    ]
)
def test_card(suit, rank, repr, error_flag):
    if error_flag:
        with pytest.raises(ValueError):
            card = bj.Card(suit, rank)
    else:
        card = bj.Card(suit, rank)
        assert str(card) == repr
        assert card.get_suit() == suit
        assert card.get_rank() == rank



def test_hand():
    hand = bj.Hand()
    assert '0' in str(hand)
    assert 'F' in str(hand)

    hand.add_card(bj.Card('C', '2'))
    assert hand.nomial_value == hand.get_value()
    assert hand.nomial_value == 2
    assert hand.usable_ace() is False
    assert hand.is_busted() is False

    hand.add_card(bj.Card('H','A'))
    assert hand.nomial_value != hand.get_value()
    assert hand.nomial_value == 3
    assert hand.get_value() == 13
    assert hand.usable_ace() is True
    assert hand.is_busted() is False

    hand.add_card(bj.Card('D', 'K'))
    assert hand.nomial_value == hand.get_value()
    assert hand.get_value() == 13
    assert hand.usable_ace() is False
    assert hand.is_busted() is False

    hand.add_card(bj.Card('S', '8'))
    assert hand.get_value() == 21
    assert hand.is_busted() is False

    hand.add_card(bj.Card('C', 'A'))
    assert hand.is_busted() is True


def test_deck():
    deck = bj.Deck()
    assert len(deck.cards) == 52

    with patch('random.shuffle'):
        deck.shuffle()
        random.shuffle.assert_called_once_with(deck.cards)
    
    cards_copy = deck.cards.copy()
    card = deck.deal_card()
    assert card in cards_copy
    assert card not in deck.cards


def test_blackjack():
    env = bj.BlackJack()

    assert len(env.A) == 2
    states = env.S
    assert len(states) == 200
    assert states == list(range(200))
    assert env.is_terminal() is True
    assert env.score == 0

def test_blackjack_deal():
    env = bj.BlackJack()

    # ensure correct amount of cards
    # and always shuffle the deck
    for _ in range(100):
        with patch.object(bj.Deck, 'shuffle'):
            env._deal()
            env.deck.shuffle.assert_called_once()
            assert env.player_hand.get_value() >= 12
            assert len(env.dealer_hand.cards) == 2

def test_blackjack_hit():
    env = bj.BlackJack()
    env.deck = bj.Deck()
    assert env._hit() == 0

    # will never go bust
    env.in_play = True
    env.player_hand = bj.Hand()
    env.player_hand.add_card(bj.Card('C','2'))
    env.player_hand.add_card(bj.Card('C','2'))
    assert env._hit() == 0
    assert env.score == 0

    # always go bust
    env.in_play = True
    env.player_hand = bj.Hand()
    env.player_hand.add_card(bj.Card('C','J'))
    env.player_hand.add_card(bj.Card('C','T'))
    env.player_hand.add_card(bj.Card('C','A'))
    assert env._hit() == -1
    assert env.score == -1

def test_blackjack_stand_loss():
    env = bj.BlackJack()
    env.deck = bj.Deck()
    env.in_play = True
    
    # player 4 values
    env.player_hand = bj.Hand()
    env.player_hand.add_card(bj.Card('C','2'))
    env.player_hand.add_card(bj.Card('C','2'))
    
    # dealer 17 values
    env.dealer_hand = bj.Hand()
    env.dealer_hand.add_card(bj.Card('C', 'K'))
    env.dealer_hand.add_card(bj.Card('C', '7'))

    # loss
    reward = env._stand()
    assert len(env.dealer_hand.cards) == 2
    assert reward == -1
    assert env.score == -1
    assert env.in_play is False

def test_blackjack_stand_win():
    env = bj.BlackJack()
    env.deck = bj.Deck()
    env.in_play = True
    
    # player 21 values
    env.player_hand = bj.Hand()
    env.player_hand.add_card(bj.Card('C','A'))
    env.player_hand.add_card(bj.Card('C','9'))
    env.player_hand.add_card(bj.Card('H','A'))
    
    # dealer 18 values
    env.dealer_hand = bj.Hand()
    env.dealer_hand.add_card(bj.Card('C', '8'))
    env.dealer_hand.add_card(bj.Card('S', 'Q'))

    # win
    reward = env._stand()
    assert len(env.dealer_hand.cards) == 2
    assert reward == 1
    assert env.score == 1
    assert env.in_play is False

def test_blackjack_stand_win2():
    env = bj.BlackJack()
    env.deck = bj.Deck()
    env.deck.cards.append(bj.Card('S', 'K'))  # drawn card
    env.in_play = True
    
    # player 10 values
    env.player_hand = bj.Hand()
    env.player_hand.add_card(bj.Card('C','5'))
    env.player_hand.add_card(bj.Card('C','5'))
    
    # dealer 12 values
    env.dealer_hand = bj.Hand()
    env.dealer_hand.add_card(bj.Card('C', '8'))
    env.dealer_hand.add_card(bj.Card('S', '4'))

    # win, dealer busted
    reward = env._stand()
    assert len(env.dealer_hand.cards) == 3
    assert env.dealer_hand.get_value() == 22
    assert reward == 1
    assert env.score == 1
    assert env.in_play is False

def test_blackjack_stand_draw():
    env = bj.BlackJack()
    env.deck = bj.Deck()
    env.deck.cards.append(bj.Card('S', 'K'))  # drawn card
    env.in_play = True
    
    # player 10 values
    env.player_hand = bj.Hand()
    env.player_hand.add_card(bj.Card('C','9'))
    env.player_hand.add_card(bj.Card('C','9'))
    
    # dealer 12 values
    env.dealer_hand = bj.Hand()
    env.dealer_hand.add_card(bj.Card('C', '9'))
    env.dealer_hand.add_card(bj.Card('S', '9'))

    # draw
    reward = env._stand()
    assert len(env.dealer_hand.cards) == 2
    assert env.dealer_hand.get_value() == 18
    assert reward == 0
    assert env.score == 0
    assert env.in_play is False
