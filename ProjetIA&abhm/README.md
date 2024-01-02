# Hex Game

Place tiles to connect your two borders of the board. Play against an AI, implement human vs. human game mode or create your own AI strategy in `strategy.py`.

## Dependencies

`pip install -r requirements`


## Run the program

```
$ python main.py -h

usage: main.py [-h] [--size SIZE] [--games GAMES] [--no-ui] [--player {human,random,minimax}] [--other {human,random,minimax}]

Runs a game of Hex.

options:
  -h, --help            show this help message and exit
  --size SIZE           Size of the board (Default: 7)
  --games GAMES         Number of games to play in tournament. Only if no human (default: 5)
  --no-ui               GUI is not displayed. Only if no human.
  --player {human,random,minimax}
                        Strategy for player1 (default: human)
  --other {human,random,minimax}
                        Strategy for player2 (default: random)
```

## Implementing a new strategie

1. Extend the `Player` class
2. Implement the `start()` function
3. Add your newly created class in `str2strat` at the bottom of the `strategy.py` file
4. Run the program with `python main.py --other my_new_ai_strat` to experiment !

Tip: to debug your AI use a small board (2, 3, 4).

## Comparing AI strategies

Use the `--no-ui` and `--games` options to run a lot of simulation quickly.
For example, `python main.py --player random --other minimax --no-ui -- games 100` will run 100 games between random AI and minimax AI.
