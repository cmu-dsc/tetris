from playfield_controller import PlayfieldController
from gamestate import *
from playfield import *
from playfield_controller import *
from numpy.random import randint
pc = PlayfieldController()
pc._playfield.insert_piece(O((-1,-1)),(-1,-1))
pc._playfield.insert_piece(O((1,-1)),(1,-1))
pc._playfield.insert_piece(O((3,-1)),(3,-1))
pc._playfield.insert_piece(O((5,-1)),(5,-1))
pc._rng_queue = np.array([O])
pc.update()
pc.move_right()
pc.move_right()
pc.move_right()
pc.move_right()
pc.move_right()
pc.move_right()
pc.move_right()
for i in range(100):
    pc.update()
    gs = pc.gamestate()
    gs.plot()
while not pc.game_over:
    # num_moves = randint(5)
    num_moves = 1
    for i in range(num_moves):
        move = randint(0,4)
        if move == 0:
            pc.move_left()
            gs = pc.gamestate()
            gs.plot()
        elif move == 1:
            pc.move_right()
            gs = pc.gamestate()
            gs.plot()
        elif move == 1:
            pc.rotate_cw()
            gs = pc.gamestate()
            gs.plot()
        elif move == 3:
            pc.rotate_ccw()
            gs = pc.gamestate()
            gs.plot()
    pc.update()
    gs = pc.gamestate()
    gs.plot()
