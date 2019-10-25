from playfield_controller import PlayfieldController
from playfield import *
from playfield_controller import *


# Global main function thing
def main():
    pygame.init()
    DISPLAY = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
    BIGFONT = pygame.font.Font('freesansbold.ttf', 100)
    pygame.display.set_caption('Tetromino')
    pc = PlayfieldController()

    showTextScreen('Tetromino')
    while True: # game loop
        runGame()

def runGame():
    #Draw board
    #have some time to update automatically
    #Initialize any needed variables such as TIME 
    # and bool variables for if piece has reached end or something
    for event in pygame.event.get():
        if(event.type == KEYUP):
            #do nothing?
        elif(event.type == KEYDOWN):
            if(event.key == K_LEFT):
                pc.move_left()
            elif(event.key == K_RIGHT):
                pc.move_right()
            elif(event.key == K_a):
                pc.rotate_cw():
            elif(event.key == K_d):
                pc.rotate_ccw():
            elif(event.key == K_DOWN):
                # change the update speed?
            elif(event.key == K_SPACE):
                # update necessary amount of times to reach bottom

    #Check if game_over or user terminate
    #Read event
        #Conditionals
    #Update board

'''
Events
'''

# Read in events to call playfield controller functions
