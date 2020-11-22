'''
A Tetris GUI

Press 'p' to pause, 's' to advance one timestep, 'r' to restart
'''
import math, copy, random
import torch
from modeling import MCTS
from numpy import rot90
from playfield_controller import PlayfieldController
from collections import deque

from cmu_112_graphics import *

def gameDimensions():
    rows = 20
    cols = 10
    cellSize = 30
    margin = 25
    return(rows, cols, cellSize, margin)
    
def playTetris():
    global tetris
    rows, cols, cellSize, margin = gameDimensions()
    width = cols * cellSize + cols * cellSize // 2 + 2 * margin
    height = rows * cellSize + 2 * margin
    tetris = Tetris(width=width, height=height)

class Tetris(App):

    index2name = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
    hexDigits = set('0123456789ABCDEF')

    def appStarted(app):
        app.model = torch.load("MCTS.pth")
        app.model.eval()
        app.pc = PlayfieldController()
        app.pc.update()
        app.tree = MCTS(model=app.model, pc=app.pc, gamma=0.95)

        app.rows, app.cols, app.cellSize, app.margin = gameDimensions()
        app.gameWidth = app.cols * app.cellSize
        # sidebar on right side of window
        app.sidebar = app.gameWidth // 2
        app.timerDelay = 100
        app.pause = False

        # Initialize Tetris pieces
        iPiece = [[  True,  True,  True,  True ]]
        jPiece = [[  True, False, False ],
                [  True,  True,  True ]]
        lPiece = [[ False, False,  True ],
                [  True,  True,  True ]]
        oPiece = [[  True,  True ],
                [  True,  True ]]
        sPiece = [[ False,  True,  True ],
                [  True,  True, False ]]
        tPiece = [[ False,  True, False ],
                [  True,  True,  True ]]
        zPiece = [[  True,  True, False ],
                [ False,  True,  True ]]
        app.tetrisPieces = [ iPiece, jPiece, lPiece, oPiece, 
                            sPiece, tPiece, zPiece ]
        app.tetrisPieceColors = [ '5FFBFD', '0029F5', 'F8AC3A', 'FFFC52',
                                  '65F84B', '8D34F6', 'F2361F' ]

        app.stepping = False
        #app.board = [['blue'] * app.cols for _ in range(app.rows)]

        app.moveHistory = deque(maxlen=5)

        app.updateApp()
    
    def updateApp(app):
        app.board = app.pc._playfield.board

        currIndex = app.pc._active_piece_num
        nextIndex = app.pc._next_piece_num

        app.fallingPiece = app.pc._active_piece._grid
        app.fallingPieceColor = app.tetrisPieceColors[currIndex]
        app.fallingPieceCol, app.fallingPieceRow = app.pc._active_piece.coords

        app.nextFallingPiece = app.tetrisPieces[nextIndex]
        app.nextFallingPieceColor = app.tetrisPieceColors[nextIndex]

    def takeStep(app):
        action = app.tree.search(app.pc, num_iter=50)

        name = f'{Tetris.index2name[app.pc._active_piece_num]} '
        if action == 0:
            app.moveHistory.appendleft(f'{name} moved left')
            app.pc.move_left()
        elif action == 1:
            app.moveHistory.appendleft(f'{name} moved right')
            app.pc.move_right()
        elif action == 2:
            app.moveHistory.appendleft(f'{name} rotated clockwise')
            app.pc.rotate_cw()
        else:
            app.moveHistory.appendleft(f'{name} did nothing')
        
        app.tree.root = app.tree.root.child_nodes[action]
        app.pc.update()

        app.updateApp()


    def timerFired(app):
        if (app.pause or app.pc._game_over): return
        app.takeStep()
        
        
    def keyPressed(app, event):
        if event.key == 'p':
            app.pause = not app.pause
        if event.key == 'r':
            app.appStarted()   
        if event.key == 's':
            app.pause = True
            app.takeStep() 
        
    #######################################
    # View Functions
    #######################################

    def drawBoard(app, canvas):
        #canvas.create_rectangle(0, 0, app.width, app.height, fill='orange')
        for r in range(app.rows):
            for c in range(app.cols):
                app.drawCell(canvas, r, c, app.board[c,r])

    def drawCell(app, canvas, r, c, color):
        for char in color:
            if char not in Tetris.hexDigits:
                fill = color
                break
        else:
            fill = f'#{color}'
        canvas.create_rectangle(app.getCoords(app.rows - r, c), 
                                app.getCoords(app.rows - (r+1), c + 1), 
                                fill=fill, width=1, outline='white')

    def drawFallingPiece(app, canvas):
        # Still seems kinda sketch but it mostly works
        piece = rot90(app.fallingPiece)
        rows, cols = piece.shape
        for r in range(rows):
            for c in range(cols):
                cellRow = app.fallingPieceRow - r + rows - 1
                cellCol = app.fallingPieceCol + c
                if piece[r,c]:
                    app.drawCell(canvas, cellRow, cellCol, app.fallingPieceColor)
                #else:
                    #app.drawCell(canvas, cellRow, cellCol, 'gray')

    def getCoords(app, r, c):
        # gets (x, y) coordinates of given row and cell on the board
        x0 = app.margin + c * app.cellSize
        y0 = app.margin + r * app.cellSize
        return (x0, y0)

    def getCell(app, x, y):
        # gets row & column of cell on the board from the given (x, y) coordinates
        # not actually used anywhere
        r = (y - app.margin) // app.cellSize
        c = (x - app.margin) // app.cellSize
        return (r, c)

    def drawSidebar(app, canvas):
        center = app.gameWidth + 1.5 * app.margin + app.sidebar // 2
        dy = app.height // 10
        yText1 = dy
        boxMargin = app.margin // 2

        yText3 = dy * 3

        canvas.create_text(center, yText1, text='Tetris!',
                        fill='black', font=f'Menlo {app.sidebar // 5} bold')

        canvas.create_rectangle(app.gameWidth + app.margin + boxMargin, dy * 2,
            app.width - boxMargin, dy * 6, fill = 'white', width = 5)
        canvas.create_text(center, yText3, anchor='n', text='Next Piece:', 
                        fill='black', font = f'Menlo {app.sidebar // 8} bold')
        cellSize = app.sidebar / 6
        xPiece = center - len(app.nextFallingPiece[0]) * cellSize // 2 
        yPiece = dy * 4
        app.drawNextFallingPiece(canvas, (xPiece, yPiece), cellSize)

        for i, move in enumerate(app.moveHistory):
            canvas.create_text(center, dy * (7.5 + i/3), text=move,
                               font=f'Menlo {app.sidebar//12}')

    def drawNextFallingPiece(app, canvas, topLeftCorner, cellSize):
        xPiece, yPiece = topLeftCorner
        for r in range(len(app.nextFallingPiece)):
            for c in range(len(app.nextFallingPiece[0])):
                if(app.nextFallingPiece[r][c]):
                    x0, y0, x1, y1 = c * cellSize, r * cellSize, \
                        (c + 1) * cellSize, (r + 1) * cellSize
                    canvas.create_rectangle(
                        x0 + xPiece, y0 + yPiece, x1 + xPiece, y1 + yPiece,
                        fill=f'#{app.nextFallingPieceColor}', width=3)

    def drawPauseMessage(app, canvas):
        (x0, y0), (x1, y1) = app.getCoords(app.rows // 2 - 1, 0), \
            app.getCoords(app.rows // 2 + 1, app.cols)

        canvas.create_rectangle(x0, y0, x1, y1, 
            fill='black', outline='white')

        canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, 
            text = 'Game Paused', font = f'Menlo {app.gameWidth // 10}',
            fill = 'white', anchor = 'center')

    def drawGameOverMessage(app, canvas):
        message = 'Game Over!'
        
        (x0, y0), (x1, y1) = app.getCoords(app.rows // 4 - 1, 0), \
            app.getCoords(app.rows // 4 + 1, app.cols)

        canvas.create_rectangle(x0, y0, x1, y1, 
            fill='black', outline='white')
        
        canvas.create_text((x0 + x1) / 2, y0 + (y1 - y0) / 2,
            text = message, font = f'Menlo {app.gameWidth // 10}', 
            fill = 'yellow', anchor = 'center')
        
    def redrawAll(app, canvas):
        app.drawBoard(canvas)
        app.drawFallingPiece(canvas)
        app.drawSidebar(canvas)
        if(app.pc._game_over):
            app.drawGameOverMessage(canvas)
        # elif(app.pause):
        #     app.drawPauseMessage(canvas)

#################################################
# main
#################################################

def main():
    playTetris()

if __name__ == '__main__':
    main()
