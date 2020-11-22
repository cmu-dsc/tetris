import math, copy, random
import torch

from cmu_112_graphics import *

#################################################
# Helper functions
#################################################

def almostEqual(d1, d2, epsilon=10**-7):
    return (abs(d2 - d1) < epsilon)

import decimal
def roundHalfUp(d):
    # Round to nearest with ties going away from zero.
    rounding = decimal.ROUND_HALF_UP
    # See other rounding options here:
    # https://docs.python.org/3/library/decimal.html#rounding-modes
    return int(decimal.Decimal(d).to_integral_value(rounding=rounding))

#################################################
# Tetris App
#################################################

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
    tetris = TetrisApp(width=width, height=height)

class TetrisApp(App):

    def getRawState(app):
        boardState = torch.tensor([0 if elem == 'blue' else 1 
                       for row in app.board for elem in row]).float()
        
        fallingPiece = torch.zeros(7).float()
        fallingPiece[app.fallingPieceIndex] = 1
        
        nextPiece = torch.zeros(7).float()
        nextPiece[app.nextFallingPieceIndex] = 1

        location = torch.tensor((app.fallingPieceCol,     # x-coordinate,
                                 app.fallingPieceRow)     # y-coordinate
                               ).float()    

        state = torch.cat([boardState, location, fallingPiece, nextPiece])
        return state
        
    def getNextAction(app):
        state = app.getRawState()
        prob_action = app.model(state)
        action = torch.argmax(prob_action).item()
        return action

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

    def appStarted(app):
        app.model = torch.load("DDQN_sparse_0.999.pth")
        app.model.eval()

        app.rows, app.cols, app.cellSize, app.margin = gameDimensions()
        app.gameWidth = app.cols * app.cellSize

        # sidebar on right side of window
        app.sidebar = app.gameWidth // 2

        app.timerDelay = 100
        app.scores = [] # list of top 10 (or fewer) scores from previous games

        app.makePieces()
        app.restart()

    def makePieces(app):
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
        app.tetrisPieceColors = [ "red", "yellow", "magenta", "pink", 
                                "cyan", "green", "orange" ]

    def restart(app):
        app.board = [['blue'] * app.cols for _ in range(app.rows)]
        app.fallingPiece = None
        app.fallingPieceColor = None
        app.numFallingPieceCols = None
        app.fallingPieceLeftLocation = None
        app.fallingPieceTopLocation = None

        app.nextFallingPiece = None
        app.nextFallingPieceColor = None
        app.score = 0

        app.gameOver = False
        app.pause = False

        app.newFallingPiece()

    def timerFired(app):
        if app.pause or app.gameOver: return

        action = app.getNextAction()
        if action == 0:
            app.moveFallingPiece(0, -1)
        elif action == 1:
            app.moveFallingPiece(0, 1)
        elif action == 2:
            app.rotateFallingPiece()

        if(not app.moveFallingPiece(1, 0)):
            # runs when game is not paused or over 
            # and piece cannot move any further down
            app.placeFallingPiece()
            app.newFallingPiece()
            app.removeFullRows()
            if(not app.fallingPieceIsLegal(app.fallingPieceRow, 
                                           app.fallingPieceCol)):
                app.gameOver = True
                app.scores.append(app.score)
                app.scores.sort(reverse=True)
                # only top 10 scores stored
                if(len(app.scores) > 10): app.scores = app.scores[:10]

    def drawBoard(app, canvas):
        canvas.create_rectangle(0, 0, app.width, app.height, fill='orange')
        for r in range(app.rows):
            for c in range(app.cols):
                app.drawCell(canvas, r, c, app.board[r][c])

    def drawCell(app, canvas, r, c, color):
        canvas.create_rectangle(app.getCoords(r, c), 
                                app.getCoords(r + 1, c + 1), 
                                fill=color, width=3)

    def newFallingPiece(app):
        # assigns values from nextFallingPiece to fallingPiece, 
        # then generates a new nextFallingPiece
        if(app.nextFallingPiece != None):
            app.fallingPiece = app.nextFallingPiece
            app.fallingPieceColor = app.nextFallingPieceColor
            app.fallingPieceIndex = app.nextFallingPieceIndex
        else:
            #randNum = random.randint(0, len(app.tetrisPieces) - 1)
            randNum = random.sample([0,3],1)[0]
            app.fallingPiece = app.tetrisPieces[randNum]
            app.fallingPieceColor = app.tetrisPieceColors[randNum]
            app.fallingPieceIndex = randNum
        
        #randNum = random.randint(0, len(app.tetrisPieces) - 1)
        randNum = random.sample([0,3],1)[0]
        app.nextFallingPiece = app.tetrisPieces[randNum]
        app.nextFallingPieceColor = app.tetrisPieceColors[randNum]
        app.nextFallingPieceIndex = randNum

        app.numFallingPieceCols = len(app.fallingPiece[0])
        
        # (app.fallingPieceCol, app.fallingPieceRow) represents the 
        # location of the top left corner of the piece on the board
        app.fallingPieceRow = 0
        app.fallingPieceCol = app.cols // 2 - app.numFallingPieceCols // 2
        
    def drawFallingPiece(app, canvas):
        if(app.fallingPiece == None): return None
        for r in range(len(app.fallingPiece)):
            for c in range(len(app.fallingPiece[0])):
                if(app.fallingPiece[r][c]):
                    cellRow = app.fallingPieceRow + r
                    cellCol = app.fallingPieceCol + c
                    app.drawCell(canvas, cellRow, cellCol, app.fallingPieceColor)

    def moveFallingPiece(app, drow, dcol):
        newRow = app.fallingPieceRow + drow
        newCol = app.fallingPieceCol + dcol
        if(app.fallingPieceIsLegal(newRow, newCol)):
            app.fallingPieceRow = newRow
            app.fallingPieceCol = newCol
            return True
        else: return False

    def fallingPieceIsLegal(app, newRow, newCol):
        for r in range(len(app.fallingPiece)):
            for c in range(len(app.fallingPiece[0])):
                if(not app.fallingPiece[r][c]): continue
                cellRow = newRow + r
                cellCol = newCol + c
                if(cellRow < 0 or cellRow >= app.rows or
                cellCol < 0 or cellCol >= app.cols or
                app.board[cellRow][cellCol] != 'blue'):
                    return False
        return True

    def keyPressed(app, event):
        if(event.key == 'p'):
            app.pause = not app.pause
        elif(event.key == 'Left' or event.key == 'a'):
            if(not app.pause): app.moveFallingPiece(0, -1)
        elif(event.key == 'Right' or event.key == 'd'):
            if(not app.pause): app.moveFallingPiece(0, 1)
        elif(event.key == 'Down' or event.key == 's'):
            if(not app.pause): app.moveFallingPiece(1, 0)
        elif(event.key == 'Up' or event.key == 'w'):
            if(not app.pause): app.rotateFallingPiece()
        elif(event.key == 'Space'):
            if(not app.pause): app.hardDrop()
        elif(event.key == 'r'):
            app.restart()

    def rotateFallingPiece(app):
        oldPiece = app.fallingPiece
        oldCols = app.numFallingPieceCols
        oldRows = len(app.fallingPiece)

        app.fallingPiece = [[False] * oldRows for _ in range(oldCols)]
        app.numFallingPieceCols = oldRows

        oldRow = app.fallingPieceRow
        oldCol = app.fallingPieceCol
        app.fallingPieceRow = oldRow + oldRows // 2 - oldCols // 2
        app.fallingPieceCol = oldCol + oldCols // 2 - oldRows // 2

        for r in range(oldCols):
            for c in range(oldRows):
                app.fallingPiece[r][c] = oldPiece[c][oldCols - 1 - r]

        if(not app.fallingPieceIsLegal(app.fallingPieceRow, app.fallingPieceCol)):
            app.fallingPiece = oldPiece
            app.numFallingPieceCols = oldCols
            app.fallingPieceRow = oldRow
            app.fallingPieceCol = oldCol

    def placeFallingPiece(app):
        for r in range(len(app.fallingPiece)):
            for c in range(len(app.fallingPiece[0])):
                if(not app.fallingPiece[r][c]): continue
                cellRow = app.fallingPieceRow + r
                cellCol = app.fallingPieceCol + c
                app.board[cellRow][cellCol] = app.fallingPieceColor

    def removeFullRows(app):
        removed = 0
        for row in range(len(app.board)):
            if(TetrisApp.isFull(app.board[row])):
                removed += 1
                app.board.pop(row)
                app.board.insert(0, ['blue'] * app.cols)
        app.score += removed ** 2

    @staticmethod
    def isFull(line):
        return not 'blue' in line

    def hardDrop(app):
        while(app.moveFallingPiece(1, 0)):
            pass

    def drawSidebar(app, canvas):
        center = app.gameWidth + 1.5 * app.margin + app.sidebar // 2
        dy = app.height // 10
        yText1 = dy
        boxMargin = app.margin // 2

        yText3 = dy * 4
        canvas.create_rectangle(app.gameWidth + app.margin + boxMargin, dy * 3,
            app.width - boxMargin, dy * 7, fill = 'white', width = 5)
        canvas.create_text(center, yText3, anchor='n', text='Next Piece:', 
                        fill='black', 
                        font = f'Courier {app.sidebar // 8} bold')
        cellSize = app.sidebar / 6
        xPiece = center - len(app.nextFallingPiece[0]) * cellSize // 2 
        yPiece = dy * 5
        app.drawNextFallingPiece(canvas, (xPiece, yPiece), cellSize)

    def drawNextFallingPiece(app, canvas, topLeftCorner, cellSize):
        xPiece, yPiece = topLeftCorner
        for r in range(len(app.nextFallingPiece)):
            for c in range(len(app.nextFallingPiece[0])):
                if(app.nextFallingPiece[r][c]):
                    x0, y0, x1, y1 = c * cellSize, r * cellSize, \
                        (c + 1) * cellSize, (r + 1) * cellSize
                    canvas.create_rectangle(
                        x0 + xPiece, y0 + yPiece, x1 + xPiece, y1 + yPiece,
                        fill=app.nextFallingPieceColor, width=3)

    def drawScore(app, canvas):
        xText = app.margin + app.gameWidth // 2
        yText = app.margin // 2
        canvas.create_text(xText, yText, text = f'Score: {app.score}', 
            fill = 'blue', font = f'Courier {app.margin // 3 * 2} bold')

    def drawHighScores(app, canvas):
        # dimensions of high scores box
        (x0, y0), (x1, y1) = app.getCoords(app.rows // 4 + 2, 1), \
            app.getCoords(app.rows - 2, app.cols - 1)
        canvas.create_rectangle(x0, y0, x1, y1, fill='white', width=3)
        
        xCenter = (x0 + x1) // 2
        dy = (y1 - y0) // 13 # vertical increment for each score displayed
        canvas.create_text(xCenter, y0 + dy, text='High Scores',
            font=f'Courier {dy * 3 // 2} bold')

        # display top 10 high scores
        for i in range(0, 11):
            if(i == 0 ):
                index, score = 'Rank', 'Score'
            else:
                index = i
                score = app.scores[i - 1] if (i - 1 < len(app.scores)) else '-'
            canvas.create_text(xCenter - (x0 + x1) // 8, y0 + (i + 2) * dy, 
                anchor='n', text=f'{index}', font=f'Courier {dy} bold')
            canvas.create_text(xCenter + (x0 + x1) // 8, y0 + (i + 2) * dy, 
                text=f'{score}', anchor='n', font=f'Courier {dy} bold')

    def drawPauseMessage(app, canvas):
        (x0, y0), (x1, y1) = app.getCoords(app.rows // 2 - 1, 0), \
            app.getCoords(app.rows // 2 + 1, app.cols)

        canvas.create_rectangle(x0, y0, x1, y1, 
            fill='black')

        canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, 
            text = 'Game Paused', font = f'Courier {app.gameWidth // 10}',
            fill = 'white', anchor = 'center')

    def drawGameOverMessage(app, canvas):
        message = 'Game Over!'
        
        (x0, y0), (x1, y1) = app.getCoords(app.rows // 4 - 1, 0), \
            app.getCoords(app.rows // 4 + 1, app.cols)

        canvas.create_rectangle(x0, y0, x1, y1, 
            fill='black')
        
        canvas.create_text((x0 + x1) / 2, y0 + (y1 - y0) / 2,
            text = message, font = f'Courier {app.gameWidth // 10}', 
            fill = 'yellow', anchor = 'center')
        
    def redrawAll(app, canvas):
        app.drawBoard(canvas)
        app.drawFallingPiece(canvas)
        app.drawScore(canvas)
        app.drawSidebar(canvas)
        if(app.gameOver):
            app.drawGameOverMessage(canvas)
        elif(app.pause):
            app.drawPauseMessage(canvas)

#################################################
# main
#################################################

def main():
    playTetris()

if __name__ == '__main__':
    main()
