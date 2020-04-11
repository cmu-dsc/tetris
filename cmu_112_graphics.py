# cmu_112_graphics.py
# version 0.8.0

# Pre-release for CMU 15-112-f19

# Require Python 3.6 or later
import sys
if ((sys.version_info[0] != 3) or (sys.version_info[1] < 6)):
    raise Exception('cmu_112_graphics.py requires Python version 3.6 or later.')

# Track version and file update timestamp
import datetime
MAJOR_VERSION = 0
MINOR_VERSION = 8.0 # version 0.8.0
LAST_UPDATED  = datetime.date(year=2019, month=10, day=7)

# Pending changes:
#   * Fix Windows-only bug: Position popup dialog box over app window (already works fine on Macs)
#   * Add documentation
#   * integrate sounds (probably from pyGame)
#   * Improved methodIsOverridden to TopLevelApp and ModalApp
#   * Save to animated gif and/or mp4 (with audio capture?)

# Deferred changes:
#   * replace/augment tkinter canvas with PIL/Pillow imageDraw (perhaps with our own fn names)
#   * use snake_case and CapWords

# Changes in v0.8.0
#   * suppress more modifier keys (Super_L, Super_R, ...)
#   * raise exception on event.keysym or event.char + works with key = 'Enter'
#   * remove tryToInstall

# Changes in v0.7.4
#   * renamed drawAll back to redrawAll :-)

# Changes in v0.7.3
#   * Ignore mousepress-drag-release and defer configure events for drags in titlebar
#   * Extend deferredRedrawAll to 100ms with replace=True and do not draw while deferred
#     (together these hopefully fix Windows-only bug: file dialog makes window not moveable)
#   * changed sizeChanged to not take event (use app.width and app.height)

# Changes in v0.7.2
#   * Singleton App.theRoot instance (hopefully fixes all those pesky Tkinter errors-on-exit)
#   * Use user32.SetProcessDPIAware to get resolution of screen grabs right on Windows-only (fine on Macs)
#   * Replaces showGraphics() with runApp(...), which is a veneer for App(...) [more intuitive for pre-OOP part of course]
#   * Fixes/updates images:
#       * disallows loading images in redrawAll (raises exception)
#       * eliminates cache from loadImage
#       * eliminates app.getTkinterImage, so user now directly calls ImageTk.PhotoImage(image))
#       * also create_image allows magic pilImage=image instead of image=ImageTk.PhotoImage(app.image)

# Changes in v0.7.1
#   * Added keyboard shortcut:
#       * cmd/ctrl/alt-x: hard exit (uses os._exit() to exit shell without tkinter error messages)
#   * Fixed bug: shortcut keys stopped working after an MVC violation (or other exception)
#   * In app.saveSnapshot(), add .png to path if missing
#   * Added: Print scripts to copy-paste into shell to install missing modules (more automated approaches proved too brittle)

# Changes in v0.7
#   * Added some image handling (requires PIL (retained) and pyscreenshot (later removed):
#       * app.loadImage()       # loads PIL/Pillow image from file, with file dialog, or from URL (http or https)
#       * app.scaleImage()      # scales a PIL/Pillow image
#       * app.getTkinterImage() # converts PIL/Pillow image to Tkinter PhotoImage for use in create_image(...)
#       * app.getSnapshot()     # get a snapshot of the canvas as a PIL/Pillow image
#       * app.saveSnapshot()    # get and save a snapshot
#   * Added app.paused, app.togglePaused(), and paused highlighting (red outline around canvas when paused)
#   * Added keyboard shortcuts:
#       * cmd/ctrl/alt-s: save a snapshot
#       * cmd/ctrl/alt-p: pause/unpause
#       * cmd/ctrl/alt-q: quit

# Changes in v0.6:
#   * Added fnPrefix option to TopLevelApp (so multiple TopLevelApp's can be in one file)
#   * Added showGraphics(drawFn) (for graphics-only drawings before we introduce animations)

# Changes in v0.5:
#   * Added:
#       * app.winx and app.winy (and add winx,winy parameters to app.__init__, and sets these on configure events)
#       * app.setSize(width, height)
#       * app.setPosition(x, y)
#       * app.quit()
#       * app.showMessage(message)
#       * app.getUserInput(prompt)
#       * App.lastUpdated (instance of datetime.date)
#   * Show popup dialog box on all exceptions (not just for MVC violations)
#   * Draw (in canvas) "Exception!  App Stopped! (See console for details)" for any exception
#   * Replace callUserMethod() with more-general @safeMethod decorator (also handles exceptions outside user methods)
#   * Only include lines from user's code (and not our framework nor tkinter) in stack traces
#   * Require Python version (3.6 or greater)

# Changes in v0.4:
#   * Added __setattr__ to enforce Type 1A MVC Violations (setting app.x in redrawAll) with better stack trace
#   * Added app.deferredRedrawAll() (avoids resizing drawing/crashing bug on some platforms)
#   * Added deferredMethodCall() and app._afterIdMap to generalize afterId handling
#   * Use (_ is None) instead of (_ == None)

# Changes in v0.3:
#   * Fixed "event not defined" bug in sizeChanged handlers.
#   * draw "MVC Violation" on Type 2 violation (calling draw methods outside redrawAll)

# Changes in v0.2:
#   * Handles another MVC violation (now detects drawing on canvas outside of redrawAll)
#   * App stops running when an exception occurs (in user code) (stops cascading errors)

# Changes in v0.1:
#   * OOPy + supports inheritance + supports multiple apps in one file + etc
#        * uses import instead of copy-paste-edit starter code + no "do not edit code below here!"
#        * no longer uses Struct (which was non-Pythonic and a confusing way to sort-of use OOP)
#   * Includes an early version of MVC violation handling (detects model changes in redrawAll)
#   * added events:
#       * appStarted (no init-vs-__init__ confusion)
#       * appStopped (for cleanup)
#       * keyReleased (well, sort of works) + mouseReleased
#       * mouseMoved + mouseDragged
#       * sizeChanged (when resizing window)
#   * improved key names (just use event.key instead of event.char and/or event.keysym + use names for 'Enter', 'Escape', ...)
#   * improved function names (renamed redrawAll to drawAll)
#   * improved (if not perfect) exiting without that irksome Tkinter error/bug
#   * app has a title in the titlebar (also shows window's dimensions)
#   * supports Modes and ModalApp (see ModalApp and Mode, and also see TestModalApp example)
#   * supports TopLevelApp (using top-level functions instead of subclasses and methods)
#   * supports version checking with App.majorVersion, App.minorVersion, and App.version
#   * logs drawing calls to support autograding views (still must write that autograder, but this is a very helpful first step)

from tkinter import *
from tkinter import messagebox, simpledialog, filedialog
import inspect, copy, traceback
import sys, os
from io import BytesIO

def failedImport(importName, installName=None):
    installName = installName or importName
    print('**********************************************************')
    print(f'** Cannot import {importName} -- it seems you need to install {installName}')
    print(f'** This may result in limited functionality or even a runtime error.')
    print('**********************************************************')
    print()

try: from PIL import Image, ImageTk, ImageGrab
except ModuleNotFoundError: failedImport('PIL', 'pillow')

try: import requests
except ModuleNotFoundError: failedImport('requests')

def getHash(obj):
    # This is used to detect MVC violations in redrawAll
    # @TODO: Make this more robust and efficient
    try:
        return getHash(obj.__dict__)
    except:
        if (isinstance(obj, list)): return getHash(tuple([getHash(v) for v in obj]))
        elif (isinstance(obj, set)): return getHash(sorted(obj))
        elif (isinstance(obj, dict)): return getHash(tuple([obj[key] for key in sorted(obj)]))
        else:
            try: return hash(obj)
            except: return getHash(repr(obj))

class WrappedCanvas(Canvas):
    # Enforces MVC: no drawing outside calls to redrawAll
    # Logs draw calls (for autograder) in canvas.loggedDrawingCalls
    def __init__(wrappedCanvas, app):
        wrappedCanvas.loggedDrawingCalls = [ ]
        wrappedCanvas.logDrawingCalls = True
        wrappedCanvas.inRedrawAll = False
        wrappedCanvas.app = app
        super().__init__(app._root, width=app.width, height=app.height)

    def log(self, methodName, args, kwargs):
        if (not self.inRedrawAll):
            self.app.mvcViolation('you may not use the canvas (the view) outside of redrawAll')
        if (self.logDrawingCalls):
            self.loggedDrawingCalls.append((methodName, args, kwargs))

    def create_arc(self, *args, **kwargs): self.log('create_arc', args, kwargs); return super().create_arc(*args, **kwargs)
    def create_bitmap(self, *args, **kwargs): self.log('create_bitmap', args, kwargs); return super().create_bitmap(*args, **kwargs)
    def create_line(self, *args, **kwargs): self.log('create_line', args, kwargs); return super().create_line(*args, **kwargs)
    def create_oval(self, *args, **kwargs): self.log('create_oval', args, kwargs); return super().create_oval(*args, **kwargs)
    def create_polygon(self, *args, **kwargs): self.log('create_polygon', args, kwargs); return super().create_polygon(*args, **kwargs)
    def create_rectangle(self, *args, **kwargs): self.log('create_rectangle', args, kwargs); return super().create_rectangle(*args, **kwargs)
    def create_text(self, *args, **kwargs): self.log('create_text', args, kwargs); return super().create_text(*args, **kwargs)
    def create_window(self, *args, **kwargs): self.log('create_window', args, kwargs); return super().create_window(*args, **kwargs)

    def create_image(self, *args, **kwargs):
        self.log('create_image', args, kwargs);
        usesImage = 'image' in kwargs
        usesPilImage = 'pilImage' in kwargs
        if ((not usesImage) and (not usesPilImage)):
            raise Exception('create_image requires an image to draw')
        elif (usesImage and usesPilImage):
            raise Exception('create_image cannot use both an image and a pilImage')
        elif (usesPilImage):
            pilImage = kwargs['pilImage']
            del kwargs['pilImage']
            if (not isinstance(pilImage, Image.Image)):
                raise Exception('create_image: pilImage value is not an instance of a PIL/Pillow image')
            image = ImageTk.PhotoImage(pilImage)
        else:
            image = kwargs['image']
            if (isinstance(image, Image.Image)):
                raise Exception('create_image: image must not be an instance of a PIL/Pillow image\n' +
                    'You perhaps meant to convert from PIL to Tkinter, like so:\n' +
                    '     canvas.create_image(x, y, image=ImageTk.PhotoImage(image))')
        kwargs['image'] = image
        return super().create_image(*args, **kwargs)

class App(object):
    majorVersion = MAJOR_VERSION
    minorVersion = MINOR_VERSION
    version = f'{majorVersion}.{minorVersion}'
    lastUpdated = LAST_UPDATED
    theRoot = None # singleton Tkinter root object

    ####################################
    # User Methods:
    ####################################
    def redrawAll(app, canvas): pass      # draw (view) the model in the canvas
    def appStarted(app): pass           # initialize the model (app.xyz)
    def appStopped(app): pass           # cleanup after app is done running
    def keyPressed(app, event): pass    # use event.key
    def keyReleased(app, event): pass   # use event.key
    def mousePressed(app, event): pass  # use event.x and event.y
    def mouseReleased(app, event): pass # use event.x and event.y
    def mouseMoved(app, event): pass    # use event.x and event.y
    def mouseDragged(app, event): pass  # use event.x and event.y
    def timerFired(app): pass           # respond to timer events
    def sizeChanged(app): pass          # respond to window size changes

    ####################################
    # Implementation:
    ####################################

    def __init__(app, width=300, height=300, x=0, y=0, title=None, autorun=True, mvcCheck=True, logDrawingCalls=True):
        app.winx, app.winy, app.width, app.height = x, y, width, height
        app.timerDelay = 100     # milliseconds
        app.mouseMovedDelay = 50 # ditto
        app.title = title
        app.mvcCheck = mvcCheck
        app.logDrawingCalls = logDrawingCalls
        app.running = app.paused = False
        app.mousePressedOutsideWindow = False
        if autorun: app.run()

    def setSize(app, width, height):
        app._root.geometry(f'{width}x{height}')

    def setPosition(app, x, y):
        app._root.geometry(f'+{x}+{y}')

    def showMessage(app, message):
        messagebox.showinfo('showMessage', message, parent=app._root)

    def getUserInput(app, prompt):
        return simpledialog.askstring('getUserInput', prompt)

    def loadImage(app, path=None):
        if (app._canvas.inRedrawAll):
            raise Exception('Cannot call loadImage in redrawAll')
        if (path is None):
            path = filedialog.askopenfilename(initialdir=os.getcwd(), title='Select file: ',filetypes = (('Image files','*.png *.gif *.jpg'),('all files','*.*')))
            if (not path): return None
        if (path.startswith('http')):
            response = requests.request('GET', path) # path is a URL!
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(path)
        return image

    def scaleImage(app, image, scale, antialias=False):
        # antialiasing is higher-quality but slower
        resample = Image.ANTIALIAS if antialias else Image.NEAREST
        return image.resize((round(image.width*scale), round(image.height*scale)), resample=resample)

    def getSnapshot(app):
        app.showRootWindow()
        x0 = app._root.winfo_rootx() + app._canvas.winfo_x()
        y0 = app._root.winfo_rooty() + app._canvas.winfo_y()
        result = ImageGrab.grab((x0,y0,x0+app.width,y0+app.height))
        return result

    def saveSnapshot(app):
        path = filedialog.asksaveasfilename(initialdir=os.getcwd(), title='Select file: ',filetypes = (('png files','*.png'),('all files','*.*')))
        if (path):
            # defer call to let filedialog close (and not grab those pixels)
            if (not path.endswith('.png')): path += '.png'
            app.deferredMethodCall(afterId='saveSnapshot', afterDelay=0, afterFn=lambda:app.getSnapshot().save(path))

    def togglePaused(app):
        app.paused = not app.paused

    def quit(app):
        app.running = False
        app._root.quit() # break out of root.mainloop() without closing window!

    def __setattr__(app, attr, val):
        d = app.__dict__
        d[attr] = val
        canvas = d.get('_canvas', None)
        if (d.get('running', False) and
            d.get('mvcCheck', False) and
            (canvas is not None) and
            canvas.inRedrawAll):
            app.mvcViolation(f'you may not change app.{attr} in the model while in redrawAll (the view)')

    def printUserTraceback(app, exception, tb):
        stack = traceback.extract_tb(tb)
        lines = traceback.format_list(stack)
        inRedrawAllWrapper = False
        printLines = [ ]
        for line in lines:
            if (('"cmu_112_graphics.py"' not in line) and
                ('/cmu_112_graphics.py' not in line) and
                ('\\cmu_112_graphics.py' not in line) and
                ('/tkinter/' not in line) and
                ('\\tkinter\\' not in line)):
                printLines.append(line)
            if ('redrawAllWrapper' in line):
                inRedrawAllWrapper = True
        if (len(printLines) == 0):
            # No user code in trace, so we have to use all the code (bummer),
            # but not if we are in a redrawAllWrapper...
            if inRedrawAllWrapper:
                printLines = ['    No traceback available. Error occurred in redrawAll.\n']
            else:
                printLines = lines
        print('Traceback (most recent call last):')
        for line in printLines: print(line, end='')
        print(f'Exception: {exception}')

    def safeMethod(appMethod):
        def m(*args, **kwargs):
            app = args[0]
            try:
                return appMethod(*args, **kwargs)
            except Exception as e:
                app.running = False
                app.printUserTraceback(e, sys.exc_info()[2])
                app._canvas.inRedrawAll = True # not really, but stops recursive MVC Violations!
                app._canvas.create_rectangle(0, 0, app.width, app.height, fill=None, width=10, outline='red')
                app._canvas.create_rectangle(10, app.height-50, app.width-10, app.height-10,
                                             fill='white', outline='red', width=4)
                app._canvas.create_text(app.width/2, app.height-40, text=f'Exception! App Stopped!', fill='red', font='Arial 12 bold')
                app._canvas.create_text(app.width/2, app.height-20, text=f'See console for details', fill='red', font='Arial 12 bold')
                app._canvas.update()
                app.showMessage(f'Exception: {e}\nClick ok then see console for details.')
        return m

    def methodIsOverridden(app, methodName):
        return (getattr(type(app), methodName) is not getattr(App, methodName))

    def mvcViolation(app, errMsg):
        app.running = False
        raise Exception('MVC Violation: ' + errMsg)

    @safeMethod
    def redrawAllWrapper(app):
        if (not app.running): return
        if ('deferredRedrawAll' in app._afterIdMap): return # wait for pending call
        app._canvas.inRedrawAll = True
        app._canvas.delete(ALL)
        width,outline = (10,'red') if app.paused else (0,'white')
        app._canvas.create_rectangle(0, 0, app.width, app.height, fill='white', width=width, outline=outline)
        app._canvas.loggedDrawingCalls = [ ]
        app._canvas.logDrawingCalls = app.logDrawingCalls
        hash1 = getHash(app) if app.mvcCheck else None
        try:
            app.redrawAll(app._canvas)
            hash2 = getHash(app) if app.mvcCheck else None
            if (hash1 != hash2):
                app.mvcViolation('you may not change the app state (the model) in redrawAll (the view)')
        finally:
            app._canvas.inRedrawAll = False
        app._canvas.update()

    def deferredMethodCall(app, afterId, afterDelay, afterFn, replace=False):
        def afterFnWrapper():
            app._afterIdMap.pop(afterId, None)
            afterFn()
        id = app._afterIdMap.get(afterId, None)
        if ((id is None) or replace):
            if id: app._root.after_cancel(id)
            app._afterIdMap[afterId] = app._root.after(afterDelay, afterFnWrapper)

    def deferredRedrawAll(app):
        app.deferredMethodCall(afterId='deferredRedrawAll', afterDelay=100, afterFn=app.redrawAllWrapper, replace=True)

    @safeMethod
    def appStartedWrapper(app):
        app.appStarted()
        app.redrawAllWrapper()

    keyNameMap = { '\t':'Tab', '\n':'Enter', '\r':'Enter', '\b':'Backspace',
                   chr(127):'Delete', chr(27):'Escape', ' ':'Space' }

    @staticmethod
    def useEventKey(attr):
        raise Exception(f'Use event.key instead of event.{attr}')

    @staticmethod
    def getEventKeyInfo(event, keysym, char):
        key = c = char
        hasModifiers = (event.state != 0)
        if ((c in [None, '']) or (len(c) > 1) or (ord(c) > 255)):
            key = keysym
            if (key.endswith('_L') or
                key.endswith('_R') or
                key.endswith('_Lock')):
                key = 'Modifier_Key'
        elif (c in App.keyNameMap):
            key = App.keyNameMap[c]
        elif ((len(c) == 1) and (1 <= ord(c) <= 26)):
            key = chr(ord('a')-1 + ord(c))
            hasModifiers = True
        return key, hasModifiers

    class KeyEventWrapper(Event):
        def __init__(self, event):
            keysym, char = event.keysym, event.char
            del event.keysym
            del event.char
            for key in event.__dict__:
                if (not key.startswith('__')):
                    self.__dict__[key] = event.__dict__[key]
            self.key, self.hasModifiers = App.getEventKeyInfo(event, keysym, char)
        keysym = property(lambda *args: App.useEventKey('keysym'),
                          lambda *args: App.useEventKey('keysym'))
        char =   property(lambda *args: App.useEventKey('char'),
                          lambda *args: App.useEventKey('char'))

    @safeMethod
    def keyPressedWrapper(app, event):
        event = App.KeyEventWrapper(event)
        if ((event.key == 's') and (event.hasModifiers)):
            app.saveSnapshot()
        elif ((event.key == 'p') and (event.hasModifiers)):
            app.togglePaused()
            app.redrawAllWrapper()
        elif ((event.key == 'q') and (event.hasModifiers)):
            app.quit()
        elif ((event.key == 'x') and (event.hasModifiers)):
            os._exit(0) # hard exit avoids tkinter error messages
        elif (app.running and
              (not app.paused) and
              app.methodIsOverridden('keyPressed') and
              (not event.key == 'Modifier_Key')):
            app.keyPressed(event)
            app.redrawAllWrapper()

    @safeMethod
    def keyReleasedWrapper(app, event):
        if (not app.running) or app.paused or (not app.methodIsOverridden('keyReleased')): return
        event = App.KeyEventWrapper(event)
        if (not event.key == 'Modifier_Key'):
            app.keyReleased(event)
            app.redrawAllWrapper()

    @safeMethod
    def mousePressedWrapper(app, event):
        if (not app.running) or app.paused: return
        if ((event.x < 0) or (event.x > app.width) or
            (event.y < 0) or (event.y > app.height)):
            app.mousePressedOutsideWindow = True
        else:
            app.mousePressedOutsideWindow = False
            app._mouseIsPressed = True
            app._lastMousePosn = (event.x, event.y)
            if (app.methodIsOverridden('mousePressed')):
                app.mousePressed(event)
                app.redrawAllWrapper()

    @safeMethod
    def mouseReleasedWrapper(app, event):
        if (not app.running) or app.paused: return
        app._mouseIsPressed = False
        if app.mousePressedOutsideWindow:
            app.mousePressedOutsideWindow = False
            app.sizeChangedWrapper()
        else:
            app._lastMousePosn = (event.x, event.y)
            if (app.methodIsOverridden('mouseReleased')):
                app.mouseReleased(event)
                app.redrawAllWrapper()

    @safeMethod
    def timerFiredWrapper(app):
        if (not app.running) or (not app.methodIsOverridden('timerFired')): return
        if (not app.paused):
            app.timerFired()
            app.redrawAllWrapper()
        app.deferredMethodCall(afterId='timerFiredWrapper', afterDelay=app.timerDelay, afterFn=app.timerFiredWrapper)

    @safeMethod
    def sizeChangedWrapper(app, event=None):
        if (not app.running): return
        if (event and ((event.width < 2) or (event.height < 2))): return
        if (app.mousePressedOutsideWindow): return
        app.width,app.height,app.winx,app.winy = [int(v) for v in app._root.winfo_geometry().replace('x','+').split('+')]
        if (app._lastWindowDims is None):
            app._lastWindowDims = (app.width, app.height, app.winx, app.winy)
        else:
            newDims =(app.width, app.height, app.winx, app.winy)
            if (app._lastWindowDims != newDims):
                app._lastWindowDims = newDims
                app.updateTitle()
                app.sizeChanged()
                app.deferredRedrawAll() # avoid resize crashing on some platforms

    @safeMethod
    def mouseMotionWrapper(app):
        if (not app.running): return
        mouseMovedExists = app.methodIsOverridden('mouseMoved')
        mouseDraggedExists = app.methodIsOverridden('mouseDragged')
        if ((not app.paused) and
            (not app.mousePressedOutsideWindow) and
            (((not app._mouseIsPressed) and mouseMovedExists) or
             (app._mouseIsPressed and mouseDraggedExists))):
            class MouseMotionEvent(object): pass
            event = MouseMotionEvent()
            root = app._root
            event.x = root.winfo_pointerx() - root.winfo_rootx()
            event.y = root.winfo_pointery() - root.winfo_rooty()
            if ((app._lastMousePosn !=  (event.x, event.y)) and
                (event.x >= 0) and (event.x <= app.width) and
                (event.y >= 0) and (event.y <= app.height)):
                if (app._mouseIsPressed): app.mouseDragged(event)
                else: app.mouseMoved(event)
                app._lastMousePosn = (event.x, event.y)
                app.redrawAllWrapper()
        if (mouseMovedExists or mouseDraggedExists):
            app.deferredMethodCall(afterId='mouseMotionWrapper', afterDelay=app.mouseMovedDelay, afterFn=app.mouseMotionWrapper)

    def updateTitle(app):
        app.title = app.title or type(app).__name__
        app._root.title(f'{app.title} ({app.width} x {app.height})')

    def getQuitMessage(app):
        appLabel = type(app).__name__
        if (app.title != appLabel):
            if (app.title.startswith(appLabel)):
                appLabel = app.title
            else:
                appLabel += f" '{app.title}'"
        return f"*** Closing {appLabel}.  Bye! ***\n"

    def showRootWindow(app):
        root = app._root
        root.update(); root.deiconify(); root.lift(); root.focus()

    def hideRootWindow(app):
        root = app._root
        root.withdraw()

    @safeMethod
    def run(app):
        app._mouseIsPressed = False
        app._lastMousePosn = (-1, -1)
        app._lastWindowDims= None # set in sizeChangedWrapper
        app._afterIdMap = dict()
        # create the singleton root window
        if (App.theRoot is None):
            App.theRoot = Tk()
            App.theRoot.createcommand('exit', lambda: '') # when user enters cmd-q, ignore here (handled in keyPressed)
            App.theRoot.protocol('WM_DELETE_WINDOW', lambda: App.theRoot.app.quit()) # when user presses 'x' in title bar
            App.theRoot.bind("<Button-1>", lambda event: App.theRoot.app.mousePressedWrapper(event))
            App.theRoot.bind("<B1-ButtonRelease>", lambda event: App.theRoot.app.mouseReleasedWrapper(event))
            App.theRoot.bind("<KeyPress>", lambda event: App.theRoot.app.keyPressedWrapper(event))
            App.theRoot.bind("<Configure>", lambda event: App.theRoot.app.sizeChangedWrapper(event))
            try:
                # account for resolution on Windows (works fine without this on Macs)
                from ctypes import windll
                windll.user32.SetProcessDPIAware()
            except:
                pass
        else:
            App.theRoot.canvas.destroy()
        app._root = root = App.theRoot # singleton root!
        root.app = app
        root.geometry(f'{app.width}x{app.height}+{app.winx}+{app.winy}')
        app.updateTitle()
        # create the canvas
        root.canvas = app._canvas = WrappedCanvas(app)
        app._canvas.pack(fill=BOTH, expand=YES)
        # initialize, start the timer, and launch the app
        app.running = True
        app.paused = False
        app.appStartedWrapper()
        app.timerFiredWrapper()
        app.mouseMotionWrapper()
        app.showRootWindow()
        root.mainloop()
        app.hideRootWindow()
        app.running = False
        for afterId in app._afterIdMap: app._root.after_cancel(app._afterIdMap[afterId])
        app._afterIdMap.clear() # for safety
        app.appStopped()
        print(app.getQuitMessage())

####################################
# TopLevelApp:
# (with top-level functions not subclassses and methods)
####################################

class TopLevelApp(App):
    apps = dict() # maps fnPrefix to app

    def __init__(app, fnPrefix='', **kwargs):
        if (fnPrefix in TopLevelApp.apps):
            print(f'Quitting previous version of {fnPrefix} TopLevelApp.')
            TopLevelApp.apps[fnPrefix].quit()
        if ((fnPrefix != '') and ('title' not in kwargs)):
            kwargs['title'] = f"TopLevelApp '{fnPrefix}'"
        TopLevelApp.apps[fnPrefix] = app
        app.fnPrefix = fnPrefix
        app._callersGlobals = inspect.stack()[1][0].f_globals
        super().__init__(**kwargs)

    def callFn(app, fn, *args):
        fn = app.fnPrefix + fn
        if (fn in app._callersGlobals): app._callersGlobals[fn](*args)

    def redrawAll(app, canvas): app.callFn('redrawAll', app, canvas)
    def appStarted(app): app.callFn('appStarted', app)
    def appStopped(app): app.callFn('appStopped', app)
    def keyPressed(app, event): app.callFn('keyPressed', app, event)
    def keyReleased(app, event): app.callFn('keyReleased', app, event)
    def mousePressed(app, event): app.callFn('mousePressed', app, event)
    def mouseReleased(app, event): app.callFn('mouseReleased', app, event)
    def mouseMoved(app, event): app.callFn('mouseMoved', app, event)
    def mouseDragged(app, event): app.callFn('mouseDragged', app, event)
    def timerFired(app): app.callFn('timerFired', app)
    def sizeChanged(app): app.callFn('sizeChanged', app)

####################################
# ModalApp + Mode:
####################################

class ModalApp(App):
    def __init__(app, activeMode=None, **kwargs):
        app.running = False
        app.activeMode = None
        app.setActiveMode(activeMode)
        super().__init__(**kwargs)

    def setActiveMode(app, mode):
        if (not isinstance(mode, Mode)): raise Exception('activeMode must be a mode!')
        if (mode.app not in [None, app]): raise Exception('Modes cannot be added to two different apps!')
        if (app.activeMode != mode):
            mode.app = app
            if (app.activeMode != None): app.activeMode.modeDeactivated()
            app.activeMode = mode
            if (app.running): app.startActiveMode()

    def startActiveMode(app):
        app.activeMode.width, app.activeMode.height = app.width, app.height
        if (not app.activeMode._appStartedCalled):
            app.activeMode.appStarted() # called once per mode
            app.activeMode._appStartedCalled = True
        app.activeMode.modeActivated()  # called each time a mode is activated
        app.redrawAllWrapper()

    def redrawAll(app, canvas):
        if (app.activeMode != None): app.activeMode.redrawAll(canvas)
    def appStarted(app):
        if (app.activeMode != None): app.startActiveMode()
    def appStopped(app):
        if (app.activeMode != None): app.activeMode.modeDeactivated()
    def keyPressed(app, event):
        if (app.activeMode != None): app.activeMode.keyPressed(event)
    def keyReleased(app, event):
        if (app.activeMode != None): app.activeMode.keyReleased(event)
    def mousePressed(app, event):
        if (app.activeMode != None): app.activeMode.mousePressed(event)
    def mouseReleased(app, event):
        if (app.activeMode != None): app.activeMode.mouseReleased(event)
    def mouseMoved(app, event):
        if (app.activeMode != None): app.activeMode.mouseMoved(event)
    def mouseDragged(app, event):
        if (app.activeMode != None): app.activeMode.mouseDragged(event)
    def timerFired(app):
        if (app.activeMode != None): app.activeMode.timerFired()
    def sizeChanged(app):
        if (app.activeMode != None):
            app.activeMode.width, app.activeMode.height = app.width, app.height
            app.activeMode.sizeChanged()

class Mode(App):
    def __init__(mode, **kwargs):
        mode.app = None
        mode._appStartedCalled = False
        super().__init__(autorun=False, **kwargs)
    def modeActivated(mode): pass
    def modeDeactivated(mode): pass

####################################
# runApp()
####################################

'''
def showGraphics(drawFn, **kwargs):
    class GraphicsApp(App):
        def __init__(app, **kwargs):
            if ('title' not in kwargs):
                kwargs['title'] = drawFn.__name__
            super().__init__(**kwargs)
        def redrawAll(app, canvas):
            drawFn(app, canvas)
    app = GraphicsApp(**kwargs)
'''
runApp = TopLevelApp

if (__name__ == '__main__'):
    try: import cmu_112_graphics_tests
    except: pass
