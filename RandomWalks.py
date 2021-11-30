import random
import matplotlib.pyplot as plt
from numpy import negative 

class Location(object):
    def __init__(self, x, y): 
        self.x = x
        self.y = y 

    def move(self, deltaX, deltaY):
        return Location(self.x + deltaX, self.y + deltaY)

    def getX(self):
        return self.x

    def getY(self):
        return self.y
    
    def distFrom(self, other):
        Xdistance = self.x - other.getX()
        Ydistance = self.y - other.getY()
        return (Xdistance**2 + Ydistance**2)**(1/2)
    
    def __str__(self):
        return f'< + {str(self.x)} ,  +{str(self.y)} >' 

    
class Field(object):
    def __init__(self):
        self.drunks = {}

    def addDrunks(self, drunk, loc):
        if drunk in self.drunks:
            raise ValueError('Already in Field')
        else:
            self.drunks[drunk] = loc 

    def moveDrunk(self, drunk):
        if drunk not in self.drunks:
            raise ValueError('Not in Field')
        else: 
            deltaX, deltaY = drunk.takeStep()
            self.drunks[drunk] = self.drunks[drunk].move(deltaX, deltaY) #self.drunks[drunk] was the location, updated via move() method    
    def getLoc(self, drunk):
        if drunk not in self.drunks:
            raise ValueError('Not in Field')
        else:
            return self.drunks[drunk]

    def FirstDrunk(self):
        listt = list(self.drunks.keys())[0]
        return listt
    
    def getDrunks(self):
        return self.drunks.keys()

class Drunk(object):
    def __init__(self, name = None):
        self.name = name 

    def __str__(self):
        if self.name != None:
            return f'{self.name}'
        return 'Anonymous'

class UsualDrunk(Drunk):
    def takeStep(self):
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        return random.choice(directions)
    def getname(self):
        return self.name

class ColdDrunk(Drunk):
    def takeStep(self):
        direction = [(0.0 , 1.0), (0.0, -2.0), (1.0 , 0.0), (-1.0 , 0.0)]
        return random.choice(direction)

class EWDrunk(Drunk):
    def takeStep(self):
        direction = [(1.0 , 0.0), (-1.0 , 0.0)]
        return random.choice(direction)


class StyleIterator(object):
    def __init__(self, styles):
        self.index = 0
        self.styles = styles
    
    def NextStyle(self):
        result = self.styles[self.index]
        if self.index == len(self.styles) - 1:
            self.index = 0
        else:
            self.index += 1
            
        return result

def walk(f, d, numSteps):
    """
    f = field, d = drunk, numSteps: int >= 0 
    Moves d numSteps times; 
    returns the distance between the final loc and the loc at the start of the walk.
    """
    startLoc = f.getLoc(d)
    for times in range(numSteps):
        f.moveDrunk(d)
    return startLoc.distFrom(f.getLoc(d))

def simWalk(numSteps, numTrials, dClass):
    """
    numSteps: int >= 0, numTrials: int > 0, dClass: a subclass of Drunk 
    returns a list of final distances for each trial 
    """
    YuumiPlayer = dClass('E-Girl')
    origin = Location(0,0)
    distances = []
    for times in range(numTrials):
        f = Field()
        f.addDrunks(YuumiPlayer, origin)
        distances.append(round(walk(f, YuumiPlayer, numSteps), 1))
    return distances

def test(walklengths, drunkkinds, numtrials):
    for x in drunkkinds:
        drunkTest(walklengths, numtrials, x)
def drunkTest(walkLengths, numtrials, dClass):
    """
    walkLengths: sequence of ints >= 0, numTrials: int > 0, dClass: a subclass of Drunk
    For each number of steps in walkLengts, runs simWalks with numtrials walks and prints the result.
    """
    means = []
    for numSteps in walkLengths:
        distances = simWalk(numSteps, numtrials, dClass)
        means.append(sum(distances)/len(distances))
        print(dClass.__name__, 'random walk of', numSteps, 'steps')
        print(f'Mean:{round(sum(distances)/len(distances), 4)}')
        print(f'Max = {max(distances)}, Min = {min(distances)}')
    return means

#test((100,1000), (UsualDrunk, ColdDrunk, EWDrunk), 10)
def simDrunk(numTrials, dClass, walkLengths):
    meanDistances  = []
    for numsteps in walkLengths:
        trials = simWalk(numsteps, numTrials, dClass)
        mean = sum(trials)/ len(trials)
        meanDistances.append(mean)

    return meanDistances

def simAll1(drunkKinds, walkLenghts, numTrials):
    styleChoice = StyleIterator(('m-', 'r:', 'k-'))
    for drunk in drunkKinds:
        nextStyle = styleChoice.NextStyle()
        print('Starting simulation of ', drunk.__name__)
        means = simDrunk(numTrials, drunk, walkLenghts)
        plt.plot(walkLenghts, means, nextStyle, label = drunk.__name__)
        plt.title(f'Mean distances from the origin ({numTrials}) trials')
        plt.xlabel('Number of steps')
        plt.ylabel('Distance from the origin')
        plt.legend(loc = 'best')
        plt.semilogx()
        plt.semilogy()
    plt.show()
#simAll1((UsualDrunk, EWDrunk, ColdDrunk), (10,100,1000,10000, 100000), 100)

def getLocs(numSteps, numTrials, dClass):
    finalLocs = []
    d = dClass()
    for times in range(numTrials):
        f = Field()
        f.addDrunks(d, Location(0,0))
        for step in range(numSteps):
            f.moveDrunk(d)
        finalLocs.append(f.getLoc(d))
    return finalLocs

def PlotLocs(drunkKinds, numSteps, numTrials):
    Style = StyleIterator(('k+', 'r^', 'mo'))
    #globalX = []
    #globalY = []
    for kind in drunkKinds:
        locs = getLocs(numSteps, numTrials, kind)
        XVal, YVal = [], []
        for loc in locs:
            XVal.append(loc.getX())
            YVal.append(loc.getY())
        meanX = sum(XVal) / len(XVal)
        meanY = sum(YVal) / len(YVal)
        curStyle = Style.NextStyle()
        plt.plot(XVal, YVal, curStyle, label = f'{kind.__name__} mean loc = <{meanX},{meanY}>')
        # globalX.append(min(XVal))
        # globalX.append(max(XVal))
        # globalY.append(min(YVal))
        # globalY.append(max(YVal))
    
    plt.title(f'End Locations ({numSteps}) Steps, ({numTrials}) Trials')
    plt.legend(loc = 'upper left', prop = {'size': 8})
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')
    plt.xlim()
    plt.ylim()
    plt.show()
#PlotLocs((UsualDrunk, ColdDrunk, EWDrunk), 100, 200)
plt.plot([15,30, 1], [20,25,2], 'k-')
plt.plot([70,60], [80,60], 'k-')
#plt.xlim(0, 30)
#plt.ylim(0,50)
plt.show()
#---------------------------------------------------------
