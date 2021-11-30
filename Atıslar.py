import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid.axislines import SubplotZero
class Kinematics(object):
    g = 9.81    
    
    def __init__(self, angle, Y0 = 0, V_x = None, V_y = None):
        self.angle = angle
        self.Y0 = Y0
        self.time = []
        #self.Vinit = V_init
        self.Vx = V_x
        self.Vy = V_y
        
    def Shooting(self, object, t_Target, t, tDomain, Vx, Vy, Vy_initial, XDistance, YDistance, angle = None):
        
        self.Locs = {}
        angle = self.angle
        # Vx = V * np.cos((angle/360)*2*np.pi)
        # Vy = V * np.sin((angle/360)*2*np.pi)
        if t != t_Target:
            print('---------------------------------------')
            print('Vy', Vy)
            print('Vx', Vx)
            print('Vyinit', Vy_initial)
            print('YDistance', YDistance,'\t' , 'XDistance', XDistance)
            print('---------------------------------')
            Vy = Vy - Kinematics.g * tDomain
            print('ADFSDFADvy', Vy)
            print('AAAA','YDistance', YDistance,'\t' , 'XDistance', XDistance, '\t' ,  't', t)
            t += tDomain
            self.Vertical.append((Vy_initial * t - (Kinematics.g * t**2) / 2))
            self.horizontal.append(Vx * t)
            self.Locs[t] = location(Vx * t, Vy_initial * t - (Kinematics.g * t**2) / 2)
            return self.Shooting(t_Target, t, tDomain, Vx, Vy, Vy_initial, XDistance, YDistance)
            # XDistance = XDistance + Vx * tDomain
        
        else:
            print('girdim')
            return None
    
    def getTime(self):
        return self.time
        
    def UpdateTime(self,t):
        self.time.append(t)
    

class Objects(object):

    def __init__(self, mass, MomentOfInertia, Force = 0): 
        self.mass = mass
        self.force = Force
        self.MofInertia = MomentOfInertia
        # {'Object1': locs, 'Object2':, locs}
    
    def getMass(self):
        return self.mass

    def getForce(self):
        return self.force

    def getMofIntertia(self):
        return self.MofInertia

    def __str__(self):
        return f'Mass = {self.mass} m/s \t Initial Force = {self.force} kg.m/s^2'


class location(object):
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y 

    def move(self, deltaX, deltaY):
        return location(self.x + deltaX, self.y + deltaY)
    
    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getYDistance(self, other):
        return self.getY() - other.getY()
    
    def getXDistance(self, other):
        return self.getX() - other.getX()

    def getDistance(self, other):
        XDistance = self.x - other.getX()
        YDistance = self.y - other.getY()
        return (XDistance**2 + YDistance**2)**(1/2)
    
    def setLoc(self, x, y):
        return location(x ,y)

    def __str__(self):
        return f'X:{self.x} \t Y:{self.y}'
    
    
class Field(object):
    def __init__(self, x = 0, y = 0):
        self.objects = {}  #  {'X': loc, 'Y': loc}   Key = object.getName() ,  value = loc object
        
    def addObject(self, object, loc):
        if object not in self.objects:
            self.objects[object] = loc
        else: 
            raise KeyError('there already exists an object you are trying to add.')
    def MoveObject(self, object):
        if object not in self.objects:
            raise KeyError('Not in Field')
        else:
            self.objects[object] = self.objects[object].move(Kinematics.getX(), Kinematics.getY()) 

    def MoveStep(self, object):
        deltaX, deltaY = Kinematics.getXDiff(), Kinematics.getYDiff

    def Shooting(self, object, t, tDomain, V, angle):
        Locs = {}
        def ShootingCore(self, object, t_Target, t, tDomain, Vy_initial, Vx, V, angle):
            print('---------------------------------------')
            print('Vx', Vx)
            print('t_target', t_Target)
            print( 't', t)
            print('object', object)
            print('tdomain',tDomain)
            print('vy_initial',Vy_initial)
            print('v', V)
            print('angle',angle)
            print('---------------------------------')
            Vy = V * np.sin(angle)
            print('Vy', Vy)
            if t - t_Target <= tDomain:
                Vy = Vy - Kinematics.g * tDomain
                t += tDomain
                # self.Vertical.append((Vy_initial * t - (Kinematics.g * t**2) / 2))
                # self.horizontal.append(Vx * t)
                Locs[t] = location(Vx * t, Vy_initial * t - (Kinematics.g * t**2) / 2)
                print('Vx', Vx)
                print('vy', Vy)
                return ShootingCore(self, object, t_Target, t, tDomain, Vy_initial, Vx ,(Vx**2+Vy**2)**(1/2), np.arctan2(Vy, Vx))
                # XDistance = XDistance + Vx * tDomain
            
            else:
                print('girdim')
                return Locs
        return ShootingCore(self,object, t, 0, tDomain, V * np.sin((angle/360) * 2 * np.pi),V * np.cos((angle/360) * 2 * np.pi) , V, angle)
    
    def __str__(self):
        return str(self.objects)

# a = {2: {'a': 's'}, '3': {1: '3,5', 2: '5,6'}}

# a[2].update({'b': 'c'})
# a



def simShooting(V, t, angle):

    if 1:
        fig = plt.figure(1)
        ax = SubplotZero(fig, 111)
        fig.add_subplot(ax)

        for direction in ["xzero", "yzero"]:
            ax.axis[direction].set_axisline_style("-|>")
            ax.axis[direction].set_visible(True)

        for direction in ["left", "right", "bottom", "top"]:
            ax.axis[direction].set_visible(False)

        X,Y = [], []
        f = Field()
        f.addObject('basdc', 1 )
        f.addObject('acasdcads', 2)
        for loc in f.Shooting('basdc', t= 15, tDomain= 0.05, V= 100, angle = 45).values():
            print(loc)
            X.append(loc.getX())
            Y.append(loc.getY())
        print(X, Y)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.scatter(X,Y)
        for i, value in enumerate(X):
            ax.annotate(i, (X[i], Y[i]))
    plt.show()   
simShooting(10, 10, 0)
# l1 = location()
# l2 = location()

# print(l1.getXDistance(l2))
# print(l1.getYDistance(l2))
# print(l1.getDistance(l2))
# print(l1.move(2,3))
