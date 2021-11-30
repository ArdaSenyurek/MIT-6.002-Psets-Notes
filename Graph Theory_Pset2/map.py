import random 
from RandomWalks import Location
import math
class PolarLocation(Location):
    def __init__(self, r, theta = 0):
        self.radius = r
        self.angle = math.pi * theta / 180
        
    def getR(self):
        return self.radius
    
    def getAngle(self):
        return self.angle
    
    def getX(self):
        return self.radius * math.cos(self.angle)
    
    def getY(self):
        return self.radius * math.cos(math.pi/2 - self.angle)
    
    def distFrom(self, other):
        return other.getR() - self.radius
    
    def __str__(self):
        return 'x:' + str(self.getX()) + 'y:' + str(self.getY())
