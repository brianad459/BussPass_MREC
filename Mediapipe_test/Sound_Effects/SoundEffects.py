
from playsound import playsound
import os

DIR = os.path.dirname(__file__)

#https://pixabay.com/sound-effects/


def win():
   path = os.path.join(DIR, 'winning.mp3')
   playsound(path)
def lose():
    path = os.path.join(DIR, 'lose.mp3')
    playsound(path)

def points():
    path = os.path.join(DIR, 'collect-points-190037.mp3')
    playsound(path)

def tie():
    path = os.path.join(DIR, 'game-over-417465.mp3')
    playsound(path)

