import pyttsx3

engine = pyttsx3.init("sapi5")
voice =  engine.getProperty("voices")
engine.setProperty("voice", voice[1].id)

def startGame (msg):
    engine.say(msg)
    engine.runAndWait()
#startGame("This is the game of Rock, Paper, and scissors ")

def HowToPlay(msg):
    engine.say(msg)
    engine.runAndWait()
#HowToPlay(" 1. Start by gesturing 'game' in ASL, fistbump with thumbs pointed upwards"
          #"2. You will then through your next gesture of either rock, paper, or scissors"
          #"2. When I'm deciding my next move through your next gesture")


def UserLost1(msg):
    engine.say(msg)
    engine.runAndWait()
#UserLost1("Haha You lost to AI")


def UserLost2(msg):
    engine.say(msg)
    engine.runAndWait()
#UserLost2("I was made in two months and you still couldn't beat me")

def computerLost1(msg):
    engine.say(msg)
    engine.runAndWait()
#computerLost1("You can have that one")

def computerLost2(msg):
    engine.say(msg)
    engine.runAndWait()
#computerLost2("You just got lucky")

def tie(msg):
    engine.say(msg)
    engine.runAndWait()
#tie("ggs")

def startPlayingAgain(msg):
    engine.say(msg)
    engine.runAndWait()
#startPlayingAgain("You can choose to play again by")

def startingGame_Issues (msg):
    engine.say(msg)
    engine.runAndWait()
#startingGame_Issues("If your move isn't detected try different angles")

def greetingMessage(msg):
    engine.say(msg)
    engine.runAndWait()
#greetingMessage("Hey, would you like to play a quick game of rock-paper-scissors?")

