import random

class Bird:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename

def readPNGs():
    birds = []
    for i in ["robin", "sparrow", "starling", "chickadee", "dove", "finch", "flicker", "magpie"]:
        for j in range(1, 11):
            filename = "./Birdsong images/" + i + "/" + i + "_" + str(j)
            bird = Bird(filename, i)
            birds.append(birds)
    return birds

def divideBirds(proportion):
    birds = readPNGs()
    random.shuffle(birds)
    index = int(len(birds) * proportion)
    trainingData = birds[:index]
    testingData = birds[index:]
    return trainingData, testingData