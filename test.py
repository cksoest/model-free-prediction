import random

sampleList = [100, 200, 300, 400, 500]

randomList = random.choices(sampleList, weights=[10, 20, 30, 40, 50], k=1)[0]

print(randomList)

