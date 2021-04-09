import librosa

arr = ["chickadee", "dove", "finch", "flicker", "magpie", "robin", "sparrow", "starling"]

birdsongs = {}

for i in arr:
    bird = []
    for j in range(1, 11):
        file_path = "./" + i + "/" + i + "_" + str(j)
        oneBird = samples, sampling_rate = librosa.load(file_path, sr = None, mono = True,
                                        offset = 0.0, duration = None)
        bird.append(oneBird)
    birdsongs[i] = bird


