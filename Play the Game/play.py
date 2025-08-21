from pyboy import PyBoy

pyboy = PyBoy("Play the Game\Legend of Zelda, The - Link's Awakening DX (USA, Europe) (SGB Enhanced).gbc")
while pyboy.tick():
    pass
pyboy.stop()