from pyboy import PyBoy

pyboy = PyBoy("Legend of Zelda, The - Link's Awakening DX (USA, Europe) (SGB Enhanced).gbc")
while pyboy.tick():
    pass
pyboy.stop()