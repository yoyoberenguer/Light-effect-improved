
"""
MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# Radial7.png
# https://steamcommunity.com/sharedfiles/filedetails/?id=465405802
# Radial9.png
# https://developer.valvesoftware.com/w/images/6/6f/Cookie_tutorial_texture_1.png
# Radial10.png
# https://forums.unrealengine.com/development-discussion/rendering/90292-horror-game-flashlight-problem
# https://leomoon.com/store/shaders/ies-lights-pack/

# NUMPY IS REQUIRED
try:
    import numpy
    from numpy import ndarray, zeros, empty, uint8, int32, float64, float32, dstack, full, ones, \
        asarray, ascontiguousarray, array
except ImportError:
    print("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")
    raise SystemExit

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d
    from pygame.image import frombuffer

except ImportError:
    print("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")
    raise SystemExit

import LIGHTS
from LIGHTS import area24


if __name__ == '__main__':
    screen = pygame.display.set_mode((800, 600))

    background = pygame.image.load('Brick.jpg').convert()
    background.set_alpha(None)
    background = pygame.transform.smoothscale(background, (800, 600))
    background_rgb = pygame.surfarray.pixels3d(background)
    w, h = background.get_size()
    back = background.copy()
    back.set_alpha(80)

    light = pygame.image.load('Radial8.png').convert_alpha()
    light = pygame.transform.smoothscale(light, (400, 400))
    lw, lh = light.get_size()
    lw2, lh2 = lw >> 1, lh >> 1
    lrgb = pygame.surfarray.pixels3d(light)
    lalpha = pygame.surfarray.pixels_alpha(light)

    pygame.mouse.set_visible(False)
    c = numpy.array([128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0], float32, copy=False)
    MOUSE_POS = [0, 0]

    index = 0
    i = 0
    while 1:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                MOUSE_POS = event.pos

        if keys[pygame.K_F8]:
            pygame.image.save(screen, 'Screendump' + str(i) + '.png')
        if keys[pygame.K_ESCAPE]:
            break

        screen.fill((0, 0, 0))
        screen.blit(back, (0, 0))

        lit_surface, sw, sh = area24(
            MOUSE_POS[0], MOUSE_POS[1], background_rgb, lalpha, intensity=8.0, color=c,
            smooth=False, saturation=False, sat_value=0.2, bloom=False, heat=False, frequency=index)

        if sw < lw and MOUSE_POS[0] <= w - lw2:
            xx = 0
        else:
            xx = MOUSE_POS[0] - lw2

        if sh < lh and MOUSE_POS[1] <= lh - lh2:
            yy = 0
        else:
            yy = MOUSE_POS[1] - lh2

        screen.blit(lit_surface, (xx, yy), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.display.flip()

        index += 0.5

        i += 1