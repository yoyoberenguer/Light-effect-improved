
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

import timeit

import LIGHTS
from LIGHTS import area24, area32, area24b, \
    area32b, apply32b, stack_buffer, area24bb, light_volumetric, light_volume,\
    flatten2d, flatten3d_rgb, flatten3d_rgba, stack_object



import random

if __name__ == '__main__':
    screen = pygame.display.set_mode((400, 400))

    background = pygame.image.load('Aliens.jpg').convert()
    background.set_alpha(None)

    # CONVERT BACKGROUND INTO AN ARRAY
    # RESIZE PICTURE TO THE SCREEN SIZE
    background = pygame.transform.smoothscale(background, (400, 400))
    background_rgb = pygame.surfarray.pixels3d(background)
    w, h = background.get_size()

    # CONVERT BACKGROUND INTO A BUFFER
    background_buffer = numpy.asarray(background.get_view('3'), numpy.uint8).transpose(1, 0, 2)
    background_buffer = background_buffer.flatten(order='C')
    back = background.copy()
    back.set_alpha(80)

    # TEXTURE
    smoke = pygame.image.load('smoke1.png').convert()
    sw, sh = smoke.get_size()
    smokes = []
    a = 0
    for rows in range(8):
        for columns in range(8):
            new_surface = Surface((256, 256), flags=RLEACCEL)
            new_surface.blit(smoke, (0, 0), (columns * 256, rows * 256, 256, 256))
            new_surface = pygame.transform.smoothscale(new_surface, (400, 400))
            array_ = pygame.surfarray.pixels3d(new_surface).transpose(1, 0, 2)
            smokes.append(array_)
            a += 1

    # LOAD THE LIGHT RADIAL SHAPE
    # IMAGE MUST BE PNG FORMAT AND CONTAINS ALPHA TRANSPARENCY LAYER
    light = pygame.image.load('Radial8.png').convert_alpha()
    light = pygame.transform.smoothscale(light, (400, 400))
    lw, lh = light.get_size()
    lw2, lh2 = lw >> 1, lh >> 1

    # TRANSFORM RADIAL IMAGE INTO ARRAY
    lrgb = pygame.surfarray.pixels3d(light)
    lalpha = pygame.surfarray.pixels_alpha(light)

    # TRANSFORM RADIAL IMAGE INTO BUFFER
    lalpha_buffer = lalpha.transpose(1, 0)
    lalpha_buffer = lalpha_buffer.flatten(order='C')

    # TRANSFORM RADIAL ARRAY (RGB + ALPHA) INTO RGBA
    lrgba_buffer = stack_buffer(background_buffer, lalpha_buffer, lw, lh, False)


    N = 1

    c = numpy.array([128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0], float32, copy=False)
    MOUSE_POS = [0, 0]

    t = timeit.timeit("area24(0, 1, background_rgb, lalpha, intensity=5.0, color=c)",
                      "from __main__ import area24, background_rgb, lalpha, c", number=N)
    print("area24 None : ", t, t / N)
    t = timeit.timeit("area24(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=True)",
                      "from __main__ import area24, background_rgb, lalpha, c", number=N)
    print("area24 BLUR: ", t, t / N)
    t = timeit.timeit("area24(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=False, saturation=True)",
                      "from __main__ import area24, background_rgb, lalpha, c", number=N)
    print("area24 SATURATION : ", t, t / N)
    t = timeit.timeit("area24(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=False, "
                      "saturation=False, sat_value=0.2, bloom=True)",
                      "from __main__ import area24, background_rgb, lalpha, c", number=N)
    print("area24 BLOOM : ", t, t / N)

    t = timeit.timeit("area32(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=False, "
                      "saturation=False, sat_value=0.2, bloom=False)",
                      "from __main__ import area32, background_rgb, lalpha, c", number=N)
    print("\narea32 None : ", t, t / N)

    t = timeit.timeit("area32(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=True, "
                      "saturation=False, sat_value=0.2, bloom=False)",
                      "from __main__ import area32, background_rgb, lalpha, c", number=N)
    print("area32 BLUR : ", t, t / N)

    t = timeit.timeit("area32(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=False, "
                      "saturation=True, sat_value=0.2, bloom=False)",
                      "from __main__ import area32, background_rgb, lalpha, c", number=N)
    print("area32 SATURATION : ", t, t / N)

    t = timeit.timeit("area32(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=False, "
                      "saturation=False, sat_value=0.2, bloom=True)",
                      "from __main__ import area32, background_rgb, lalpha, c", number=N)
    print("area32 BLOOM : ", t, t / N)

    total_t = timeit.timeit("area24b(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=False, "
                            "saturation=False, sat_value=0.2, bloom=False)",
                            "from __main__ import background_rgb, c,"
                            "lalpha, area24b, array, float32",
                            number=N)
    print("\narea24b NONE ", total_t, total_t / N)

    total_t = timeit.timeit("area24b(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=True, "
                            "saturation=False, sat_value=0.2, bloom=False)",
                            "from __main__ import background_rgb, c,"
                            "lalpha, area24b, array, float32",
                            number=N)
    print("area24b BLUR ", total_t, total_t / N)

    total_t = timeit.timeit("area24b(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=False, "
                            "saturation=True, sat_value=0.2, bloom=False)",
                            "from __main__ import background_rgb, c,"
                            "lalpha, area24b, array, float32",
                            number=N)
    print("area24b SATURATION ", total_t, total_t / N)

    total_t = timeit.timeit("area24b(0, 1, background_rgb, lalpha, intensity=5.0, color=c, smooth=False, "
                            "saturation=False, sat_value=0.2, bloom=True)",
                            "from __main__ import background_rgb, c,"
                            "lalpha, area24b, array, float32",
                            number=N)
    print("area24b BLOOM ", total_t, total_t / N)

    total_t = timeit.timeit("area24bb(0, 1, background_buffer, w, h,"
                            "lalpha_buffer, lw, lh, c, 3.0, smooth=False, saturation=False, "
                            "sat_value=0.2, bloom=False)", "from __main__ import background_buffer, c,"
                                                           "lalpha_buffer, area24bb, lw, lh, c, w, h ",
                            number=N)
    print("\narea24bb None ", total_t, total_t / N)

    total_t = timeit.timeit("area24bb(0, 1, background_buffer, w, h,"
                            "lalpha_buffer, lw, lh, c, 3.0, smooth=True, saturation=False, "
                            "sat_value=0.2, bloom=False)", "from __main__ import background_buffer, c,"
                                                           "lalpha_buffer, area24bb, lw, lh, c, w, h ",
                            number=N)
    print("area24bb BLUR ", total_t, total_t / N)

    total_t = timeit.timeit("area24bb(0, 1, background_buffer, w, h,"
                            "lalpha_buffer, lw, lh, c, 3.0, smooth=False, saturation=True, "
                            "sat_value=0.2, bloom=False, heat=False, frequency=1.0)",
                            "from __main__ import background_buffer, c,"
                            "lalpha_buffer, area24bb, lw, lh, c, w, h ",
                            number=N)
    print("area24bb SATURATION ", total_t, total_t / N)

    total_t = timeit.timeit("area24bb(0, 1, background_buffer, w, h,"
                            "lalpha_buffer, lw, lh, c, 3.0, smooth=False, saturation=False, "
                            "sat_value=0.2, bloom=True, heat=False, frequency=1.0"
                            ")", "from __main__ import background_buffer, c,"
                            "lalpha_buffer, area24bb, lw, lh, c, w, h ",
                            number=N)
    print("area24bb BLOOM ", total_t, total_t / N)

    pygame.mouse.set_visible(False)

    intensity = 0.0
    index = 0
    red = 0
    i=0
    while 1:

        pygame.event.pump()
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                MOUSE_POS = event.pos

        if keys[pygame.K_F8]:
            pygame.image.save(screen, 'Screendump' + str(i) + '.png')

        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            break

        screen.fill((0, 0, 0))

        screen.blit(back, (0, 0))

        lit_surface, sw, sh = area24(
            MOUSE_POS[0], MOUSE_POS[1], background_rgb, lalpha, intensity=8.0, color=c,
            smooth=False, saturation=False, sat_value=0.2, bloom=False, heat=False, frequency=index)

        # lit_surface, sw, sh = area32(
        #     MOUSE_POS[0], MOUSE_POS[1], background_rgb, lalpha, intensity=8.0, color=c,
        #     smooth=False, saturation=False, sat_value=0.2, bloom=False, heat=True, frequency=index)

        # lit_surface, sw, sh = area24b(
        #     MOUSE_POS[0], MOUSE_POS[1], background_rgb,
        #     lalpha, c, 8.0, smooth=False, saturation=False,
        #     sat_value=0.2, bloom=True, heat=False, frequency=index)  # , array_=smokes[index])

        # c = numpy.array([128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0], float32, copy=False)
        # lit_surface, sw, sh = area24bb(
        #     MOUSE_POS[0], MOUSE_POS[1], background_buffer, w, h,
        #     lalpha_buffer, lw, lh, c, 8.0, smooth=False,
        #     saturation=False, sat_value=1.0, bloom=False, heat=True, frequency=index)

        # c = numpy.array([128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0], float32, copy=False)
        # lit_surface1, sw1, sh1 = area24bb(
        #     250, 150, background_buffer, w, h,
        #     lalpha_buffer, lw, lh, c, 5.0, smooth=False,
        #     saturation=False, sat_value=1.0, bloom=False, heat=False, frequency=index)

        # c = numpy.array([128.0 / 255.0, 201.0 / 255.0, 220.0 / 255.0], float32, copy=False)
        # lit_surface, sw, sh = light_volume(MOUSE_POS[0], MOUSE_POS[1],
        #                                    background_rgb, lalpha, 8.0, c, smokes[int(index)])
        # c = numpy.array([128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0], float32, copy=False)

        # lit_surface = apply32b(background_buffer.flatten(), lalpha_buffer, 1.0, c, lw, lh)
        # lit_surface = apply32bf(lrgba_buffer, 1.0, c, lw, lh)
        # lit_surface = LIGHTS.area24bb(MOUSE_POS[0], MOUSE_POS[1], background_buffer, 400, 400,
        #                              lalpha_buffer, 400, 400, c, 1.0, smooth=False, saturation=True, bloom=False)
        # lit_surface = area24b(MOUSE_POS[0], MOUSE_POS[1], background_rgb,
        #     lalpha, c, 2.0, smooth=False, saturation=False, bloom=True)
        # MOUSE_POS = max(MOUSE_POS[0], 0), max(MOUSE_POS[1], 0)

        if sw < lw and MOUSE_POS[0] <= w - lw2:
            xx = 0
        else:
            xx = MOUSE_POS[0] - lw2

        if sh < lh and MOUSE_POS[1] <= lh - lh2:
            yy = 0
        else:
            yy = MOUSE_POS[1] - lh2

        # if sw1 < lw and 250 <= w - lw2:
        #     xxx = 0
        # else:
        #     xxx = 250 - lw2
        #
        # if sh1 < lh and 150 <= lh - lh2:
        #     yyy = 0
        # else:
        #     yyy = 150 - lh2

        screen.blit(lit_surface, (xx, yy), special_flags=pygame.BLEND_RGBA_ADD)
        # screen.blit(lit_surface1, (xxx, yyy), special_flags=pygame.BLEND_RGBA_ADD)
        # screen.blit(pygame.surfarray.make_surface(numpy.asarray(smokes[index])), (0, 0))
        pygame.display.flip()

        intensity += 0.1
        if intensity > 3:
            intensity = 0

        index += 0.5
        if index > len(smokes)-1:
            index = 0
        red += 1
        if red > 255:
            red = 0

        i += 1