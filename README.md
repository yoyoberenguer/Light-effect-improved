# Light-effect-improved
Light effect on pygame surface

![alt text](https://github.com/yoyoberenguer/light-effect-improved/blob/master/Screendump725.png)

![alt text](https://github.com/yoyoberenguer/light-effect-improved/blob/master/Screendump1180.png)

![alt text](https://github.com/yoyoberenguer/light-effect-improved/blob/master/Volume.gif)

## PROJECT:
```
This Library provide a selection of fast CYTHON methods designed to create realistic
light effects on PYGAME surface (SDL surface).
All light's algorithms are derived from my original project (LightEffect, per-pixel light effects in
a 2d environment) that you can find using the following link: https://github.com/yoyoberenguer/LightEffect
Those algorithms have been massively improved using CYTHON and multiprocessing techniques.

The light effect algorithms are compatible for both 24, 32-bit format textures (PNG file are recomanded
for both background image and radial alpha mask).
You will find different light effect techniques that does the same job for example
area24 (array) and area24b (buffer). The main difference reside in the type of array passed to 
the function (either buffer or numpy array).
```
## TECHNIQUE:

![alt text](https://github.com/yoyoberenguer/Light-effect-improved/blob/master/LIGHT.PNG)

```
The technique behind the scene is very simple:

1) A portion of the screen is taken (corresponding to the light's radial mask size),
   often an RGB block of pixels under the light source.
2) Then we are applying changes to the RGB block using the pre-defined settings such
   as light coloration, light intensity value and other techniques explain below
   (smoothing, saturation, bloom effect, heat wave convection effect)
3) The final array is build from both portions (RGB block and Alpha block, process also called STACKING)
   in order to provide an array shape (w, h, 4) to be converted to a pygame surface with
   pygame.image.frombuffer method.
4) The resulting image is blit onto the background with the additive mode (blending mode) using pygame
   special flag BLEND_RGBA_ADD
   Note: For 32-bit surface, additive mode is not required as the surface contains per-pixel alpha
   transparency channel.Blending it to the background will create a rectangular surface shape and alter
   the alpha channel (creating an undesirable rendering effect).
```

## EFFECTS
```
Some effect can be added to the light source to increase realistic rendering and or to alter
light source effect.
Here is the list of available effect you can use:
- Smooth     : Smooth the final light source effect with a GAUSSIAN BLUR, kernel 5x5 in real time
- Saturation : Create a saturation effect. You can set a variable between [-1.00 ... 1.0] to adjust the
               saturation level. Below zero, the light source turns slowly to a greyscale and above zero,
               the RGB block will have saturated pixels. Default value is 0.2 and create a moderate saturation effect.
               The saturation effect is achieve using HSL algorithm that I have also attached to this project.
               HSL algorithms are build in C language (external references with rgb_to_hsl and hsl_to_rgb.
               Both techniques are using C pointer and allocate memory blocks for each function calls
               in order to return a tuple hue, saturation, lightness. This imply that each block of memory needs
               to be freed after each function call. This is done automatically but be aware of that particularity if
               you are using HSL algorithms in a different project.
               see https://github.com/yoyoberenguer/HSL for more details.
- Bloom      : Bloom effect is a computer graphics effect used in video games, demos, and high dynamic range
               rendering to reproduce an imaging artifact of real-world cameras.
               In our scenario, the bloom effect will enhance the light effect when the light source
               is pointed toward another bright area / light spot etc. It use a bright pass filter (that can be
               adjusted with the variable threshold default 0). Threshold determine if a pixel will be included in the
               bloom process. The highest the threshold the fewer pixel will be included into the bloom process.
               See https://github.com/yoyoberenguer/BLOOM for more details concerning the bloom method.
- Heat       : Heat wave effect or convection effect. This algorithm create an illusion of hot air circulating in
               the light source. A mask is used to determine the condition allowing the pixels distortion.

If none of the above methods are used, a classic light source rendering effect is returned using only
coloration and light intensity parameters.
```
## REQUIREMENT:
```
- python > 3.0
- numpy arrays
- pygame with SDL version 1.2 (SDL version 2 untested)
  Cython
- A compiler such visual studio, MSVC, CGYWIN setup correctly
  on your system.
  - a C compiler for windows (Visual Studio, MinGW etc) install on your system 
  and linked to your windows environment.
  Note that some adjustment might be needed once a compiler is install on your system, 
  refer to external documentation or tutorial in order to setup this process.
  e.g https://devblogs.microsoft.com/python/unable-to-find-vcvarsall-bat/
```
## MULTI - PROCESSING CAPABILITY
```
The flag OPENMP can be changed any time if you wish to use multiprocessing
or not (default True, using multi-processing).
Also you can change the number of threads needed with the flag THREAD_NUMBER (default 10 threads)
```
## BUILDING PROJECT:
```
In a command prompt and under the directory containing the source files
C:\>python setup_lights.py build_ext --inplace
C:\>python setup_Saturation.py build_ext --inplace

If the compilation fail, refers to the requirement section and make sure cython 
and a C-compiler are correctly install on your system. 
```
## DEMO
```
Edit the file test_lights.py in your favorite python IDE and run it 

```

## HOW TO: 

The below source code is also in the python file pylight.py

Run pylight.py with your favorite python IDE 

```
import LIGHTS
from LIGHTS import area24

screen = pygame.display.set_mode((800, 600))

background = pygame.image.load('A1.png').convert()
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

lw, lh = light.get_size()
lw2, lh2 = lw >> 1, lh >> 1

# EFFECTS ARE NON CUMULATIVE 
lit_surface, sw, sh = area24(
   MOUSE_POS[0],                       # Mouse position x 
   MOUSE_POS[1],                       # Mouse position y
   background_rgb,                     # Background image transform into a numpy.ndarray (w, h, 3)
   lalpha,                             # Radial mask image transform into a numpy.ndarray (w, h)
   intensity=8.0,                      # Light intensity > 0  
   color=c,                            # Light coloration (numpy.array) 
                                       # e.g numpy.array([128.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0], 
                                       float32, copy=False)
   smooth=False,                       # boolean, True smooth the output image with GAUSSIAN blur 5x5 
   saturation=False, sat_value=0.2,    # boolean, True saturate the output image (use HSL algorithm), sat_value is
                                       # default 20% (moderate saturation)
   bloom=False,                        # boolean, Bloom effect to the final output image
   heat=False, frequency=index)        # boolean, Heat effect on the output image, frequency is an incrementing 
                                       # variable (default 0.5 each frame)
  # sw and sh are equivalent to the 
  # light effect width and height (returned by area24)
  # lw and lh are the radial mask width and height
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
```
      
