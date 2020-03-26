
###cython: boundscheck=False, wraparound=False, nonecheck=False, optimize.use_switch=True

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

# PROJECT:
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

# TECHNIQUE:
The technique behind the scene is very simple:

1) A portion of the screen is taken (corresponding to the size of the light's radial mask),
   often an RGB block of pixels under the light source.
2) Then applying changes to the RGB block using the pre-defined settings such
   as light coloration, light intensity value and other techniques that will be explain below,
   (smoothing, saturation, bloom effect, heat wave convection effect)
3) The final array is build from both portions (RGB block and Alpha block, process also called STACKING)
   in order to provide an array shape (w, h, 4) to be converted to a pygame surface with
   pygame.image.frombuffer method.
4) The resulting image is blit onto the background with the additive mode (blending mode) using pygame
   special flag BLEND_RGBA_ADD
   Note: For 32-bit surface, additive mode is not required as the surface contains per-pixel alpha
   transparency channel. Blending it to the background will create a rectangular surface shape and alter
   the alpha channel (creating an undesirable rendering effect).

# EFFECTS
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

REQUIREMENT:
- python > 3.0
- numpy arrays
- pygame with SDL version 1.2 (SDL version 2 untested)
  Cython
- A compiler such visual studio, MSVC, CGYWIN setup correctly
  on your system

# MULTI - PROCESSING CAPABILITY
The flag OPENMP can be changed any time if you wish to use multiprocessing
or not (default True, using multi-processing).
Also you can change the number of threads needed with the flag THREAD_NUMBER (default 10 threads)

BUILDING PROJECT:
Use the following command:
C:\>python setup_lights.py build_ext --inplace


"""

# NUMPY IS REQUIRED
try:
    import numpy
    from numpy import ndarray, zeros, empty, uint8, int32, float64, float32, dstack, full, ones,\
    asarray, ascontiguousarray
except ImportError:
    print("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")
    raise SystemExit

# CYTHON IS REQUIRED
try:
    cimport cython
    from cython.parallel cimport prange
except ImportError:
    print("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")
    raise SystemExit

cimport numpy as np


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

cimport numpy as np
from libc.stdio cimport printf
from libc.stdlib cimport free, rand
from libc.math cimport round, fmin, fmax, sin


cdef extern from 'library.c' nogil:
    double * rgb_to_hsl(double r, double g, double b);
    double * hsl_to_rgb(double h, double s, double l);

# C-structure to store 3d array index values
cdef struct xyz:
    int x;
    int y;
    int z;


DEF OPENMP = True

if OPENMP == True:
    DEF THREAD_NUMBER = 10
else:
    DEF THREAD_NUMNER = 1

DEF SCHEDULE = 'static'


DEF HALF = 1.0/2.0
DEF ONE_THIRD = 1.0/3.0
DEF ONE_FOURTH = 1.0/4.0
DEF ONE_FIFTH = 1.0/5.0
DEF ONE_SIXTH = 1.0/6.0
DEF ONE_SEVENTH = 1.0/7.0
DEF ONE_HEIGHT = 1.0/8.0
DEF ONE_NINTH = 1.0/9.0
DEF ONE_TENTH = 1.0/10.0
DEF ONE_ELEVENTH = 1.0/11.0
DEF ONE_TWELVE = 1.0/12.0
DEF ONE_255 = 1.0/255.0
DEF ONE_360 = 1.0/360.0
DEF TWO_THIRD = 2.0/3.0


# ----------------- INTERFACE --------------------

# ------------------------------------------------
#                     ARRAY 
# CREATE REALISTIC LIGHT EFFECT ON 24-BIT SURFACE.
# DETERMINE THE PORTION OF THE SCREEN EXPOSED TO THE LIGHT SOURCE.
# COMPATIBLE WITH 24 BIT ONLY
def area24(x, y, background_rgb, mask_alpha,
           intensity, color=numpy.array([128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0],
           numpy.float32, copy=False), smooth=False, saturation=False,
           sat_value=0.2, bloom=False, heat=False, frequency=1.0):
    return area24_c(x, y, background_rgb, mask_alpha, intensity, color,
                    smooth, saturation, sat_value, bloom, heat, frequency)

# CREATE REALISTIC LIGHT EFFECT ON 32-BIT SURFACE.
# COMPATIBLE WITH 32-BIT ONLY
def area32(x, y, background_rgb, mask_alpha,
           intensity, color=numpy.array([128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0],
           numpy.float32, copy=False), smooth=False, saturation=False,
           sat_value=0.2, bloom=False, heat=False, frequency=1.0):
    return area32_c(x, y, background_rgb, mask_alpha, intensity, color,
                    smooth, saturation, sat_value, bloom, heat, frequency)

# ------------------ BUFFERS --------------------------

# NO USED
# Create a light effect on the given portion of the screen (compatible 32 bit)
def apply32b(rgb_buffer_, alpha_buffer_, intensity, color, w, h):
    return apply32b_c(rgb_buffer_, alpha_buffer_, intensity, color, w, h)


# ------------------------------------------------

# ------------------------------------------------
#                     BUFFER 
# CREATE REALISTIC LIGHT EFFECT ON 24-BIT SURFACE.
# DETERMINE THE PORTION OF THE SCREEN EXPOSED TO THE LIGHT SOURCE AND
# APPLY LIGHT EFFECT.COMPATIBLE 24 BIT ONLY
def area24b(x, y, background_rgb, mask_alpha,
                  color, intensity, smooth=False,
                  saturation=False, sat_value=0.2, bloom=False, heat=False, frequency=1.0, array_=None)->Surface:
    return area24b_c(x, y, background_rgb, mask_alpha,
                           color, intensity, smooth, saturation, sat_value, bloom, heat, frequency, array_)

# UNDER TEST (ONLY BUFFERS) NOT COMPLETED YET
def area24bb(x, y, background_rgb, w, h, mask_alpha, mw, mh,
                  color, intensity, smooth=False,
                  saturation=False, sat_value=0.2, bloom=False, heat=False, frequency=1.0)->Surface:
    return area24bb_c(x, y, background_rgb, w, h, mask_alpha, mw, mh,
                           color, intensity, smooth, saturation, sat_value, bloom, heat, frequency)

# CREATE REALISTIC LIGHT EFFECT ON 32-BIT SURFACE.
# DETERMINE THE PORTION OF THE SCREEN EXPOSED TO THE LIGHT SOURCE AND
# APPLY LIGHT EFFECT.COMPATIBLE 32 BIT ONLY
# FIXME: NOT FINALIZED
def area32b(x, y, background_rgb, mask_alpha,
                  color, intensity, smooth)->Surface:
    return area32b_c(x, y, background_rgb,
                           mask_alpha, color, intensity, smooth)

# ------------------------------------------------

# CREATE 2D LIGHT EFFECT WITH VOLUMETRIC EFFECT APPLY TO ALPHA CHANNEL
def light_volume(x, y, background_rgb,  mask_alpha, intensity, color, volume=None)->Surface:
    return light_volume_c(x, y, background_rgb, mask_alpha, intensity, color, volume)
# SUB FUNCTION
def light_volumetric(rgb, alpha, intensity,
                     color, volume)->Surface:
    return light_volumetric_c(rgb, alpha, intensity, color, volume)


# --------- FLATTEN ARRAY
# FLATTEN 2d -> BUFFER
def flatten2d(array):
    return flatten2d_c(array)

# FLATTEN RGB 3d ARRAY -> BUFFER
def flatten3d_rgb(array):
    return flatten3d_rgb_c(array)

# FLATTEN RGBA 3d ARRAY -> BUFFER
def flatten3d_rgba(array):
    return flatten3d_rgba_c(array)


# -------- NORMALISATION
# ARRAY
# NORMALIZED A 2d ARRAY (SELECTIVE NORMALISATION)
def array2d_normalized_thresh(array, threshold = 127):
    return array2d_normalized_thresh_c(array, threshold)

# NORMALIZED AN ARRAY
def array2d_normalized(array):
    return array2d_normalized_c(array)

# BUFFER NORMALISATION (SELECTIVE)
def buffer_normalized_thresh(buffer, threshold = 127):
    return buffer_normalized_thresh_c(buffer, threshold)

# BUFFER NORMALISATION
def buffer_normalized(array):
    return buffer_normalized_c(array)

# ----------- STACKING
# STACK RGB AND ALPHA BUFFERS
def stack_buffer(rgb_array_, alpha_, w, h, transpose):
    return stack_buffer_c(rgb_array_, alpha_, w, h, transpose)

# STACK OBJECTS RGB AND ALPHA
def stack_object(rgb_array_, alpha_, transpose=False):
    return stack_object_c(rgb_array_, alpha_, transpose)


# ----------- GAUSSIAN BLUR
# APPLY GAUSSIAN BLUR KERNEL 5x5 (COMPATIBLE WITH RGB ARRAY)
def blur5x5_array24(rgb_array_):
    return blur5x5_array24_c(rgb_array_)

# APPLY GAUSSIAN BLUR KERNEL 5x5 (COMPATIBLE WITH RGBA ARRAY)
def blur5x5_array32(rgb_array_):
    return blur5x5_array32_c(rgb_array_)

# APPLY GAUSSIAN BLUR KERNEL 5x5 (COMPATIBLE WITH BUFFER SURFACE 24 BIT)
def blur5x5_buffer24(rgb_array_, width, height, depth, float [::1] mask=None):
    return blur5x5_buffer24_c(rgb_array_, width, height, depth, mask)

# APPLY GAUSSIAN BLUR KERNEL 5x5 (COMPATIBLE WITH BUFFER SURFACE 32 BIT)
def blur5x5_buffer32(rgba_array_, width, height, depth, float [::1] mask=None):
    return blur5x5_buffer32_c(rgba_array_, width, height, depth, mask)

# ------------ SATURATION
# APPLY SATURATION TO AN RGB BUFFER USING A MASK(COMPATIBLE SURFACE 24 BIT)
def saturation_buffer_mask(array_, shift_, mask_)->Surface:
    return saturation_buffer_mask_c(array_, shift_, mask_)

# APPLY SATURATION TO AN RGB ARRAY USING A MASK(COMPATIBLE SURFACE 24 BIT)
def saturation_array24_mask(array_, shift_, mask_)->Surface:
    return saturation_array24_mask_c(array_, shift_, mask_)

# APPLY SATURATION TO AN RGBA ARRAY USING A MASK(COMPATIBLE SURFACE 32 BIT)
def saturation_array32_mask(array_, alpha_, shift_, mask_)->Surface:
    return saturation_array32_mask_c(array_, alpha_, shift_, mask_)

# APPLY SATURATION TO AN RGB ARRAY
def saturation_array24(array_, shift_):
    return saturation_array24_c(array_, shift_)

# APPLY SATURATION TO AN RGBA ARRAY
def saturation_array32(array_, alpha_, shift_):
    return saturation_array32_c(array_, alpha_, shift_)

# ------------- BRIGHT PASS FILTER
# BRIGHT PASS FILTER FOR SURFACE 24-BIT
def bpf24_b(image, threshold = 128):
    return bpf24_b_c(image, threshold)

# BRIGHT PASS FILTER FOR SURFACE 32-BIT
def bpf32_b(image, threshold = 128):
    return bpf32_b_c(image, threshold)

# -------------- FILTERING
# FILTERING SURFACE 24-BIT
def filtering24(surface_, mask_):
    return filtering24_c(surface_, mask_)

# FILTERING SURFACE 32-BIT
def filtering32(surface_, mask_):
    return filtering32_c(surface_, mask_)

# -------------- MAPPING ARRAY
# CONVERT 3D INDEXING TO 1D BUFFER INDEXING
def to1d(x, y, z, width, depth):
    return to1d_c(x, y, z, width, depth)

# CONVERT 1D BUFFER INDEXING TO 3D INDEXING
def to3d(index, width, depth):
    return to3d_c(index, width, depth)

# FLIP VERTICALLY A SINGLE BUFFER VALUE.
def vmap_buffer(index, width, height, depth):
    return vmap_buffer_c(index, width, height, depth)

# FLIP(RE-INDEXING) VERTICALLY AN ENTIRE ARRAY OR MEMORYVIEWSLICE
def vfb_rgb(source, target, width, height):
    return vfb_rgb_c(source, target, width, height)


# --------------- BLOOM
# BLOOM EFFECT FOR 24 BIT SURFACE
def bloom_effect_buffer24(surface_, threshold_, smooth_=1, mask_=None):
    return bloom_effect_buffer24_c(surface_, threshold_, smooth_, mask_)

# BLOOM EFFECT FOR 32 BIT SURFACE
def bloom_effect_buffer32(surface_, threshold_, smooth_=1, mask_=None):
    return bloom_effect_buffer32_c(surface_, threshold_, smooth_, mask_)

# ---------------- HEAT EFFECT
# HORIZONTAL HEAT WAVE EFFECT FOR RGB ARRAY (24 BIT SURFACE)
def heatwave_array24_horiz(
        rgba_array, mask_array, frequency, amplitude, attenuation=0.10, threshold=64):
    return heatwave_array24_horiz_c(
        rgba_array, mask_array, frequency, amplitude, attenuation, threshold)

# HORIZONTAL HEAT WAVE EFFECT FOR RGBA ARRAY (32 BIT SURFACE)
def heatwave_array32_horiz(
        rgba_array, mask_array, frequency, amplitude, attenuation=0.10, threshold=64):
    return heatwave_array32_horiz_c(
        rgba_array, mask_array, frequency, amplitude, attenuation, threshold)

# HORIZONTAL HEAT WAVE EFFECT FOR RGB BUFFER (COMPATIBLE 24-BIT SURFACE)
def heatwave_buffer24_horiz(rgb_buffer, mask_buffer, width, height, frequency,
                              amplitude, attenuation=0.10, threshold=64):
    return heatwave_buffer24_horiz_c(rgb_buffer, mask_buffer, width, height, frequency,
                              amplitude, attenuation, threshold)

# Create a greyscale 2d array (w, h)
def greyscale_3d_to_2d(array):
    return greyscale_3d_to_2d_c(array)

# ----------------IMPLEMENTATION -----------------


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef apply32b_c(unsigned char [:] rgb_buffer_,
               unsigned char [:] alpha_buffer_,
               float intensity,
               float [:] color,
               int w, int h):
    """
    Create a light effect on the given portion of the screen
    If the output surface is blit with the additive mode, the transparency alpha 
    will be merged with the background (rectangular image).
    In order to keep the light aspect (radial shape) do not blend with additive mode.
    Intensity can be adjusted in range (0.0 ... N). 
    
    :param rgb_buffer_: Portion of the screen to be exposed (Buffer 1d numpy.ndarray or MemoryViewSlice)
    Buffer containing RGB format pixels (numpy.uint8)
    buffer = surface.transpose(1, 0)
    buffer = buffer.flatten(order='C')
    :param alpha_buffer_: Light alpha buffer. 1d Buffer numpy.ndarray or MemoryvViewSlice 
    :param intensity: Float; Value defining the light intensity  
    :param color: numpy.ndarray; Light color numpy.ndarray filled with RGB floating values normalized 
    such as: array([128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0], float32, copy=False) 
    :param w: integer; light width
    :param h: integer; light height
    :return: Return a pygame.Surface 32-bit 
    """

    assert intensity >= 0.0, '\nIntensity value cannot be < 0.0'
    # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((w, h), SRCALPHA)

    cdef int a_length, b_length

    try:
        a_length = len(alpha_buffer_)
    except (ValueError, pygame.error):
        raise ValueError('\nAlpha buffer length not understood')

    try:
        b_length = len(rgb_buffer_)
    except (ValueError, pygame.error):
        raise ValueError('\nAlpha buffer length not understood')

    assert b_length == w * h * 3, \
        'Incorrect RGB buffer length, expecting %s got %s ' % (w * h * 3, b_length)
    assert a_length == w * h, \
        '\nIncorrect alpha buffer length, expecting %s got %s ' % (w * h, a_length)

    cdef:
        int i, j = 0
        unsigned char [::1] new_array = numpy.empty(w * h * 4, uint8)

    with nogil:
         for i in prange(0, b_length, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
              j = <int>(i / 4)
              new_array[i    ] = min(<unsigned char>(rgb_buffer_[i - j    ] * intensity * color[0]), 255)
              new_array[i + 1] = min(<unsigned char>(rgb_buffer_[i - j + 1] * intensity * color[1]), 255)
              new_array[i + 2] = min(<unsigned char>(rgb_buffer_[i - j + 2] * intensity * color[2]), 255)
              new_array[i + 3] = alpha_buffer_[j]

    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef area24_c(int x, int y, np.ndarray[np.uint8_t, ndim=3] background_rgb,
              np.ndarray[np.uint8_t, ndim=2] mask_alpha, float intensity=1.0,
              float [:] color=numpy.array([128.0, 128.0, 128.0], dtype=numpy.float32, copy=False),
              bint smooth=False, bint saturation=False, float sat_value=0.2, bint bloom=False,
              bint heat=False, float frequency=1):
    """
    Create a realistic light effect on a pygame.Surface or texture.
    
    You can blit the output surface with additive mode using pygame flag BLEND_RGBA_ADD.
    
    Modes definition 
    ================
    SMOOTH : Apply a Gaussian blur with kernel 5x5 over the output texture, the light effect will 
    be slightly blurred.
    Timing : 5ms for a 400x400x3 texture against 0.4812155ms without it)
    
    SATURATION : Create a saturation effect (increase of the texture lightness using HSL color conversion
    algorithm. saturation threshold value should be included in range [-1.0, 1.0] default is 0.2 
    Saturation above 0.5 will deteriorate the output coloration. Threshold value below zero will 
    greyscale output texture.
    Timing :  37ms for a 400x400x3 texture against 0.4812155ms without it)
    
    BLOOM: Create a bloom effect to the output texture.
    see https://github.com/yoyoberenguer/BLOOM for more information about bloom algorithm. 
    Bloom effect is CPU demanding (25ms for a 400x400x3 texture against 0.4812155ms without it)
    
    HEAT: Create a heat effect on the output itexture (using the alpha channel) 
    
    intensity: 
    Intensity is a float value defining how bright will be the light effect. 
    If intensity is zero, a new pygame.Surface is returned with RLEACCEL flag (empty surface)
    
    EFFECTS ARE NON CUMULATIVE
    
    Color allows you to change the light coloration, if omitted, the light color by default is 
    R = 128.0, G = 128.0 and B = 128.0 
       
    :param x: integer, light x coordinates (must be in range [0..max screen.size x] 
    :param y: integer, light y coordinates (must be in range [0..max screen size y]
    :param background_rgb: numpy.ndarray (w, h, 3) uint8. 3d array shape containing all RGB values
    of the background surface (display background).
    :param mask_alpha: numpy.ndarray (w, h) uint8, 2d array with light texture alpha values.
    For better appearances, choose a texture with a radial mask shape (maximum light intensity in the center)  
    :param color: numpy.array; Light color (RGB float), default 
    array([128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0], float32, copy=False)
    :param intensity: float; Light intensity range [0.0 ... 20.0]   
    :param bloom: boolean; Bloom effect, default False
    :param sat_value: float; Set the saturation value 
    :param saturation: boolean; Saturation effect
    :param smooth: boolean; Blur effect
    :param frequency: float; frequency must be incremental
    :param heat: boolean; Allow heat wave effect 
    :return: Return a pygame surface 24 bit without per-pixel information,
    surface with same size as the light texture. Represent the lit surface.
    """

    assert intensity >= 0.0, '\nIntensity value cannot be > 0.0'


    cdef int w, h, lx, ly, ax, ay
    try:
        w, h = background_rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    try:
        ax, ay = (<object>mask_alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((ax, ay), SRCALPHA), ax, ay

    # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((ax, ay), SRCALPHA), ax, ay

    lx = ax >> 1
    ly = ay >> 1

    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgb = empty((ax, ay, 3), uint8, order='C')
        np.ndarray[np.uint8_t, ndim=2] alpha = empty((ax, ay), uint8, order='C')
        int i=0, j=0
        float f
        int w_low = lx
        int w_high = lx
        int h_low = ly
        int h_high = ly

    if x < lx:
        w_low = x
    elif x > w - lx:
        w_high = w - x

    if y < ly:
        h_low = y
    elif y >  h - ly:
        h_high = h - y

    # RGB block and ALPHA
    rgb = background_rgb[x - w_low:x + w_high, y - h_low:y + h_high, :]
    alpha = mask_alpha[lx - w_low:lx + w_high, ly - h_low:ly + h_high]

    # METHOD 1 (numpy array transpose)
    # no significant gain
    # rgb = rgb.transpose(1, 0, 2)
    # ax, ay = rgb.shape[:2]
    # cdef unsigned char [:, :, :] new_array = empty((ax, ay, 3), numpy.uint8)
    #
    # with nogil:
    #     for i in prange(ax, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
    #         for j in range(ay):
    #             f = alpha[j, i] * ONE_255 * intensity
    #             new_array[i, j, 0] = <unsigned char>fmin(rgb[i, j, 0] * f, 255.0)
    #             new_array[i, j, 1] = <unsigned char>fmin(rgb[i, j, 1] * f, 255.0)
    #             new_array[i, j, 2] = <unsigned char>fmin(rgb[i, j, 2] * f, 255.0)
    # return pygame.image.frombuffer(new_array, (ay, ax), 'RGB'), ay, ax

    # METHOD 2
    # RGB ARRAY IS TRANSPOSED IN THE LOOP
    ax, ay = rgb.shape[:2]
    cdef:
        unsigned char [:, :, ::1] new_array = empty((ay, ax, 3), numpy.uint8)

    # NOTE the array is transpose
    with nogil:
        for i in prange(ax, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(ay):
                f = alpha[i, j] * ONE_255 * intensity
                new_array[j, i, 0] = <unsigned char>fmin(rgb[i, j, 0] * f * color[0], 255.0)
                new_array[j, i, 1] = <unsigned char>fmin(rgb[i, j, 1] * f * color[1], 255.0)
                new_array[j, i, 2] = <unsigned char>fmin(rgb[i, j, 2] * f * color[2], 255.0)
    # As the array is transposed we
    # we need to adjust ax and ay (swapped).
    ay, ax = new_array.shape[:2]

    # SMOOTH
    # Apply a gaussian 5x5 to smooth the output image
    # Only the RGB array is needed as we are working with
    # 24 bit surface.
    # Transparency is already included into new_array (see f variable above)
    if smooth:
        array_ = blur5x5_array24_c(new_array)
        surface = pygame.image.frombuffer(array_, (ax, ay), "RGB")

    # SATURATION
    # mask array is equivalent to array alpha normalized (float values)
    # sat_value variable can be adjusted at the function call
    # new_array is a portion of the background, the array is flipped.
    # NOTE the mask is optional, it filters the output image pixels and
    # remove bright edges or image contours.
    # If you wish to remove the mask alpha to gain an extra processing time
    # use the method saturation_array24_c instead (no mask).
    # e.g surface = saturation_array24(new_array, sat_value)
    elif saturation:
        mask = array2d_normalized_c(alpha)
        surface = saturation_array24_mask_c(new_array, sat_value, mask)
        # surface = saturation_array24(new_array, sat_value)

    # BLOOM
    elif bloom:
        surface = pygame.image.frombuffer(new_array, (ax, ay), 'RGB')
        # All alpha pixel values will be re-scaled between [0...1.0]
        mask = array2d_normalized_c(alpha)
        # threshold_ = 0 (highest brightness)
        # Bright pass filter will compute all pixels
        surface = bloom_effect_buffer24_c(surface, threshold_=0, smooth_=1, mask_=mask)

    elif heat:
        # alpha = numpy.full((ax, ay), 255, numpy.uint8)
        surface = heatwave_array24_horiz_c(numpy.asarray(new_array).transpose(1, 0, 2), # --> array is transposed
            alpha, frequency, (frequency % 2) / 1000.0, attenuation=0.10, threshold=64)

    else:
        surface = pygame.image.frombuffer(new_array, (ax, ay), 'RGB')

    return surface, ax, ay




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef area32_c(int x, int y, np.ndarray[np.uint8_t, ndim=3] background_rgb,
              np.ndarray[np.uint8_t, ndim=2] mask_alpha, float intensity=1.0,
              float [:] color=numpy.array([128.0, 128.0, 128.0], dtype=numpy.float32, copy=False),
              smooth=False, saturation=False, sat_value=0.2, bloom=False, heat=False, frequency=1.0):
    """
    
    Create a realistic light effect on a pygame.Surface or texture.
    
    You can blit the output surface with additive mode using pygame flag BLEND_RGBA_ADD.
    
    Modes definition 
    ================
    SMOOTH : Apply a Gaussian blur with kernel 5x5 over the output texture, the light effect will 
    be slightly blurred.
    Timing : 5ms for a 400x400x3 texture against 0.4812155ms without it)
    
    SATURATION : Create a saturation effect (increase of the texture lightness using HSL color conversion
    algorithm. saturation threshold value should be included in range [-1.0, 1.0] default is 0.2 
    Saturation above 0.5 will deteriorate the output coloration. Threshold value below zero will 
    greyscale output texture.
    Timing :  37ms for a 400x400x3 texture against 0.4812155ms without it)
    
    BLOOM: Create a bloom effect to the output texture.
    see https://github.com/yoyoberenguer/BLOOM for more information about bloom algorithm. 
    Bloom effect is CPU demanding (25ms for a 400x400x3 texture against 0.4812155ms without it)
    
    HEAT: Create a heat effect on the output itexture (using the alpha channel) 
    
    intensity: 
    Intensity is a float value defining how bright will be the light effect. 
    If intensity is zero, a new pygame.Surface is returned with RLEACCEL flag (empty surface)
    
    EFFECTS ARE NON CUMULATIVE
    
    Color allows you to change the light coloration, if omitted, the light color by default is 
    R = 128.0, G = 128.0 and B = 128.0 
    
    
    :param x: integer, light x coordinates (must be in range [0..max screen.size x] 
    :param y: integer, light y coordinates (must be in range [0..max screen size y]
    :param background_rgb: numpy.ndarray (w, h, 3) uint8. 3d array shape containing all RGB values
    of the background surface (display background).
    :param mask_alpha: numpy.ndarray (w, h) uint8, 2d array with light texture alpha values.
    For better appearances, choose a texture with a radial mask shape (maximum light intensity in the center)
    :param intensity: float; Light intensity range [0.0 ... 20.0] 
    :param color: numpy.array; Light color (RGB float), default 
    array([128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0], float32, copy=False)  
    :param bloom: boolean; Bloom effect, default False
    :param sat_value: float; Set the saturation value 
    :param saturation: boolean; Saturation effect
    :param smooth: boolean; Blur effect
    :param heat: boolean; Allow heat effect
    :param frequency: float; frequency must be incremental
    :return: Return a pygame surface 32 bit wit per-pixel information,
    """

    assert intensity >= 0.0, '\nIntensity value cannot be > 0.0'


    cdef int w, h, lx, ly, ax, ay
    try:
        w, h = background_rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    try:
        ax, ay = (<object>mask_alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((ax, ay), SRCALPHA), ax, ay

    # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((ax, ay), SRCALPHA), ax, ay

    lx = ax >> 1
    ly = ay >> 1

    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgb = empty((lx, ly, 3), uint8, order='C')
        np.ndarray[np.uint8_t, ndim=2] alpha = empty((lx, ly), uint8, order='C')
        int i=0, j=0
        float f
        int w_low = lx
        int w_high = lx
        int h_low = ly
        int h_high = ly

    if x < lx:
        w_low = x
    elif x > w - lx:
        w_high = w - x

    if y < ly:
        h_low = y
    elif y >  h - ly:
        h_high = h - y

    # RGB block and ALPHA
    rgb = background_rgb[x - w_low:x + w_high, y - h_low:y + h_high, :]
    alpha = mask_alpha[lx - w_low:lx + w_high, ly - h_low:ly + h_high]

    ax, ay = rgb.shape[:2]
    cdef unsigned char [:, :, ::1] new_array = empty((ay, ax, 4), numpy.uint8)  # TRANSPOSED

    with nogil:
        for i in prange(ax, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(ay):
                f = alpha[i, j] * ONE_255 * intensity
                new_array[j, i, 0] = <unsigned char>fmin(rgb[i, j, 0] * f, 255.0)
                new_array[j, i, 1] = <unsigned char>fmin(rgb[i, j, 1] * f, 255.0)
                new_array[j, i, 2] = <unsigned char>fmin(rgb[i, j, 2] * f, 255.0)
                new_array[j, i, 3] = alpha[i, j]

    # As the array is transposed ax and ay are swapped.
    ax, ay = new_array.shape[:2]

    # SMOOTH
    # Apply a gaussian 5x5 to smooth the output image
    # Only the RGB array is needed as we are working with
    # 24 bit surface.
    # Transparency is already included into new_array (see f variable above)
    if smooth:
        ax, ay = ay, ax
        array_ = blur5x5_array32_c(new_array)
        surface = pygame.image.frombuffer(array_, (ax, ay), "RGBA")

    # SATURATION
    # sat_value variable can be adjusted at the function call
    # new_array is a portion of the background, the array is flipped.
    elif saturation:
        surface = saturation_array32_c(new_array, alpha, sat_value)
        ax, ay = ay, ax
    # BLOOM
    elif bloom:
        ax, ay = ay, ax
        surface = pygame.image.frombuffer(new_array, (ax, ay), 'RGBA')
        # All alpha pixel values will be re-scaled between [0...1.0]
        mask = array2d_normalized_c(alpha)
        # threshold_ = 0 (highest brightness)
        # Bright pass filter will compute all pixels
        surface = bloom_effect_buffer32_c(surface, threshold_=0, smooth_=1, mask_=mask)

    elif heat:
        ax, ay = ay, ax
        # alpha = numpy.full((ax, ay), 255, dtype=numpy.uint8)
        surface = heatwave_array32_horiz_c(numpy.asarray(new_array).transpose(1, 0, 2), # --> array is transposed
            alpha, frequency, (frequency % 2) / 1000.0, attenuation=0.10, threshold=64)

    else:
        # (ay, ax) as array is transposed
        surface = pygame.image.frombuffer(new_array, (ay, ax), 'RGBA')
        # swap values
        ax, ay = ay, ax

    return surface, ax, ay




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef area24b_c(int x, int y, unsigned char [:, :, :] background_rgb,
               unsigned char [:, :] mask_alpha, float [:] color, float intensity,
               bint smooth, bint saturation, float sat_value, bint bloom, bint heat,
               float frequency, array_):
    """
    Create a realistic light effect on a pygame.Surface or texture.
    
    You can blit the output surface with additive mode using pygame flag BLEND_RGBA_ADD.
    
    Modes definition 
    ================
    SMOOTH : Apply a Gaussian blur with kernel 5x5 over the output texture, the light effect will 
    be slightly blurred.
    Timing : 5ms for a 400x400x3 texture against 0.4812155ms without it)
    
    SATURATION : Create a saturation effect (increase of the texture lightness using HSL color conversion
    algorithm. saturation threshold value should be included in range [-1.0, 1.0] default is 0.2 
    Saturation above 0.5 will deteriorate the output coloration. Threshold value below zero will 
    greyscale output texture.
    Timing :  37ms for a 400x400x3 texture against 0.4812155ms without it)
    
    BLOOM: Create a bloom effect to the output texture.
    see https://github.com/yoyoberenguer/BLOOM for more information about bloom algorithm. 
    Bloom effect is CPU demanding (25ms for a 400x400x3 texture against 0.4812155ms without it)
    
    intensity: 
    Intensity is a float value defining how bright will be the light effect. 
    If intensity is zero, a new pygame.Surface is returned with RLEACCEL flag (empty surface)
    
    
    EFFECTS ARE NON CUMULATIVE

    :param sat_value:
    :param x: integer; x coordinate
    :param y: integer; y coordinate
    :param background_rgb: 3d numpy.ndarray (w, h, 3) containing RGB  values of the background image
    :param mask_alpha: 2d numpy.ndarray (w, h) containing light mask alpha
    :param color: numpy.ndarray containing light colours (RGB values, unsigned char values)
    :param intensity: light intensity or brightness factor
    :param smooth: bool; smooth the final image (call a gaussian 5x5 function)
    :param saturation: bool; change output image saturation, create a black and
    white array from the mask_alpha argument.
    :param bloom: bool; create a bloom effect.
    :param heat: bool; create a heat effect
    :param frequency: float; incremental variable
    :return: return a pygame.Surface same size than the mask alpha (w, h) without per-pixel information
    """

    cdef int w, h
    try:
        w, h = (<object>background_rgb).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef int ax, ay, ax_, ay_
    try:
        ax, ay = (<object>mask_alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    assert intensity > 0.0, '\nIntensity value cannot be < 0.0'

    cdef int lx, ly
    lx = ax >> 1
    ly = ay >> 1

    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((w, h), pygame.RLEACCEL), ax, ay

    cdef:
        int b_length = ax * ay
        # unsigned char [::1] rgb = numpy.empty(b_length * 3, uint8)

    # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((ax, ay), pygame.RLEACCEL), ax, ay

    cdef:
        int i=0, j=0, ii, ix, jy, jy_=0, index=0
        float m, c1, c2, c3
        int w_low, w_high, h_low, h_high

    w_low  = max(x - lx, 0)
    w_high = min(x + lx, w)
    h_low  = max(y - ly, 0)
    h_high = min(y + ly, h)

    c1 = color[0] * intensity
    c2 = color[1] * intensity
    c3 = color[2] * intensity

    # new dimensions for RGB and ALPHA arrays
    ax_, ay_ = w_high - w_low, h_high - h_low

    cdef:
        unsigned char [::1] rgb = numpy.empty(ax_ * ay_ * 3, uint8)
        unsigned char [:, :] new_mask = numpy.empty((ax_, ay_), uint8)
        unsigned char [:, :] other_array = array_
        int ax3 = ax_ * 3
        int ayh = ay - h_high
        int adiff = ax - ax_

    with nogil:
        for j in prange(h_low, h_high, 1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            jy = j - h_low  # jy must start from zero

            # adjust the mask alpha (@top and bottom)
            jy_ = jy
            if ay_ < ay and y < ly:
                    jy_ = j + ayh

            for i in range(w_low, w_high, 1):
                ix = i - w_low  # ix must start at zero

                ## ii = to1d_c(ix, jy, 0, ax_, 3)
                ii = jy * ax3 + ix * 3

                # adjust the mask alpha (left side)
                if ax_ < ax and x < lx:
                    ix = ix + adiff

                new_mask[i - w_low, jy] = mask_alpha[ix, jy_]

                m = mask_alpha[ix, jy_] * ONE_255

                rgb[ii    ] = <unsigned char>fmin((background_rgb[i, j, 0]) * m * c1, 255.0)
                rgb[ii + 1] = <unsigned char>fmin((background_rgb[i, j, 1]) * m * c2, 255.0)
                rgb[ii + 2] = <unsigned char>fmin((background_rgb[i, j, 2]) * m * c3, 255.0)

    # SMOOTH
    # Apply a gaussian 5x5 to smooth the output image
    # Only the RGB array is needed as we are working with
    # 24 bit surface.
    if smooth:
        surface, array_notuse = blur5x5_buffer24_c(rgb, ax_, ay_, 3)

    # SATURATION
    # here the alpha mask (new_mask) is use for filtering
    # the output image contours
    elif saturation:
        mask = array2d_normalized_c(new_mask)
        surface = saturation_buffer_mask_c(rgb, sat_value, mask)

    # BLOOM
    elif bloom:
        surface = pygame.image.frombuffer(rgb, (ax_, ay_), 'RGB')
        # threshold = 0
        # All alpha pixel values will be re-scaled between [0...1.0]
        # Also threshold = 0 give a higher light spread and a larger lens
        # Threshold = 128 narrow the lens.

        mask = array2d_normalized_c(new_mask)

        # threshold_ = 0 (highest brightness)
        # Bright pass filter will compute all pixels
        surface = bloom_effect_buffer24_c(surface, threshold_=0, smooth_=1, mask_=mask)

    elif heat:
        surface = heatwave_buffer24_horiz_c(rgb,
            new_mask, ax_, ay_, frequency, (frequency % 2) / 100.0, attenuation=0.10, threshold=80)

    else:
        surface = pygame.image.frombuffer(rgb, (ax_, ay_), 'RGB')

    return surface, ax_, ay_



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef area24bb_c(int x, int y, unsigned char [:] background_rgb, int w, int h,
                unsigned char [:] mask_alpha, int ax, int ay,
                float [:] color, float intensity,
                bint smooth, bint saturation, float sat_value, bint bloom, bint heat, float frequency):
    """
    
    Create a realistic light effect on a pygame.Surface or texture.
    
    You can blit the output surface with additive mode using pygame flag BLEND_RGBA_ADD.
    
    Modes definition 
    ================
    SMOOTH : Apply a Gaussian blur with kernel 5x5 over the output texture, the light effect will 
    be slightly blurred.
    Timing : 5ms for a 400x400x3 texture against 0.4812155ms without it)
    
    SATURATION : Create a saturation effect (increase of the texture lightness using HSL color conversion
    algorithm. saturation threshold value should be included in range [-1.0, 1.0] default is 0.2 
    Saturation above 0.5 will deteriorate the output coloration. Threshold value below zero will 
    greyscale output texture.
    Timing :  37ms for a 400x400x3 texture against 0.4812155ms without it)
    
    BLOOM: Create a bloom effect to the output texture.
    see https://github.com/yoyoberenguer/BLOOM for more information about bloom algorithm. 
    Bloom effect is CPU demanding (25ms for a 400x400x3 texture against 0.4812155ms without it)
    
    intensity: 
    Intensity is a float value defining how bright will be the light effect. 
    If intensity is zero, a new pygame.Surface is returned with RLEACCEL flag (empty surface)
    
    
    EFFECTS ARE NON CUMULATIVE

    :param bloom: 
    :param ay: integer; Light surface height
    :param ax: integer; Light surface width 
    :param h: integer; background height 
    :param w: intenger; background width
    :param x: integer; x coordinate of the light source center
    :param y: integer; y coordinate of the light source center
    :param background_rgb: MemoryViewSlice or numpy.ndarray 1d buffer 
    contains all the RGB pixels of the background surface
    :param mask_alpha: 1d buffer containing alpha values of the light 
    :param color: numpy.ndarray containing light colours (RGB values, unsigned char values)
    :param intensity: float to multiply rgb values to increase color saturation
    :param smooth: bool; smooth the final image (call a gaussian 5x5 function)
    :param saturation: bool; change output image saturation
    :param sat_value: saturation value
    :param bloom: bool; create a bloom effect.
    :param heat: bool; create a heat wave effect
    :param frequency: float; incremental variable
    :return: return a pygame.Surface 24 bit format without per-pixel information
    """

    cdef int a_length, b_length

    try:
        a_length = len(background_rgb)
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    try:
        b_length = len(mask_alpha)
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    if a_length != w * h * 3:
        raise ValueError('\nArgument <background_rgb> '
                         'buffer length does not match given '
                         'width and height expecting %s got %s ' % (w * h * 3, a_length)
        )

    if b_length != ax * ay:
        raise ValueError('\nArgument <mask_alpha> '
                         'buffer length does not match given width and height'
        )

    cdef int lx, ly
    lx = ax >> 1
    ly = ay >> 1

    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((w, h), pygame.RLEACCEL), ax, ay

    # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((ax, ay), pygame.RLEACCEL), ax, ay

    cdef:
        float m, c1, c2, c3
        int w_low, w_high, h_low, h_high

    w_low  = max(x - lx, 0)
    w_high = min(x + lx, w)
    h_low  = max(y - ly, 0)
    h_high = min(y + ly, h)

    c1 = color[0] * intensity
    c2 = color[1] * intensity
    c3 = color[2] * intensity

    cdef int ax_, ay_
    ax_, ay_ = w_high - w_low, h_high - h_low

    cdef:
        unsigned char [::1] rgb = numpy.empty(ax_ * ay_ * 3, uint8)
        unsigned char [:] new_mask = numpy.empty(ax_ * ay_, uint8)
        int i, j, index, ii, jy, ix, jy_
        int ax3 = ax_ * 3
        int w3 = w * 3
        int ayh = ay - h_high
        int adiff = ax - ax_

    with nogil:

        for j in prange(h_low, h_high, 1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            jy = j - h_low  # iy must start from zero

            # adjust the mask alpha (@top and bottom)
            jy_ = jy
            if ay_ < ay and y < ly:
                    jy_ = j + ayh

            for i in range(w_low, w_high, 1):

                ii = j * w3 + i * 3

                ix = i - w_low  # ix must start at zero

                index = jy * ax3 + ix * 3

                 # adjust the mask alpha (left side)
                if ax_ < ax and x < lx:
                    ix = ix + adiff

                new_mask[<int>(index/3)] = mask_alpha[jy_* ax + ix]


                m = mask_alpha[jy_* ax + ix] * ONE_255

                # multiply pixel by alpha otherwise return a rectangle shape
                rgb[index    ] = <unsigned char>fmin(background_rgb[ii    ] * m * c1, 255.0)
                rgb[index + 1] = <unsigned char>fmin(background_rgb[ii + 1] * m * c2, 255.0)
                rgb[index + 2] = <unsigned char>fmin(background_rgb[ii + 2] * m * c3, 255.0)
    mask = new_mask

    # SMOOTH
    if smooth:
        mask = buffer_monochrome_thresh_c(new_mask, threshold_=0)
        surface, not_use = blur5x5_buffer24_c(rgb, ax_, ay_, 3, mask) #, numpy.asarray(new_mask, numpy.float32))

    # SATURATION
    elif saturation:
        new_mask = vfb_c(new_mask, numpy.empty(ax_ * ay_, dtype=numpy.uint8), ax_, ay_)  # Flip the buffer
        mask = buffer_normalized_thresh_c(new_mask, threshold=0)
        mask = numpy.asarray(mask, dtype=float32).reshape(ax_, ay_)
        surface = saturation_buffer_mask_c(rgb, 0.20, mask)
        ...

    # BLOOM
    elif bloom:
        surface = pygame.image.frombuffer(rgb, (ax_, ay_), 'RGB')
        new_mask = vfb_c(new_mask, numpy.empty(ax_ * ay_, dtype=numpy.uint8), ax_, ay_)  # Flip the buffer
        mask = buffer_normalized_thresh_c(new_mask, threshold=0)                         # Normalized buffer
        mask = numpy.asarray(mask, dtype=float32).reshape(ax_, ay_)                      # Transform to 2d (w, h)

        surface = bloom_effect_buffer24_c(surface, threshold_=0, smooth_=1, mask_=mask)

    elif heat:
        new_mask = vfb_c(new_mask, numpy.empty(ax_ * ay_, dtype=numpy.uint8), ax_, ay_)  # Flip the buffer
        mask = numpy.asarray(new_mask, dtype=numpy.uint8).reshape(ax_, ay_)                  # Transform to 2d (w, h)
        surface = heatwave_buffer24_horiz_c(rgb,
            mask, ax_, ay_, frequency, (frequency % 2) / 100.0, attenuation=0.10, threshold=80)


    else:
        surface = pygame.image.frombuffer(rgb, (ax_, ay_), 'RGB')

    return surface, ax_, ay_



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef area32b_c(int x, int y, unsigned char [:, :, :] background_rgb,
                unsigned char [:, :] mask_alpha, float [:] color, float intensity, bint smooth):
    """
    Create a realistic light effect on a pygame.Surface or texture
    C buffer method (slightly slower than light_area_c32 method)
    When blitting the surface onto the background display, do not use any blending effect
    This algorithm do not use multiprocessing unlike light_area_c32
    
    :param x: integer; x coordinate 
    :param y: integer; y coordinate
    :param background_rgb: 3d numpy.ndarray (w, h, 3) containing RGB  values of the background image 
    :param mask_alpha: 2d numpy.ndarray (w, h) containing light mask alpha 
    :param color: numpy.ndarray containing light colours (RGB values, unsigned char values)
    :param intensity: float to multiply rgb values to increase color saturation
    :param smooth: bool; smooth the final image
    :return: return a pygame.Surface same size than the mask alpha (w, h) with per - pixel information
    """

    cdef int w, h
    try:
        w, h = (<object>background_rgb).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef int ax, ay
    try:
        ax, ay = (<object>mask_alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef int lx, ly
    lx = ax >> 1
    ly = ay >> 1

    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((w, h), SRCALPHA)

    cdef:
        int b_length = ax * ay
        unsigned char [::1] rgba = numpy.empty(b_length * 4, uint8)
        int w_low = lx
        int w_high = lx
        int h_low = ly
        int h_high = ly

    if x < lx:
        w_low = x
    elif x > w - lx:
        w_high = w - x

    if y < ly:
        h_low = y
    elif y >  h - ly:
        h_high = h - y

    # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((ax, ay), SRCALPHA)

    cdef:
        int ix=0, iy=0, x_low = x - w_low, y_low = y - h_low
        int i=0, j=0, ii=0
        float c1, c2, c3

    c1 = color[0] * intensity
    c2 = color[1] * intensity
    c3 = color[2] * intensity
    with nogil:
        for j in prange(y_low, y + h_high, 1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            iy = j - y_low
            for i in range(x_low, x + w_high, 1):
                    ix = i - x_low
                    ii = (iy * ax + ix) * 4
                    rgba[ii    ] = <unsigned char>fmin(background_rgb[i, j, 0] * c1, 255.0)
                    rgba[ii + 1] = <unsigned char>fmin(background_rgb[i, j, 1] * c2, 255.0)
                    rgba[ii + 2] = <unsigned char>fmin(background_rgb[i, j, 2] * c3, 255.0)
                    rgba[ii + 3] = mask_alpha[ix, iy]//2
    if smooth:
        # TODO
        # blur_image = gaussian_blur5x5_array_32_c(numpy.asarray(rgba).reshape(ax, ay, 4))
        # return pygame.image.frombuffer(blur_image, (ay, ax), 'RGBA')
        raise NotImplemented
    else:
        return pygame.image.frombuffer(rgba, (ay, ax), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef light_volume_c(int x, int y, np.ndarray[np.uint8_t, ndim=3] background_rgb,
                    np.ndarray[np.uint8_t, ndim=2] mask_alpha, float intensity, float [:] color,
                    np.ndarray[np.uint8_t, ndim=3] volume):

    cdef int w, h, lx, ly

    try:
        w, h = background_rgb.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

     # return an empty Surface when intensity = 0.0
    if intensity == 0.0:
        return Surface((w, h), SRCALPHA), w, h

    try:
        lx, ly = (<object>mask_alpha).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    lx = lx >> 1
    ly = ly >> 1

    # Return an empty surface if the x or y are not within the normal range.
    if (x < 0) or (x > w - 1) or (y < 0) or (y > h - 1):
        return Surface((w, h), SRCALPHA), w, h

    cdef:
        np.ndarray[np.uint8_t, ndim=3] rgb = \
            numpy.empty((lx, ly, 3), numpy.uint8, order='C')
        np.ndarray[np.uint8_t, ndim=2] alpha = \
            numpy.empty((lx, ly), numpy.uint8, order='C')

        int w_low = lx
        int w_high = lx
        int h_low = ly
        int h_high = ly

    if x < lx:
        w_low = x
    elif x > w - lx:
        w_high = w - x

    if y < ly:
        h_low = y
    elif y >  h - ly:
        h_high = h - y

    # RGB block and ALPHA
    rgb = background_rgb[x - w_low:x + w_high, y - h_low:y + h_high, :]
    alpha = mask_alpha[lx - w_low:lx + w_high, ly - h_low:ly + h_high]

    if volume is not None:
        vol = light_volumetric_c(rgb, alpha, intensity, color, volume)
        return vol, vol.get_width(), vol.get_height()

    cdef int ax, ay
    ax, ay = rgb.shape[:2]

    cdef:
        unsigned char [:, :, ::1] new_array = empty((ay, ax, 3), uint8)
        int i=0, j=0
        float f

    with nogil:
        for i in prange(ax, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(ay):
                f = alpha[i, j] * ONE_255 * intensity
                new_array[j, i, 0] = <unsigned char>\
                    fmin(rgb[i, j, 0] * f * color[0], 255)
                new_array[j, i, 1] = <unsigned char>\
                    fmin(rgb[i, j, 1] * f * color[1], 255)
                new_array[j, i, 2] = <unsigned char>\
                    fmin(rgb[i, j, 2] * f * color[2], 255)

    ay, ax = new_array.shape[:2]
    return pygame.image.frombuffer(new_array, (ax, ay), 'RGB'), ax, ay




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef light_volumetric_c(unsigned char[:, :, :] rgb, unsigned char[:, :] alpha,
                        float intensity, float [:] color, unsigned char[:, :, :] volume):
    """
    
    :param rgb: numpy.ndarray (w, h, 3) uint8, array containing all the background RGB colors values
    :param alpha: numpy.ndarray (w, h) uint8 represent the light mask alpha transparency
    :param intensity: float, light intensity default value for volumetric effect is 1e-6, adjust the value to have
    the right light illumination.
    :param color: numpy.ndarray, Light color (RGB values)
    :param volume: numpy.ndarray, array containing the 2d volumetric texture to merge with the background RGB values
    The texture should be slightly transparent with white shades colors. Texture with black nuances
    will increase opacity
    :return: Surface, Returns a surface representing a 2d light effect with a 2d volumetric
    effect display the radial mask.
    """

    cdef int w, h, vol_width, vol_height

    try:
        w, h = (<object>alpha).shape[:2]
        vol_width, vol_height = (<object>volume).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    # assert (vol_width != w or vol_height != h), \
    #        'Assertion error, Alpha (w:%s, h:%s) and Volume (w:%s, h:%s) arrays shapes are not identical.' \
    #        % (w, h, vol_width, vol_height)

    cdef:
        unsigned char [:, :, ::1] new_array = empty((h, w, 3), uint8)
        int i=0, j=0
        float f

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                f = alpha[i, j] * ONE_255 * intensity
                new_array[j, i, 0] = <unsigned char>\
                    fmin(rgb[i, j, 0] * f * color[0] * volume[i, j, 0] * ONE_255, 255)
                new_array[j, i, 1] = <unsigned char>\
                    fmin(rgb[i, j, 1] * f * color[1] * volume[i, j, 1] * ONE_255, 255)
                new_array[j, i, 2] = <unsigned char>\
                    fmin(rgb[i, j, 2] * f * color[2] * volume[i, j, 2] * ONE_255, 255)

    cdef int ax, ay
    ax, ay = new_array.shape[:2]
    return pygame.image.frombuffer(new_array, (w, h), 'RGB')


# ----------------------------------------------- GAUSSIAN BLUR -----------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef blur5x5_buffer24_c(unsigned char [:] rgb_buffer,
                      int width, int height, int depth, float [::1] mask=None):
    """
    Method using a C-buffer as input image (width * height * depth) uint8 data type
    5 x5 Gaussian kernel used:
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    It uses convolution property to process the image in two passes (horizontal and vertical passes).
    Pixels convoluted outside the image edges will be set to an adjacent pixel edge values

    :param mask: 1d buffer (MemoryViewSlice) mask filled with values in range [0 ... 1.0]  
    :param depth: integer; image depth (3)RGB, default 3
    :param height: integer; image height
    :param width:  integer; image width
    :param rgb_buffer: 1d buffer representing a 24bit format pygame.Surface  
    :return: 24-bit Pygame.Surface without per-pixel information and array 
    """

    cdef:
        int b_length= len(rgb_buffer)


    # check if the buffer length equal theoretical length
    if b_length != (width * height * depth):
        print("\nIncorrect 24-bit format image.")

    # kernel 5x5 separable
    cdef:
        float [::1] kernel = \
            numpy.array(([1.0/16.0,
                          4.0/16.0,
                          6.0/16.0,
                          4.0/16.0,
                          1.0/16.0]), dtype=numpy.float32, copy=False)

        short int kernel_half = 2
        short int kernel_length = len(kernel)
        int xx, yy, index, i, ii
        float k, r, g, b
        char kernel_offset
        unsigned char red, green, blue
        xyz v;

        # convolve array contains pixels of the first pass(horizontal convolution)
        # convolved array contains pixels of the second pass.
        # buffer_ source pixels
        unsigned char [::1] convolve = numpy.empty(width * height * depth, numpy.uint8)
        unsigned char [::1] convolved = numpy.empty(width * height * depth, numpy.uint8)
        unsigned char [:] buffer_ = rgb_buffer

    with nogil:
        # horizontal convolution
        # goes through all RGB values of the buffer and apply the convolution
        for i in prange(0, b_length, depth, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

            if mask is not None:
                if mask[i//3] == 0.0:
                    convolve[i    ] = 0
                    convolve[i + 1] = 0
                    convolve[i + 2] = 0
                    continue

            r, g, b = 0, 0, 0

            # v.x point to the row value of the equivalent 3d array (width, height, depth)
            # v.y point to the column value ...
            # v.z is always = 0 as the i value point always
            # to the red color of a pixel in the C-buffer structure
            v = to3d_c(i, width, depth)

            # testing
            # index = to1d_c(v.x, v.y, v.z, width, 3)
            # print(v.x, v.y, v.z, i, index)

            for kernel_offset in range(-kernel_half, kernel_half + 1):

                k = kernel[kernel_offset + kernel_half]

                # Convert 1d indexing into a 3d indexing
                # v.x correspond to the row index value in a 3d array
                # v.x is always pointing to the red color of a pixel (see for i loop with
                # step = 3) in the C-buffer data structure.
                xx = v.x + kernel_offset

                # avoid buffer overflow
                if xx < 0 or xx > (width - 1):
                    red, green, blue = 0, 0, 0

                else:
                    # Convert the 3d indexing into 1d buffer indexing
                    # The index value must always point to a red pixel
                    # v.z = 0
                    index = to1d_c(xx, v.y, v.z, width, depth)

                    # load the color value from the current pixel
                    red = buffer_[index]
                    green = buffer_[index + 1]
                    blue = buffer_[index + 2]


                r = r + red * k
                g = g + green * k
                b = b + blue * k

            # place the new RGB values into an empty array (convolve)
            convolve[i    ] = <unsigned char>r
            convolve[i + 1] = <unsigned char>g
            convolve[i + 2] = <unsigned char>b

        # Vertical convolution
        # In order to vertically convolve the kernel, we have to re-order the index value
        # to fetch data vertically with the vmap_buffer function.
        for i in prange(0, b_length, depth, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

                # Use the mask as a filter
                if mask is not None:
                    if mask[i//3] == 0.0:
                        convolved[i    ] = 0
                        convolved[i + 1] = 0
                        convolved[i + 2] = 0
                        continue

                r, g, b = 0, 0, 0

                v = to3d_c(i, width, depth)

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    yy = v.y + kernel_offset

                    if yy < 0 or yy > (height-1):

                        red, green, blue = 0, 0, 0
                    else:

                        ii = to1d_c(v.x, yy, v.z, width, depth)
                        red, green, blue = convolve[ii],\
                            convolve[ii+1], convolve[ii+2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[i    ] = <unsigned char>r
                convolved[i + 1] = <unsigned char>g
                convolved[i + 2] = <unsigned char>b

    return pygame.image.frombuffer(convolved, (width, height), "RGB"), convolved



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef blur5x5_buffer32_c(unsigned char [:] rgba_buffer,
                      int width, int height, int depth, float [::1] mask=None):
    """
    Method using a C-buffer as input image (width * height * depth) uint8 data type
    5 x5 Gaussian kernel used:
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    It uses convolution property to process the image in two passes (horizontal and vertical passes).
    Pixels convoluted outside the image edges will be set to an adjacent pixel edge values

    :param mask: 1d buffer (MemoryViewSlice) mask filled with values in range [0 ... 1.0]  
    :param depth: integer; image depth (3)RGB, default 3
    :param height: integer; image height
    :param width:  integer; image width
    :param rgba_buffer: 1d buffer representing a 24bit format pygame.Surface  
    :return: 24-bit Pygame.Surface without per-pixel information and array 
    """

    cdef:
        int b_length= len(rgba_buffer)


    # check if the buffer length equal theoretical length
    if b_length != (width * height * depth):
        raise ValueError(
            "\nIncorrect 32-bit format image, expecting %s got %s " % (width * height * depth, b_length))

    # kernel 5x5 separable
    cdef:
        float [::1] kernel = \
            numpy.array(([1.0/16.0,
                          4.0/16.0,
                          6.0/16.0,
                          4.0/16.0,
                          1.0/16.0]), dtype=numpy.float32, copy=False)

        short int kernel_half = 2
        short int kernel_length = len(kernel)
        int xx, yy, index, i, ii
        float k, r, g, b
        char kernel_offset
        unsigned char red, green, blue
        xyz v;

        # convolve array contains pixels of the first pass(horizontal convolution)
        # convolved array contains pixels of the second pass.
        # buffer_ source pixels
        unsigned char [::1] convolve = numpy.empty(width * height * depth, numpy.uint8)
        unsigned char [::1] convolved = numpy.empty(width * height * depth, numpy.uint8)
        unsigned char [:] buffer_ = numpy.frombuffer(rgba_buffer, numpy.uint8)

    with nogil:
        # horizontal convolution
        # goes through all RGB values of the buffer and apply the convolution
        for i in prange(0, b_length, depth, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

            if mask is not None:
                if mask[i//3] == 0.0:
                    convolve[i    ] = 0
                    convolve[i + 1] = 0
                    convolve[i + 2] = 0
                    convolve[i + 3] = 0
                    continue

            r, g, b = 0, 0, 0

            # v.x point to the row value of the equivalent 3d array (width, height, depth)
            # v.y point to the column value ...
            # v.z is always = 0 as the i value point always
            # to the red color of a pixel in the C-buffer structure
            v = to3d_c(i, width, depth)

            # testing
            # index = to1d_c(v.x, v.y, v.z, width, 4)
            # print(v.x, v.y, v.z, i, index)

            for kernel_offset in range(-kernel_half, kernel_half + 1):

                k = kernel[kernel_offset + kernel_half]

                # Convert 1d indexing into a 3d indexing
                # v.x correspond to the row index value in a 3d array
                # v.x is always pointing to the red color of a pixel (see for i loop with
                # step = 4) in the C-buffer data structure.
                xx = v.x + kernel_offset

                # avoid buffer overflow
                if xx < 0 or xx > (width - 1):
                    red, green, blue = 0, 0, 0

                else:
                    # Convert the 3d indexing into 1d buffer indexing
                    # The index value must always point to a red pixel
                    # v.z = 0
                    index = to1d_c(xx, v.y, v.z, width, depth)

                    # load the color value from the current pixel
                    red = buffer_[index]
                    green = buffer_[index + 1]
                    blue = buffer_[index + 2]



                r = r + red * k
                g = g + green * k
                b = b + blue * k

            # place the new RGB values into an empty array (convolve)
            convolve[i    ] = <unsigned char>r
            convolve[i + 1] = <unsigned char>g
            convolve[i + 2] = <unsigned char>b
            convolve[i + 3] = buffer_[i + 3]

        # Vertical convolution
        # In order to vertically convolve the kernel, we have to re-order the index value
        # to fetch data vertically with the vmap_buffer function.
        for i in prange(0, b_length, depth, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

                # Use the mask as a filter
                if mask is not None:
                    if mask[i//3] == 0.0:
                        convolved[i    ] = 0
                        convolved[i + 1] = 0
                        convolved[i + 2] = 0
                        convolved[i + 3] = 0
                        continue

                r, g, b = 0, 0, 0

                v = to3d_c(i, width, depth)

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    yy = v.y + kernel_offset

                    if yy < 0 or yy > (height-1):

                        red, green, blue = 0, 0, 0
                    else:

                        ii = to1d_c(v.x, yy, v.z, width, depth)
                        red, green, blue = convolve[ii],\
                            convolve[ii+1], convolve[ii+2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[i    ] = <unsigned char>r
                convolved[i + 1] = <unsigned char>g
                convolved[i + 2] = <unsigned char>b
                convolved[i + 3] = buffer_[i + 3]

    return pygame.image.frombuffer(convolved, (width, height), "RGBA"), convolved



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] blur5x5_array24_c(unsigned char [:, :, :] rgb_array_):
    """
    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param rgb_array_: numpy.ndarray type (w, h, 3) uint8 
    :return: Return 24-bit a numpy.ndarray type (w, h, 3) uint8
    """


    cdef int w, h, dim
    try:
        w, h, dim = (<object>rgb_array_).shape[:3]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')


    # kernel_ = numpy.array(([1.0 / 16.0,
    #                        4.0 / 16.0,
    #                        6.0 / 16.0,
    #                        4.0 / 16.0,
    #                        1.0 / 16.0]), dtype=float32, copy=False)

    # kernel 5x5 separable
    cdef:
        # float [::1] kernel = kernel_
        float[5] kernel = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]
        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((w, h, 3), dtype=uint8)
        short int kernel_length = len(kernel)
        int x, y, xx, yy
        float k, r, g, b, s
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule='static', num_threads=4):  # range [0..h-1)

            for x in range(0, w):  # range [0..w-1]

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = rgb_array_[0, y, 0],\
                        rgb_array_[0, y, 1], rgb_array_[0, y, 2]
                    elif xx > (w - 1):
                        red, green, blue = rgb_array_[w-1, y, 0],\
                        rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule='static', num_threads=4):

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = convolve[x, 0, 0],\
                        convolve[x, 0, 1], convolve[x, 0, 2]
                    elif yy > (h -1):
                        red, green, blue = convolve[x, h-1, 0],\
                        convolve[x, h-1, 1], convolve[x, h-1, 2]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[x, y, 0], convolved[x, y, 1], convolved[x, y, 2] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b

    return convolved



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] blur5x5_array32_c(unsigned char [:, :, :] rgb_array_):
    """
    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param rgb_array_: 3d numpy.ndarray type (w, h, 4) uint8, RGBA values
    :return: Return a numpy.ndarray type (w, h, 4) uint8
    """

    cdef int w, h, dim
    try:
        w, h, dim = rgb_array_.shape[:3]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')


    # kernel_ = numpy.array(([1.0 / 16.0,
    #                        4.0 / 16.0,
    #                        6.0 / 16.0,
    #                        4.0 / 16.0,
    #                        1.0 / 16.0]), dtype=float32, copy=False)

    # kernel 5x5 separable
    cdef:
        # float [::1] kernel = kernel_
        float[5] kernel = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]
        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((w, h, 4), dtype=uint8)
        short int kernel_length = len(kernel)
        int x, y, xx, yy
        float k, r, g, b
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule='static', num_threads=4):

            for x in range(0, w):

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = rgb_array_[0, y, 0],\
                        rgb_array_[0, y, 1], rgb_array_[0, y, 2]
                    elif xx > (w - 1):
                        red, green, blue = rgb_array_[w-1, y, 0],\
                        rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule='static', num_threads=4):

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = convolve[x, 0, 0],\
                        convolve[x, 0, 1], convolve[x, 0, 2]
                    elif yy > (h -1):
                        red, green, blue = convolve[x, h-1, 0],\
                        convolve[x, h-1, 1], convolve[x, h-1, 2]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[x, y, 0], convolved[x, y, 1],\
                convolved[x, y, 2], convolved[x, y, 3] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b, rgb_array_[x, y, 3]

    return convolved





# ************* BELOW UNIT ARE REDUNDANT (ALREADY IN PEL.pyx ****************************


# ------------------------------------ SATURATION -------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_array24_mask_c(unsigned char [:, :, :] array_, float shift_, float [:, :] mask_array):
    """
    Change the saturation level of a pygame.Surface (compatible with 24bit only).
    Transform RGB model into HSL model and <shift_> saturation value.
    Optional mask_array to determine area to be modified.
    The mask should be a 2d array filled with float values
    
    :param array_: 3d numpy.ndarray shapes (w, h, 3) representing a 24bit format pygame.Surface.
    :param shift_: Value must be in range [-1.0 ... 1.0],
                   between [-1.0 ... 0.0] decrease saturation.
                   between [0.0  ... 1.0] increase saturation.
    :param mask_array: float numpy.ndarray shape (width, height) 
    :return: a pygame.Surface 24-bit without per-pixel information 
    """

    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    try:
        height, width = numpy.asarray(array_).shape[:2]
    except (ValueError, pygame.error):
        raise ValueError(
            '\nArray type not compatible, expecting MemoryViewSlice got %s ' % type(array_))

    cdef:
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        unsigned char r, g, b
        float h, l, s
        double *hsl = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]
        int i, j

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):

                # load pixel RGB values
                r = array_[j, i, 0]
                g = array_[j, i, 1]
                b = array_[j, i, 2]

                if mask_array[i, j] > 0.0:
                    # # change saturation
                    hsl = rgb_to_hsl(r * ONE_255, g * ONE_255, b * ONE_255)
                    h = hsl[0]
                    s = hsl[1]
                    l = hsl[2]
                    s = min((s + shift_), 1.0)
                    s = max(s, 0.0)
                    rgb = hsl_to_rgb(h, s, l)
                    r = <unsigned char>(rgb[0] * 255.0)
                    g = <unsigned char>(rgb[1] * 255.0)
                    b = <unsigned char>(rgb[2] * 255.0)
                    free(rgb)
                    free(hsl)

                new_array[j, i, 0] = r
                new_array[j, i, 1] = g
                new_array[j, i, 2] = b

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_array32_mask_c(unsigned char [:, :, :] array_, unsigned char [:, :] alpha_,
                               float shift_, float [:, :] mask_array=None):
    """
    Change the saturation level of a pygame.Surface (compatible with 32bit only).
    Transform RGBA model into HSL model and <shift_> saturation value.
    Optional mask_array to determine area to be modified.
    The mask_array should be a 2d array filled with float values 
    
    :param alpha_: 2d numpy.array or MemoryViewSlice containing surface alpha values
    :param array_: 3d numpy.ndarray shapes (w, h, 4) representing a 32-bit format pygame.Surface.
    :param shift_: Value must be in range [-1.0 ... 1.0],
                   between [-1.0 ... 0.0] decrease saturation.
                   between [0.0  ... 1.0] increase saturation.
    :param mask_array: float numpy.ndarray shape (width, height) 
    :return: a pygame.Surface 32-bit with per-pixel information 
    """

    assert -1.0 <= shift_ <= 1.0, '\nshift_ argument must be in range [-1.0 .. 1.0].'
    assert mask_array is not None, '\nmask_array argument cannot be None.'

    cdef int width, height

    try:
        height, width = (<object>array_).shape[:2]
    except (ValueError, pygame.error):
        try:
            height, width = numpy.asarray(array_).shape[:2]
        except (ValueError, pygame.error):
            raise ValueError('\nArray type not compatible,'
                             ' expecting MemoryViewSlice got %s ' % type(array_))

    cdef:
        unsigned char [:, :, ::1] new_array = empty((height, width, 4), dtype=uint8)
        unsigned char r, g, b
        float h, l, s
        double *hsl = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]
        int i, j

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):

                # load pixel RGB values
                r = array_[j, i, 0]
                g = array_[j, i, 1]
                b = array_[j, i, 2]

                if mask_array[i, j] > 0:
                    # # change saturation
                    hsl = rgb_to_hsl(r * ONE_255, g * ONE_255, b * ONE_255)
                    h = hsl[0]
                    s = hsl[1]
                    l = hsl[2]
                    s = min((s + shift_), 1.0)
                    s = max(s, 0.0)
                    rgb = hsl_to_rgb(h, s, l)
                    r = <unsigned char>(rgb[0] * 255.0)
                    g = <unsigned char>(rgb[1] * 255.0)
                    b = <unsigned char>(rgb[2] * 255.0)
                    free(rgb)
                    free(hsl)

                new_array[j, i, 0] = r
                new_array[j, i, 1] = g
                new_array[j, i, 2] = b
                new_array[j, i, 3] = alpha_[i, j]

    return pygame.image.frombuffer(new_array, (width, height), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_array24_c(unsigned char [:, :, :] array_, float shift_):
    """
    Change the saturation level of an array / pygame.Surface (compatible with 24-bit format image)
    Transform RGB model into HSL model and add <shift_> value to the saturation 
    
    :param array_: numpy.ndarray (w, h, 3) uint8 representing 24 bit format surface
    :param shift_: Value must be in range [-1.0 ... 1.0], negative values decrease saturation
    :return: a pygame.Surface 24-bit without per-pixel information 
    """

    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    try:
        width, height = (<object>array_).shape[:2]
    except (pygame.error, ValueError):
        raise ValueError(
            '\nArray type <array_> '
            'not understood, expecting numpy.ndarray or MemoryViewSlice got %s ' % type(array_))

    cdef:
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        int i=0, j=0
        unsigned char r, g, b
        float h, l, s
        double *hsl = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):
                r, g, b = array_[i, j, 0], array_[i, j, 1], array_[i, j, 2]
                hsl = rgb_to_hsl(<float>r * ONE_255, <float>g * ONE_255, <float>b * ONE_255)
                h = hsl[0]
                s = hsl[1]
                l = hsl[2]
                s = min((s + shift_), 0.5)
                s = max(s, 0.0)
                rgb = hsl_to_rgb(h, s, l)
                new_array[j, i, 0] = <unsigned char>(rgb[0] * 255.0)
                new_array[j, i, 1] = <unsigned char>(rgb[1] * 255.0)
                new_array[j, i, 2] = <unsigned char>(rgb[2] * 255.0)
                free(rgb)
                free(hsl)

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_array32_c(unsigned char [:, :, :] array_,
                          unsigned char [:, :] alpha_, float shift_):
    """
    Change the saturation level of an array/ pygame.Surface (compatible with 32-bit format image only)
    Transform RGB model into HSL model and add <shift_> value to the saturation 
    
    :param array_: numpy.ndarray shapes (w, h, 4) representing a pygame Surface 32 bit format
    :param alpha_: numpy.ndarray shapes (w, h) containing all alpha values
    :param shift_: Value must be in range [-1.0 ... 1.0], negative values decrease saturation  
    :return: a pygame.Surface 32-bit with per-pixel information 
    """

    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height, alpha_width, alpha_height

    try:
        # numpy.ndarray ?
        width, height =(<object>array_).shape[:2]
    except (ValueError, pygame.error):
        try:
            # MemoryViewSlice ?
            width, height = numpy.array(array_).shape[:2]
        except (ValueError, pygame.error):
            raise ValueError('\n'
                'Array <array_> type not understood '
                             'expecting numpy.ndarray or MemoryViewSlice got %s ' % type(array_))
    try:
        # numpy.ndarray ?
        alpha_width, alpha_height = (<object>alpha_).shape[:2]
    except (ValueError, pygame.error):
        try:
            # MemoryViewSlice ?
            width, height = numpy.array(array_).shape[:2]
        except (ValueError, pygame.error):
            raise ValueError('\n'
                'Array <alpha_> type not understood '
                             'exp'
                             'ecting numpy.ndarray or MemoryViewSlice got %s ' % type(alpha_))
    # REMOVE as array is flipped
    # if width != alpha_width or height != alpha_height:
    #     raise ValueError("\nArray size mismatch, array (w:%s, h:%s); alpha (w:%s, h:%s)"
    #                      % (width, height, alpha_width, alpha_height))

    cdef:
        unsigned char [:, :, ::1] new_array = empty((width, height, 4), dtype=uint8)
        int i=0, j=0
        float r, g, b
        float h, l, s
        double *hsl = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(height):

                # Load RGB
                r, g, b = array_[i, j, 0], array_[i, j, 1], array_[i, j, 2]

                hsl = rgb_to_hsl(r * ONE_255, g * ONE_255, b * ONE_255)
                h = hsl[0]
                s = hsl[1]
                l = hsl[2]
                s = min((s + shift_), 1.0)
                s = max(s, 0.0)
                rgb = hsl_to_rgb(h, s, l)
                r = rgb[0] * 255.0
                g = rgb[1] * 255.0
                b = rgb[2] * 255.0
                new_array[i, j, 0] = <unsigned char>r
                new_array[i, j, 1] = <unsigned char>g
                new_array[i, j, 2] = <unsigned char>b
                new_array[i, j, 3] = alpha_[j, i]

                free(rgb)
                free(hsl)

    return pygame.image.frombuffer(new_array, (height, width), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_buffer_mask_c(unsigned char [:] buffer_, float shift_, float [:, :] mask_array):
    """
    Change the saturation level of all selected pixels from a buffer.
    Transform RGB model into HSL model and <shift_> values.
    mask_array argument cannot be null. The mask should be a 2d array
    (filled with normalized float values in range[0.0 ... 1.0]). 
    mask_array[i, j] with indices i, j represent a monochrome value (R=G=B)
    
    :param buffer_: 1d Buffer representing a 24bit format pygame.Surface
    :param shift_: Value must be in range [-1.0 ... 1.0],
                   between [-1.0 ... 0.0] decrease saturation.
                   between [0.0  ... 1.0] increase saturation.
    :param mask_array: numpy.ndarray with shape (width, height) mask_array width and height 
    must be equal to the buffer length
    :return: a pygame.Surface 24-bit without per-pixel information 
    """

    assert isinstance(shift_, float), \
           'Expecting float for argument shift_, got %s ' % type(shift_)
    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int b_length
    try:
        b_length = len(<object>buffer_)
    except ValueError:
        raise ValueError("\nIncompatible buffer type got %s." % type(buffer_))

    cdef int width, height
    if mask_array is not None:
        try:
            width, height = (<object>mask_array).shape[:2]
        except (ValueError, pygame.error) as e:
            raise ValueError("\nIncompatible buffer type got %s." % type(buffer_))
    else:
        raise ValueError("\nIncompatible buffer type got %s." % type(buffer_))


    if width * height != (b_length // 3):
        raise ValueError("\nMask length and "
                         "buffer length mismatch, %s %s" % (b_length, width * height))

    cdef:
        int i=0, j=0, ii=0, ix
        unsigned char [:, :, ::1] new_array = empty((height, width, 3), dtype=uint8)
        unsigned char  r, g, b
        float h, l, s
        double *hsl = [0.0, 0.0, 0.0]
        double *rgb = [0.0, 0.0, 0.0]
        xyz pixel

    with nogil:
        for ii in prange(0, b_length, 3, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            # load pixel RGB values
            r = buffer_[ii]
            g = buffer_[ii + 1]
            b = buffer_[ii + 2]
            pixel = to3d_c(ii, width, 3)

            if mask_array[pixel.x, pixel.y] > 0.0:
                hsl = rgb_to_hsl(<float>r * ONE_255, <float>g * ONE_255, <float>b * ONE_255)
                h = hsl[0]
                s = hsl[1]
                l = hsl[2]
                s = min((s + shift_), 1.0)
                s = max(s, 0.0)
                rgb = hsl_to_rgb(h, s, l)
                r = <unsigned char>(rgb[0] * 255.0)
                g = <unsigned char>(rgb[1] * 255.0)
                b = <unsigned char>(rgb[2] * 255.0)

                free(rgb)
                free(hsl)

            new_array[pixel.y, pixel.x, 0] = r
            new_array[pixel.y, pixel.x, 1] = g
            new_array[pixel.y, pixel.x, 2] = b

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


# -----------------------------------ARRAY/ BUFFER INDEXING (MAPPING) -----------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline xyz to3d_c(int index, int width, int depth)nogil:
    """
    Map a 1d buffer pixel values into a 3d array, e.g buffer[index] --> array[i, j, k]
    Both (buffer and array) must have the same length (width * height * depth)
    To speed up the process, no checks are performed upon the function call and
    index, width and depth values must be > 0.

    :param index: integer; Buffer index value
    :param width: integer; image width
    :param depth: integer; image depth (3)RGB, (4)RGBA
    :return: Array index/key [x][y][z] pointing to a pixel RGB(A) identical
    to the buffer index value. Array index values are placed into a C structure (xyz)
    """
    cdef xyz v;
    cdef int ix = index // depth
    v.y = <int>(ix / width)
    v.x = ix % width
    v.z = index % depth
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline int to1d_c(int x, int y, int z, int width, int depth)nogil:
    """
    Map a 3d array value RGB(A) into a 1d buffer. e.g array[i, j, k] --> buffer[index]
   
    To speed up the process, no checks are performed upon the function call and
    x, y, z, width and depth values must be > 0 and both (buffer and array) must
    have the same length (width * height * depth)
    
    :param x: integer; array row value   
    :param y: integer; array column value
    :param z: integer; RGB(3) or RGBA(4) 
    :param width: source image width 
    :param depth: integer; source image depth (3)RGB or (4)RGBA
    :return: return the index value into a buffer for the given 3d array indices [x][y][z]. 
    """
    return <int>(y * width * depth + x * depth + z)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline int vmap_buffer_c(int index, int width, int height, int depth)nogil:
    """
    Vertically flipped a single buffer value.
     
    :param index: integer; index value 
    :param width: integer; image width
    :param height: integer; image height
    :param depth: integer; image depth (3)RGB or (4)RGBA
    :return: integer value pointing to the pixel in the buffer (traversed vertically). 
    """
    cdef:
        int ix
        int x, y, z
    ix = index // 4
    y = int(ix / height)
    x = ix % height
    z = index % depth
    return (x * width * depth) + (depth * y) + z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:] vfb_rgb_c(unsigned char [:] source, unsigned char [:] target,
                                   int width, int height)nogil:
    """
    Vertically flipped buffer type RGB
    
    Flip a C-buffer vertically filled with RGB values
    Re-sample a buffer in order to swap rows and columns of its equivalent 3d model
    For a 3d numpy.array this function would be equivalent to a transpose (1, 0, 2)
    Buffer length must be equivalent to width x height x RGB otherwise a valuerror
    will be raised.
    This method is using Multiprocessing OPENMP
    e.g
    Here is a 9 pixels buffer (length = 27), pixel format RGB
    
    buffer = [RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, RGB7, RGB8, RGB9]
    Equivalent 3d model would be (3x3x3):
    3d model = [RGB1 RGB2 RGB3]
               [RGB4 RGB5 RGB6]
               [RGB7 RGB8 RGB9]

    After vbf_rgb:
    output buffer = [RGB1, RGB4, RGB7, RGB2, RGB5, RGB8, RGB3, RGB6, RGB9]
    and its equivalent 3d model
    3D model = [RGB1, RGB4, RGB7]
               [RGB2, RGB5, RGB8]
               [RGB3, RGB6, RGB9]
        
    :param source: 1d buffer to flip vertically (unsigned char values).
     The array length is known with (width * height * depth). The buffer represent 
     image 's pixels RGB. 
     
    :param target: Target buffer must have same length than source buffer)
    :param width: integer; width of the image 
    :param height: integer; height of the image
    :return: Return a vertically flipped buffer 
    """
    cdef:
        int i, j, k, index
        unsigned char [:] flipped_array = target

    for i in prange(0, width * 3, 3):
        for j in range(0, height):
            index = i + (width * 3 * j)
            for k in range(3):
                flipped_array[(j * 3) + (i * height) + k] =  <unsigned char>source[index + k]

    return flipped_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:] vfb_c(unsigned char [:] source, unsigned char [:] target,
                                   int width, int height)nogil:
    cdef:
        int i, j
        unsigned char [:] flipped_array = target

    for i in prange(0, width):
        for j in range(0, height):
            flipped_array[j + (i * height)] =  <unsigned char>source[i + (width * j)]

    return flipped_array


# ------------------------------------------- ARRAY NORMALIZATION ------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef  float [:, :] array2d_normalized_thresh_c(unsigned char [:, :] array_, int threshold = 127):

    """
    NORMALIZED 2d ARRAY (selective normalization with threshold)
    Transform/convert an array_ shapes (w, h) containing unsigned char values 
    into a MemoryViewSlice (2d array_) with float values rescale in range [0 ... 1.0]
    UNDER THE THRESHOLD VALUE, all pixels will be black and ABOVE all pixels will be normalized.

    :param array_: numpy.array_ shape (w, h) containing unsigned int values (uint8)
    :param threshold: unsigned int; Threshold for the pixel, under that value all pixels will be black and
    above all pixels will be normalized.Default is 127
    :return: a MemoryViewSlice 2d array_ shape (w, h) with float values in range [0 ... 1.0] 
    
    """
    cdef:
        int w, h
    try:
        # assume (w, h) type array_
        w, h = array_.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood. Only 2d array_ shape (w, h) are compatible.')

    cdef:
        int i = 0, j = 0
        float [:, :] array_f = numpy.asarray(array_, dtype='float32')

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                if array_f[i, j] > threshold:
                    array_f[i, j] = <float>(array_f[i, j] * ONE_255)
                else:
                    array_f[i, j] = 0.0
    return array_f

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef  float [:, :] array2d_normalized_c(unsigned char [:, :] array):

    """
    NORMALIZED AN ARRAY 
    Transform/convert an array shapes (w, h) containing unsigned char values 
    into a MemoryViewSlice (2d array) with float values rescale in range [0 ... 1.0]

    :param array: numpy.array shape (w, h) containing unsigned int values (uint8)
    :return: a MemoryViewSlice 2d array shape (w, h) with float values in range [0 ... 1.0] 
    
    """
    cdef:
        int w, h
    try:
        # assume (w, h) type array
        w, h = array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood. Only 2d array shape (w, h) are compatible.')

    cdef:
        int i = 0, j = 0
        float [:, :] array_f = numpy.empty((w, h), numpy.float32)

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                array_f[i, j] = <float>(array[i, j] * ONE_255)
    return array_f

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef  float [:] buffer_normalized_thresh_c(unsigned char [:] buffer_, int threshold=127):

    """
    NORMALIZED A BUFFER
    Transform/convert a BUFFER containing unsigned char values 
    into a MemoryViewSlice same shape with float values rescale in range [0 ... 1.0]

    :param threshold: integer; Threshold value
    :param buffer_: BUFFER containing unsigned int values (uint8)
    :return: a MemoryViewSlice with float values in range [0 ... 1.0] 
    
    """
    cdef:
        int b_length, i=0
    try:
        # assume (w, h) type array
       b_length = len(<object>buffer_)
    except (ValueError, pygame.error) as e:
        raise ValueError('\nBuffer type not understood, compatible only with buffers.')

    cdef:
        float [:] array_f = numpy.empty(b_length, numpy.float32)

    with nogil:
        for i in prange(b_length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            if buffer_[i] > threshold:
                array_f[i] = <float>(buffer_[i] * ONE_255)
            else:
                array_f[i] = 0.0
    return array_f

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef  float [:] buffer_normalized_c(unsigned char [:] buffer_):

    """
    NORMALIZED A BUFFER
    Transform/convert a BUFFER containing unsigned char values 
    into a MemoryViewSlice same shape with float values rescale in range [0 ... 1.0]

    :param buffer_: BUFFER containing unsigned int values (uint8)
    :return: a MemoryViewSlice with float values in range [0 ... 1.0] 
    
    """
    cdef:
        int b_length, i=0
    try:
        # assume (w, h) type array
       b_length = len(buffer_)
    except (ValueError, pygame.error) as e:
        raise ValueError('\nBuffer type not understood, compatible only with buffers.')

    cdef:
        float [:] array_f = numpy.frombuffer(buffer_, dtype='float32')

    with nogil:
        for i in prange(b_length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                array_f[i] = <float>(buffer_[i] * ONE_255)
    return array_f


# ----------------------------------------- ARRAY/BUFFER TRANSFORMATION ------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef  float [:] buffer_monochrome_thresh_c(unsigned char [:] buffer_, unsigned int threshold_ = 127):

    """
    Transform/convert a buffer containing unsigned char values in 
    range [0 ... 255]) into equivalent C-Buffer RGB structure mono-chromatic (float values
    in range [0.0 ... 1.0]) same length, all value below threshold_ will be zeroed.

    :param buffer_: 1d buffer containing unsigned char values (uint8)
    :param threshold_: unsigned int; Pixel threshold
    :return: 1d C-Buffer contiguous structure (MemoryViewSlice) containing all 
    pixels RGB float values (range [0...1.0] R=G=B). 
    
    """
    assert isinstance(threshold_, int), \
           "Argument threshold should be a python int, got %s " % type(threshold_)
    cdef:
        int b_length
    try:
        # assume (w, h) type array
        b_length = len(<object>buffer_)

    except (ValueError, pygame.error) as e:
        raise ValueError('\nBuffer not understood.')

    cdef:
        float [::1] flat = numpy.empty(b_length, numpy.float32)
        int i = 0

    with nogil:
        for i in prange(b_length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            if buffer_[i] > threshold_:
                flat[i] = <float>(buffer_[i] * ONE_255)
            else:
                flat[i] = 0
    return flat


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef  unsigned char [:] flatten2d_c(unsigned char [:, :] array2d):

    """
    FLATTEN A 2D ARRAY SHAPE (W, H) INTO 1D C-BUFFER STRUCTURE OF LENGTH W * H

    :param array2d: numpy.array2d or MemoryViewSlice shape (w, h) of length w * h
    :return: 1D C-BUFFER contiguous structure (MemoryViewSlice) 
     
    """
    cdef:
        int w, h
    try:
        w, h = (<object>array2d).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood, compatible only with 2d array2d shape (w, h).')

    cdef:
        unsigned char [::1] flat = numpy.empty((w * h), dtype=numpy.uint8)
        int i = 0, j = 0, index

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for i in range(w):
                index = j * w + i
                flat[index] = array2d[i, j]
    return flat


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:] flatten3d_rgb_c(unsigned char [:, :, :] rgb_array):
    """
    FLATTEN AN ARRAY SHAPE (w, h, 3) CONTAINING RGB VALUES INTO 
    1D C-BUFFER RGB STRUCTURE OF LENGTH w * h * 3

    :param rgb_array: numpy.rgb_array shape (w, h, 3) RGB FORMAT
    :return: 1d C-Buffer contiguous structure (MemoryViewSlice) containing RGB pixels values.
    NOTE: to convert the BUFFER back into a pygame.Surface, prefer pygame.image.frombuffer(buffer, (w, h), 'RGB')
    instead of pygame.surfarray.make_surface.
    
    """

    cdef:
        int w, h, dim

    try:
        w, h, dim = (<object>rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood, compatible with 3d rgb_array only type(w, h, 3).')

    assert dim == 3, "Incompatible 3d rgb_array"

    cdef:
        unsigned char [::1] flat = numpy.empty((w * h * 3), dtype=numpy.uint8)
        int i = 0, j = 0, index
        # xyz v;

    with nogil:
        for j in prange(0, h, 1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for i in range(0, w):
                # index = to1d_c(x=i, y=j, z=0, width=w, depth=3)
                index = <int>(j * w * 3 + i * 3)
                flat[index  ] = rgb_array[i, j, 0]
                flat[index+1] = rgb_array[i, j, 1]
                flat[index+2] = rgb_array[i, j, 2]

    return flat


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:] flatten3d_rgba_c(unsigned char [:, :, :] rgba_array):

    """
    FLATTEN ARRAY SHAPE (W, H, 4) CONTAINING RGBA VALUES INTO
    A C-BUFFER LENGTH w * h * 4

    :param rgba_array: numpy.rgba_array shape (w, h, 4) containing RGBA unsigned char values (uint8)
    :return: 1d C-Buffer contiguous structure (MemoryViewSlice) containing RGBA pixels values.
    
    """

    cdef:
        int w, h, dim
    try:
        w, h, dim = (<object>rgba_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood, compatible with 3d rgba_array only type(w, h, 4).')
    assert dim == 4, "3d rgba_array is not shape (w, h, 4)"
    cdef:
        unsigned char [::1] flat = numpy.empty((w * h * 4), dtype=numpy.uint8)
        int i = 0, j = 0, index

    with nogil:
        for j in prange(h, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for i in range(w):
                index = <int>(j * w * 4 + i * 4)
                flat[index    ] = rgba_array[i, j, 0]
                flat[index + 1] = rgba_array[i, j, 1]
                flat[index + 2] = rgba_array[i, j, 2]
                flat[index + 3] = rgba_array[i, j, 3]
    return flat

# ----------------------------------------- BRIGHT PASS FILTERS ------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bpf24_b_c(image, int threshold = 128):
    """
    Bright pass filter for 24bit image (method using c-buffer)
    
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = c_buffer[i] * 0.299 + c_buffer[i+1] * 0.587 + c_buffer[i+2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.
    
    :param image: pygame.Surface 24 bit format (RGB)  without per-pixel information
    :param threshold: integer; Threshold to consider for filtering pixels luminance values,
    default is 128 range [0..255] unsigned char (python integer)
    :return: Return a 24 bit pygame.Surface filtered (only bright area of the image remains).
    """

    # Fallback to default threshold value if argument
    # threshold value is incorrect
    if 0 > threshold > 255:
        printf("\nArgument threshold must be in range [0...255], fallback to default value 128.")
        threshold = 128

    cdef:
        int w, h
    w, h = image.get_size()

    # make sure the surface is 24-bit format RGB
    if not image.get_bitsize() == 24:
        raise ValueError('Surface is not 24-bit format.')

    try:

        buffer_ = image.get_view('2')

    except (pygame.Error, ValueError):
        raise ValueError('\nInvalid surface.')

    # check sizes
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)

    cdef:
        int b_length = buffer_.length
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [::1] out_buffer = numpy.empty(b_length, numpy.uint8)
        int i = 0
        float lum, c

    with nogil:
        for i in prange(0, b_length, 3, schedule='static', num_threads=4):
            # ITU-R BT.601 luma coefficients
            lum = c_buffer[i] * 0.299 + c_buffer[i+1] * 0.587 + c_buffer[i+2] * 0.114
            if lum > threshold:
                c = (lum - threshold) / lum
                out_buffer[i  ] = <unsigned char>(c_buffer[i  ] * c)
                out_buffer[i+1] = <unsigned char>(c_buffer[i+1] * c)
                out_buffer[i+2] = <unsigned char>(c_buffer[i+2] * c)
            else:
                out_buffer[i], out_buffer[i+1], out_buffer[i+2] = 0, 0, 0

    return pygame.image.frombuffer(out_buffer, (w, h), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bpf32_b_c(image, int threshold = 128):
    """
    Bright pass filter for 32-bit image (method using c-buffer)
    
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = c_buffer[i] * 0.299 + c_buffer[i+1] * 0.587 + c_buffer[i+2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.
    
    :param image: pygame.Surface 32 bit format (RGBA)  without per-pixel information
    :param threshold: integer; Threshold to consider for filtering pixels luminance values
    :return: Return a 32-bit pygame.Surface filtered (only bright area of the image remains).
    """

    # Fallback to default threshold value if arguement
    # threshold value is incorrect
    if 0 > threshold > 255:
        printf("\nArgument threshold must be in range [0...255], fallback to default value 128.")
        threshold = 128

    assert isinstance(image, pygame.Surface), \
           "\nExpecting pygame surface for arguement image, got %s " % type(image)

    cdef:
        int w, h
    w, h = image.get_size()

    # make sure the surface is 32-bit format RGBA
    if not image.get_bitsize() == 32:
        raise ValueError('Surface is not 32-bit format.')

    try:

        buffer_ = image.get_view('2')

    except (pygame.Error, ValueError):
        raise ValueError('\nInvalid surface.')

    cdef:
        int b_length = buffer_.length
        unsigned char [:] c_buffer = numpy.frombuffer(buffer_, numpy.uint8)
        unsigned char [::1] out_buffer = numpy.empty(b_length, numpy.uint8)
        int i = 0
        float lum, c

    with nogil:
        for i in prange(0, b_length, 4, schedule='static', num_threads=4):
            # ITU-R BT.601 luma coefficients
            lum = c_buffer[i] * 0.299 + c_buffer[i+1] * 0.587 + c_buffer[i+2] * 0.114
            if lum > threshold:

                c = (lum - threshold) / lum
                out_buffer[i] = <unsigned char>(c_buffer[i] * c)
                out_buffer[i+1] = <unsigned char>(c_buffer[i+1] * c)
                out_buffer[i+2] = <unsigned char>(c_buffer[i+2] * c)
                out_buffer[i+3] = 255
            else:
                out_buffer[i], out_buffer[i+1], \
                out_buffer[i+2], out_buffer[i+3] = 0, 0, 0, 0

    return pygame.image.frombuffer(out_buffer, (w, h), 'RGBA')

# --------------------------------------------- FILTERING ----------------------------------------------


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef filtering24_c(surface_, mask_):
    """
    Multiply mask values with an array representing the surface pixels (Compatible 24 bit only).
    Mask values are floats in range (0 ... 1.0)

    :param surface_: pygame.Surface compatible 24-bit
    :param mask_: 2d array (MemoryViewSlice) containing alpha values (float).
    The mask_ output image is monochromatic (values range [0 ... 1.0] and R=B=G.
    :return: Return a pygame.Surface 24 bit
    """
    cdef int w, h, w_, h_
    w, h = surface_.get_size()
    try:
        w_, h_ = (<object>mask_).shape[:2]
    except (ValueError, pygame.error):
       raise ValueError(
           '\nArgument mask_ type not understood, '
           'expecting numpy.ndarray type (w, h) got %s ' % type(mask_))


    assert w == w_ and h == h_, \
        '\nSurface and mask size does not match (w:%s, h:%s), ' \
        '(w_:%s, h_:%s) ' % (w, h, w_, h_)

    try:
        rgb_ = pygame.surfarray.pixels3d(surface_)
    except (ValueError, pygame.error):
        try:
            rgb_ = pygame.surfarray.array3d(surface_)
        except (ValueError, pygame.error):
            raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb = rgb_.transpose(1, 0, 2)
        unsigned char [:, :, ::1] rgb1 = numpy.empty((h, w, 3), numpy.uint8)
        float [:, :] mask = numpy.asarray(mask_, numpy.float32)
        int i, j
    with nogil:
        for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                rgb1[j, i, 0] = <unsigned char>(rgb[j, i, 0] * mask[i, j])
                rgb1[j, i, 1] = <unsigned char>(rgb[j, i, 1] * mask[i, j])
                rgb1[j, i, 2] = <unsigned char>(rgb[j, i, 2] * mask[i, j])

    return pygame.image.frombuffer(rgb1, (w, h), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef filtering32_c(surface_, mask_):
    """
    Multiply mask values with an array representing the surface pixels (Compatible 32 bit only).
    Mask values are floats in range (0 ... 1.0)

    :param surface_: pygame.Surface compatible 32-bit
    :param mask_: 2d array (MemoryViewSlice) containing alpha values (float).
    The mask_ output image is monochromatic (values range [0 ... 1.0] and R=B=G.
    :return: Return a pygame.Surface 32-bit
    """

    cdef int w, h, w_, h_
    w, h = surface_.get_size()

    try:
        w_, h_ = (<object>mask_).shape[:2]
    except (ValueError, pygame.error):
        raise ValueError(
            '\nArgument mask_ type not understood, expecting numpy.ndarray got %s ' % type(mask_))

    assert w == w_ and h == h_, 'Surface and mask size does not match.'

    try:
        rgb_ = pygame.surfarray.pixels3d(surface_)
    except (ValueError, pygame.error):
        try:
            rgb_ = pygame.surfarray.array3d(surface_)
        except (ValueError, pygame.error):
            raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb = rgb_
        unsigned char [:, :, ::1] rgb1 = numpy.empty((h, w, 4), numpy.uint8)
        float [:, :] mask = numpy.asarray(mask_, numpy.float32)
        int i, j
    with nogil:
        for i in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                rgb1[j, i, 0] = <unsigned char>(rgb[i, j, 0] * mask[i, j])
                rgb1[j, i, 1] = <unsigned char>(rgb[i, j, 1] * mask[i, j])
                rgb1[j, i, 2] = <unsigned char>(rgb[i, j, 2] * mask[i, j])
                rgb1[j, i, 3] = <unsigned char>(mask[i, j] * 255.0)

    return pygame.image.frombuffer(rgb1, (w, h), 'RGBA')

# ----------------------------------------- BLOOM EFFECT -------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef bloom_effect_buffer24_c(surface_, int threshold_, int smooth_=1, mask_=None):
    """
    Create a bloom effect on a pygame.Surface (compatible 24 bit surface)
    This method is using C-buffer structure.

    definition:
        Bloom is a computer graphics effect used in video games, demos,
        and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.

    1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
      bpf24_b_c or bpf32_b_c (adjust the threshold value to get the best filter effect).
    2)Downside the newly created bpf image by factor x2, x4, x8, x16 using the pygame scale method (no need to
      use smoothscale (bilinear filtering method).
    3)Apply a Gaussian blur 5x5 effect on each of the downsized bpf images (if smooth_ is > 1, then the Gaussian
      filter 5x5 will by applied more than once. Note, this have little effect on the final image quality.
    4)Re-scale all the bpf images using a bilinear filter (width and height of original image).
      Using an un-filtered rescaling method will pixelate the final output image.
      For best performances sets smoothscale acceleration.
      A value of 'GENERIC' turns off acceleration. 'MMX' uses MMX instructions only.
      'SSE' allows SSE extensions as well.
    5)Blit all the bpf images on the original surface, use pygame additive blend mode for
      a smooth and brighter effect.

    Notes:
    The downscaling process of all sub-images could be done in a single process to increase performance.

    :param mask_: 2d array shape (w, h)  
    :param surface_: pygame.Surface 24 bit format surface
    :param threshold_: integer; Threshold value used by the bright pass algorithm (default 128)
    :param smooth_: integer; Number of Guaussian blur 5x5 to apply to downsided images.
    :return : Returns a pygame.Surface with a bloom effect (24 bit surface)


    """
    # Create a copy of the pygame surface,
    # TODO: implement a C function for copying SDL surface (pygame surfac copy is too slow)
    # TODO: TRY MASK OPTION IN blur5x5_buffer24_c and Filtering and compare speed
    # very slow
    surface_cp = surface_.copy()

    assert smooth_ > 0, \
           "\nArgument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
           "\nArgument threshold_ must be in range [0...255] got %s " % threshold_

    cdef:
        int w, h, bitsize
        int w2, h2, w4, h4, w8, h8, w16, h16

    w, h = surface_.get_size()
    bitsize = surface_.get_bitsize()

    if not bitsize == 24:
        raise ValueError('\nIncorrect image format, expecting 24-bit got %s ' % bitsize)

    bpf_surface =  bpf24_b_c(surface_, threshold=threshold_)

    w2, h2 = w >> 1, h >> 1
    s2 = pygame.transform.scale(bpf_surface, (w2, h2))
    array_ = numpy.array(s2.get_view("3"), dtype=uint8, copy=False).transpose(1, 0, 2)
    b2 = array_.flatten(order='C')
    if smooth_ > 1:
        for r in range(smooth_):
            b2_blurred, b2 = blur5x5_buffer24_c(b2, w2, h2, 3)#, mask_)
    else:
        b2_blurred, b2 = blur5x5_buffer24_c(b2, w2, h2, 3)#, mask_)

    # downscale x 4 using fast scale pygame algorithm (no re-sampling)
    w4, h4 = w >> 2, h >> 2
    s4 = pygame.transform.scale(bpf_surface, (w4, h4))
    array_ = numpy.array(s4.get_view("3"), dtype=uint8, copy=False).transpose(1, 0, 2)
    b4 = array_.flatten(order='C')
    if smooth_ > 1:
        for r in range(smooth_):
            b4_blurred, b4 = blur5x5_buffer24_c(b4, w4, h4, 3)#, mask_)
    else:
        b4_blurred, b4 = blur5x5_buffer24_c(b4, w4, h4, 3)#, mask_)

    # downscale x 8 using fast scale pygame algorithm (no re-sampling)
    w8, h8 = w >> 3, h >> 3
    s8 = pygame.transform.scale(bpf_surface, (w8, h8))
    array_ = numpy.array(s8.get_view("3"), dtype=uint8, copy=False).transpose(1, 0, 2)
    b8 = array_.flatten(order='C')
    if smooth_ > 1:
        for r in range(smooth_):
            b8_blurred, b8 = blur5x5_buffer24_c(b8, w8, h8, 3)#, mask_)
    else:
        b8_blurred, b8 = blur5x5_buffer24_c(b8, w8, h8, 3)#, mask_)

    # downscale x 16 using fast scale pygame algorithm (no re-sampling)
    w16, h16 = w >> 4, h >> 4
    s16 = pygame.transform.scale(bpf_surface, (w16, h16))
    array_ = numpy.array(s16.get_view("3"), dtype=uint8, copy=False).transpose(1, 0, 2)
    b16 = array_.flatten(order='C')
    if smooth_ > 1:
        for r in range(smooth_):
            b16_blurred, b16 = blur5x5_buffer24_c(b16, w16, h16, 3)#, mask_)
    else:
        b16_blurred, b16 = blur5x5_buffer24_c(b16, w16, h16, 3)#, mask_)

    s2 = pygame.transform.smoothscale(b2_blurred, (w , h))
    s4 = pygame.transform.smoothscale(b4_blurred, (w , h))
    s8 = pygame.transform.smoothscale(b8_blurred, (w, h))
    s16 = pygame.transform.smoothscale(b16_blurred, (w, h))

    surface_cp.blit(s2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s4, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s8, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s16, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    if mask_ is not None:
        # Multiply mask surface pixels with mask values.
        # RGB pixels = 0 when mask value = 0.0, otherwise
        # modify RGB amplitude
        surface_cp = filtering24_c(surface_cp, mask_)

    return surface_cp



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef bloom_effect_buffer32_c(surface_, int threshold_, int smooth_=1, mask_=None):
    """
    Create a bloom effect on a pygame.Surface (compatible 32 bit surface)
    This method is using C-buffer structure.

    definition:
        Bloom is a computer graphics effect used in video games, demos,
        and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.

    1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
      bpf24_b_c or bpf32_b_c (adjust the threshold value to get the best filter effect).
    2)Downside the newly created bpf image by factor x2, x4, x8, x16 using the pygame scale method (no need to
      use smoothscale (bilinear filtering method).
    3)Apply a Gaussian blur 5x5 effect on each of the downsized bpf images (if smooth_ is > 1, then the Gaussian
      filter 5x5 will by applied more than once. Note, this have little effect on the final image quality.
    4)Re-scale all the bpf images using a bilinear filter (width and height of original image).
      Using an un-filtered rescaling method will pixelate the final output image.
      For best performances sets smoothscale acceleration.
      A value of 'GENERIC' turns off acceleration. 'MMX' uses MMX instructions only.
      'SSE' allows SSE extensions as well.
    5)Blit all the bpf images on the original surface, use pygame additive blend mode for
      a smooth and brighter effect.

    Notes:
    The downscaling process of all sub-images could be done in a single process to increase performance.

    :param mask_: 2d array shape (w, h)  
    :param surface_: pygame.Surface 32 bit format surface
    :param threshold_: integer; Threshold value used by the bright pass algorithm (default 128)
    :param smooth_: integer; Number of Guaussian blur 5x5 to apply to downsided images.
    :return : Returns a pygame.Surface with a bloom effect (32 bit surface)


    """
    # Create a copy of the pygame surface,
    # TODO: implement a C function for copying SDL surface (pygame surfac copy is too slow)
    # TODO: TRY MASK OPTION IN blur5x5_buffer24_c and Filtering and compare speed
    # very slow
    surface_cp = surface_.copy()

    assert smooth_ > 0, \
           "\nArgument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
           "\nArgument threshold_ must be in range [0...255] got %s " % threshold_

    cdef:
        int w, h, bitsize
        int w2, h2, w4, h4, w8, h8, w16, h16

    w, h = surface_.get_size()
    bitsize = surface_.get_bitsize()

    if not bitsize == 32:
        raise ValueError('\nIncorrect image format, expecting 32-bit got %s ' % bitsize)

    bpf_surface =  bpf32_b_c(surface_, threshold=threshold_)

    w2, h2 = w >> 1, h >> 1
    s2 = pygame.transform.scale(bpf_surface, (w2, h2))
    b2 = numpy.frombuffer(s2.get_view("2"), numpy.uint8)
    if smooth_ > 1:
        for r in range(smooth_):
            b2_blurred, b2 = blur5x5_buffer32_c(b2, w2, h2, 4)#, mask_)
    else:
        b2_blurred, b2 = blur5x5_buffer32_c(b2, w2, h2, 4)#, mask_)

    # downscale x 4 using fast scale pygame algorithm (no re-sampling)
    w4, h4 = w >> 2, h >> 2
    s4 = pygame.transform.scale(bpf_surface, (w4, h4))
    b4 = numpy.frombuffer(s4.get_view("2"), numpy.uint8)
    if smooth_ > 1:
        for r in range(smooth_):
            b4_blurred, b4 = blur5x5_buffer32_c(b4, w4, h4, 4)#, mask_)
    else:
        b4_blurred, b4 = blur5x5_buffer32_c(b4, w4, h4, 4)#, mask_)

    # downscale x 8 using fast scale pygame algorithm (no re-sampling)
    w8, h8 = w >> 3, h >> 3
    s8 = pygame.transform.scale(bpf_surface, (w8, h8))
    b8 = numpy.frombuffer(s8.get_view("2"), numpy.uint8)
    if smooth_ > 1:
        for r in range(smooth_):
            b8_blurred, b8 = blur5x5_buffer32_c(b8, w8, h8, 4)#, mask_)
    else:
        b8_blurred, b8 = blur5x5_buffer32_c(b8, w8, h8, 4)#, mask_)

    # downscale x 16 using fast scale pygame algorithm (no re-sampling)
    w16, h16 = w >> 4, h >> 4
    s16 = pygame.transform.scale(bpf_surface, (w16, h16))
    b16 = numpy.frombuffer(s16.get_view("2"), numpy.uint8)
    if smooth_ > 1:
        for r in range(smooth_):
            b16_blurred, b16 = blur5x5_buffer32_c(b16, w16, h16, 4)#, mask_)
    else:
        b16_blurred, b16 = blur5x5_buffer32_c(b16, w16, h16, 4)#, mask_)

    s2 = pygame.transform.smoothscale(b2_blurred, (w , h))
    s4 = pygame.transform.smoothscale(b4_blurred, (w , h))
    s8 = pygame.transform.smoothscale(b8_blurred, (w, h))
    s16 = pygame.transform.smoothscale(b16_blurred, (w, h))

    surface_cp.blit(s2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s4, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s8, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s16, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


    if mask_ is not None:
        # Multiply mask surface pixels with mask values.
        # RGB pixels = 0 when mask value = 0.0, otherwise
        # modify RGB amplitude
        surface_cp = filtering32_c(surface_cp.convert_alpha(), mask_)

    return surface_cp

#-------------------------------------- STACKING ------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef stack_object_c(unsigned char[:, :, :] rgb_array_,
                    unsigned char[:, :] alpha_, bint transpose=False):
    """
    Stack RGB pixel values together with alpha values and return a python object,
    numpy.ndarray (faster than numpy.dstack)
    If transpose is True, transpose rows and columns of output array.
    
    :param transpose: boolean; Transpose rows and columns
    :param rgb_array_: numpy.ndarray (w, h, 3) uint8 containing RGB values 
    :param alpha_: numpy.ndarray (w, h) uint8 containing alpha values 
    :return: return a contiguous numpy.ndarray (w, h, 4) uint8, stack array of RGBA pixel values
    The values are copied into a new array.
    """
    cdef int width, height
    try:
        width, height = (<object> rgb_array_).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char[:, :, ::1] new_array =  numpy.empty((width, height, 4), dtype=uint8)
        unsigned char[:, :, ::1] new_array_t =  numpy.empty((height, width, 4), dtype=uint8)
        int i=0, j=0
    # Equivalent to a numpy.dstack
    with nogil:
        # Transpose rows and columns
        if transpose:
            for j in prange(0, height, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for i in range(0, width):
                    new_array_t[j, i, 0] = rgb_array_[i, j, 0]
                    new_array_t[j, i, 1] = rgb_array_[i, j, 1]
                    new_array_t[j, i, 2] = rgb_array_[i, j, 2]
                    new_array_t[j, i, 3] =  alpha_[i, j]

        else:
            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for j in range(0, height):
                    new_array[i, j, 0] = rgb_array_[i, j, 0]
                    new_array[i, j, 1] = rgb_array_[i, j, 1]
                    new_array[i, j, 2] = rgb_array_[i, j, 2]
                    new_array[i, j, 3] =  alpha_[i, j]

    return asarray(new_array) if transpose == False else asarray(new_array_t)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cdef unsigned char[::1] stack_buffer_c(rgb_array_, alpha_, int w, int h, bint transpose=False):
    """
    Stack RGB & ALPHA MemoryViewSlice C-buffers structures together.
    If transpose is True, the output MemoryViewSlice is flipped.
    
    :param h: integer; Texture height
    :param w: integer; Texture width
    :param transpose: boolean; Transpose rows and columns (default False)
    :param rgb_array_: MemoryViewSlice or pygame.BufferProxy (C-buffer type) representing the texture
    RGB values filled with uint8
    :param alpha_:  MemoryViewSlice or pygame.BufferProxy (C-buffer type) representing the texture
    alpha values filled with uint8 
    :return: Return a contiguous MemoryViewSlice representing RGBA pixel values
    """

    cdef:
        int b_length = w * h * 3
        int new_length = w * h * 4
        unsigned char [:] rgb_array = rgb_array_
        unsigned char [:] alpha = alpha_
        unsigned char [::1] new_buffer =  numpy.empty(new_length, dtype=numpy.uint8)
        unsigned char [::1] flipped_array = numpy.empty(new_length, dtype=numpy.uint8)
        int i=0, j=0, ii, jj, index, k
        int w4 = w * 4

    with nogil:

        for i in prange(0, b_length, 3, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                ii = i // 3
                jj = ii * 4
                new_buffer[jj]   = rgb_array[i]
                new_buffer[jj+1] = rgb_array[i+1]
                new_buffer[jj+2] = rgb_array[i+2]
                new_buffer[jj+3] = alpha[ii]

        if transpose:
            for i in prange(0, w4, 4, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                for j in range(0, h):
                    index = i + (w4 * j)
                    k = (j * 4) + (i * h)
                    flipped_array[k    ] = new_buffer[index    ]
                    flipped_array[k + 1] = new_buffer[index + 1]
                    flipped_array[k + 2] = new_buffer[index + 2]
                    flipped_array[k + 3] = new_buffer[index + 3]
            return flipped_array

    return new_buffer

#------------------------------------------------ HEAT EFFECT -----------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef heatwave_array24_horiz_c(unsigned char [:, :, :] rgb_array,
                            unsigned char [:, :] mask_array,
                            float frequency, float amplitude, float attenuation=0.10,
                            unsigned char threshold=64):
    """
    
    DISTORTION EQUATION: 
    distortion = sin(x * attenuation + frequency) * amplitude * mask_array[x, y]
    Amplitude is equivalent to ((frequency % 2) / 1000.0) and will define the maximum pixel displacement.
    The highest the frequency the lowest the heat wave  
    
    e.g : 
    surface = heatwave_array24_horiz_c(numpy.asarray(new_array).transpose(1, 0, 2),
            alpha, heat_value, (frequency % 2) / 1000.0, attenuation=0.10)
            
    :param rgb_array: numpy.ndarray or MemoryViewSlice, array shape (w, h, 3) containing RGB values
    :param mask_array: numpy.ndarray or  MemoryViewSlice shape (w, h) containing alpha values
    :param frequency: float; increment value. The highest the frequency the lowest the heat wave  
    :param amplitude: float; variable amplitude. Max amplitude is 10e-3 * 255 = 2.55 
    when alpha is 255 otherwise 10e-3 * alpha.
    :param attenuation: float; default 0.10
    :param threshold: unsigned char; Compare the alpha value with the threshold.
     if alpha value > threshold, apply the displacement to the texture otherwise no change
    :return: Return a pygame.Surface 24 bit format 
    """


    cdef int w, h
    w, h = (<object>rgb_array).shape[:2]

    cdef:
        unsigned char [:, :, ::1] new_array = empty((h, w, 3), dtype=numpy.uint8)
        int x = 0, y = 0, xx, yy
        float distortion


    with nogil:
        for x in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for y in range(h):
                distortion = sin(x * attenuation + frequency) * amplitude * mask_array[x, y]

                xx = <int>(x  + distortion + rand() * 0.00002)
                if xx > w - 1:
                    xx = w - 1
                if xx < 0:
                    xx = 0

                if mask_array[x, y] > threshold:
                    new_array[y, x, 0] = rgb_array[xx, y, 0]
                    new_array[y, x, 1] = rgb_array[xx, y, 1]
                    new_array[y, x, 2] = rgb_array[xx, y, 2]
                else:
                    new_array[y, x, 0] = rgb_array[x, y, 0]
                    new_array[y, x, 1] = rgb_array[x, y, 1]
                    new_array[y, x, 2] = rgb_array[x, y, 2]

    return pygame.image.frombuffer(new_array, (w, h), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef heatwave_array32_horiz_c(unsigned char [:, :, :] rgba_array,
                            unsigned char [:, :] mask_array,
                            float frequency, float amplitude, float attenuation=0.10,
                            unsigned char threshold=64):
    """
    
    DISTORTION EQUATION: 
    distortion = sin(x * attenuation + frequency) * amplitude * mask_array[x, y]
    Amplitude is equivalent to ((frequency % 2) / 1000.0) and will define the maximum pixel displacement.
    The highest the frequency the lowest the heat wave  
    
    e.g : 
    surface = heatwave_array32_horiz_c(numpy.asarray(new_array).transpose(1, 0, 2),
            alpha, heat_value, (frequency % 2) / 1000.0, attenuation=0.10)
            
    :param rgba_array: numpy.ndarray or MemoryViewSlice, array shape (w, h, 4) containing RGBA values
    :param mask_array: numpy.ndarray or  MemoryViewSlice shape (w, h) containing alpha values
    :param frequency: float; increment value. The highest the frequency the lowest the heat wave  
    :param amplitude: float; variable amplitude. Max amplitude is 10e-3 * 255 = 2.55 
    when alpha is 255 otherwise 10e-3 * alpha.
    :param attenuation: float; default 0.10
    :param threshold: unsigned char; Compare the alpha value with the threshold.
     if alpha value > threshold, apply the displacement to the texture otherwise no change
    :return: Return a pygame.Surface 32 bit format with per-pixel information
    """


    cdef int w, h
    w, h = (<object>rgba_array).shape[:2]

    cdef:
        unsigned char [:, :, ::1] new_array = empty((h, w, 4), dtype=numpy.uint8)
        int x = 0, y = 0, xx, yy
        float distortion


    with nogil:
        for x in prange(0, w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for y in range(h):
                distortion = sin(x * attenuation + frequency) * amplitude * mask_array[x, y]

                xx = <int>(x  + distortion + rand() * 0.00002)
                if xx > w - 1:
                    xx = w - 1
                if xx < 0:
                    xx = 0

                if mask_array[x, y] > threshold:
                    new_array[y, x, 0] = rgba_array[xx, y, 0]
                    new_array[y, x, 1] = rgba_array[xx, y, 1]
                    new_array[y, x, 2] = rgba_array[xx, y, 2]
                else:
                    new_array[y, x, 0] = rgba_array[x, y, 0]
                    new_array[y, x, 1] = rgba_array[x, y, 1]
                    new_array[y, x, 2] = rgba_array[x, y, 2]

    return pygame.image.frombuffer(new_array, (w, h), 'RGBA')





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef heatwave_buffer24_horiz_c(unsigned char [:] rgb_buffer,
                               unsigned char [:, :] mask_buffer,
                               int width, int height,
                               float frequency, float amplitude, float attenuation=0.10,
                               unsigned char threshold=64):
    """

    :param rgb_buffer:
    :param mask_buffer:
    :param width:
    :param height:
    :param frequency:
    :param amplitude:
    :param attenuation:
    :param threshold:
    :return:
    """

    cdef int b_length
    b_length = len(<object>rgb_buffer)

    cdef:
        unsigned char [:] new_array = empty(b_length, dtype=numpy.uint8)
        int i=0, index, xx
        float distortion
        xyz v;

    with nogil:
        for i in range(0, b_length, 3): # , schedule=SCHEDULE, num_threads=THREAD_NUMBER):

            # buffer to 3d indexing
            v = to3d_c(index=i, width=width, depth=3) # --> point to the red

            distortion = sin(v.x * attenuation + frequency) * amplitude * mask_buffer[v.x, v.y]

            xx = <int>(v.x  + distortion + rand() * 0.00002)

            if xx > width-1:
                xx = width-1
            if xx < 0:
                xx = 0

            # 3d indexing to 1d buffer
            index = to1d_c(x=xx, y=v.y, z=0, width=width, depth=3)

            if mask_buffer[v.x, v.y] > threshold:
                new_array[i  ] = rgb_buffer[index  ]
                new_array[i+1] = rgb_buffer[index+1]
                new_array[i+2] = rgb_buffer[index+2]
            else:
                new_array[i  ] = rgb_buffer[i  ]
                new_array[i+1] = rgb_buffer[i+1]
                new_array[i+2] = rgb_buffer[i+2]

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef heatwave_buffer24_vertical_c(unsigned char [:] rgb_buffer,
                                  unsigned char [:, :] mask_buffer,
                                  int width, int height,
                                  float frequency, float amplitude, float attenuation=0.10,
                                  unsigned char threshold=64):
    """

    :param rgb_buffer:
    :param mask_buffer:
    :param width:
    :param height:
    :param frequency:
    :param amplitude:
    :param attenuation:
    :param threshold:
    :return:
    """

    cdef int b_length
    b_length = len(<object>rgb_buffer)

    cdef:
        unsigned char [:] new_array = empty(b_length, dtype=numpy.uint8)
        int i=0, index, yy
        float distortion
        xyz v;

    with nogil:
        for i in range(0, b_length, 3): # , schedule=SCHEDULE, num_threads=THREAD_NUMBER):

            # buffer to 3d indexing
            v = to3d_c(index=i, width=width, depth=3) # --> point to the red

            distortion = sin(v.x * attenuation + frequency) * amplitude * mask_buffer[v.x, v.y]

            yy = <int>(v.y  + distortion + rand() * 0.00002)

            if yy > height-1:
                yy = height-1
            if yy < 0:
                yy = 0

            # 3d indexing to 1d buffer
            index = to1d_c(x=v.x, y=yy, z=0, width=width, depth=3)

            if mask_buffer[v.x, v.y] > threshold:
                new_array[i  ] = rgb_buffer[index  ]
                new_array[i+1] = rgb_buffer[index+1]
                new_array[i+2] = rgb_buffer[index+2]
            else:
                new_array[i  ] = rgb_buffer[i  ]
                new_array[i+1] = rgb_buffer[i+1]
                new_array[i+2] = rgb_buffer[i+2]

    return pygame.image.frombuffer(new_array, (width, height), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef greyscale_3d_to_2d_c(array):
    """
    Create a greyscale 2d array (w, h)
    
    :param array: numpy.ndarray; 3d array type (w, h, 3) containing RGB values 
    :return: return a 2d numpy.ndarray type (w, h) containing greyscale value 
    
    NOTE:
        if you intend to convert the output greyscale surface to a pygame.Surface using
        pygame.surfarray.make_surface(array), be aware that the resulting will not 
        be a greyscale surface. In order to create a valid greyscale surface from the output 
        array, you need first to convert the 2d array into a 3d array then create a surface. 
    """

    cdef int w_, h_

    try:
        w_, h_ = array.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        int w = w_, h = h_
        unsigned char[:, :, :] rgb_array = array
        unsigned char[:, ::1] rgb_out = empty((w, h), dtype=uint8)
        int red, green, blue
        int i=0, j=0
        unsigned char c

    with nogil:
        for i in prange(w, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for j in range(h):
                red, green, blue = rgb_array[i, j, 0], rgb_array[i, j, 1], rgb_array[i, j, 2]
                c = <unsigned char>((red + green + blue) * 0.3333)
                rgb_out[i, j] = c
    return asarray(rgb_out)
