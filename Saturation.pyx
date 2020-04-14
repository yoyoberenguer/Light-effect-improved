#cython: boundscheck=False, wraparound=False, nonecheck=False, optimize.use_switch=True

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

# NUMPY IS REQUIRED
try:
    import numpy
    from numpy import ndarray, zeros, empty, uint8, int32, float64, float32, dstack, full, ones,\
    asarray, ascontiguousarray
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

# CYTHON IS REQUIRED
try:
    cimport cython
    from cython.parallel cimport prange
except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d
    from pygame.image import frombuffer

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")


# MAPPING LIBRARY IS REQUIRED
try:
    import MAPPING
except ImportError:
    raise ImportError("\n<MAPPING> library is missing on your system.")

if MAPPING.__version__ < 1.00:
        raise ValueError("\nMAPPING version 1.00 is required.")
print("\nMAPPING version: %s " % MAPPING.__version__)

try:
    from MAPPING cimport xyz, to1d_c, to3d_c, vfb_rgb_c, vfb_c
except ImportError:
    raise ImportError("\n<MAPPING> Cannot import methods.")


from libc.stdio cimport printf
from libc.stdlib cimport free, rand
from libc.math cimport round, fmin, fmax, sin

__version__ = 1.00

DEF OPENMP = True

if OPENMP:
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


cdef extern from 'library.c' nogil:
    double * rgb_to_hsl(double r, double g, double b);
    double * hsl_to_rgb(double h, double s, double l);


# C-structure to store 3d array index values
cdef struct xyz:
    int x;
    int y;
    int z;

# ----------------- INTERFACE --------------------


# ------------ SATURATION
# APPLY SATURATION TO AN RGB BUFFER USING A MASK(COMPATIBLE SURFACE 24 BIT)
def saturation_buffer_mask(array_, shift_, mask_)->Surface:
    return saturation_buffer_mask_c(array_, shift_, mask_)

# TODO: CREATE SATURATION_BUFFER_MASK FOR 32 BIT

# APPLY SATURATION TO AN RGB ARRAY USING A MASK(COMPATIBLE SURFACE 24 BIT)
def saturation_array24_mask(array_, shift_, mask_, swap_row_column=False)->Surface:
    return saturation_array24_mask_c(array_, shift_, mask_, swap_row_column)

# APPLY SATURATION TO AN RGBA ARRAY USING A MASK(COMPATIBLE SURFACE 32 BIT)
def saturation_array32_mask(array_, alpha_, shift_, mask_, swap_row_column=False)->Surface:
    return saturation_array32_mask_c(array_, alpha_, shift_, mask_, swap_row_column)

# APPLY SATURATION TO AN RGB ARRAY
def saturation_array24(array_, shift_, swap_row_column=False):
    return saturation_array24_c(array_, shift_, swap_row_column)

# APPLY SATURATION TO AN RGBA ARRAY
def saturation_array32(array_, alpha_, shift_, swap_row_column=False):
    return saturation_array32_c(array_, alpha_, shift_, swap_row_column)
# ----------------IMPLEMENTATION -----------------


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef saturation_array24_mask_c(unsigned char [:, :, :] array_,
                               float shift_, float [:, :] mask_array, bint swap_row_column):
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
    :param swap_row_column: swap row and column values (only apply to array_) 
    :return: a pygame.Surface 24-bit without per-pixel information 

    """

    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    try:
        if swap_row_column:
            height, width = numpy.asarray(array_).shape[:2]
        else:
            width, height = numpy.asarray(array_).shape[:2]
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
                               float shift_, float [:, :] mask_array=None, bint swap_row_column=False):
    """
    Change the saturation level of a pygame.Surface (compatible with 32bit only).
    Transform RGBA model into HSL model and <shift_> saturation value.
    Optional mask_array to determine area to be modified.
    The mask_array should be a 2d array filled with float values 
    
    :param swap_row_column: swap row and column values (only apply to array_) 
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
        if swap_row_column:
            height, width = (<object>array_).shape[:2]
        else:
            width, height = (<object>array_).shape[:2]
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
cdef saturation_array24_c(unsigned char [:, :, :] array_, float shift_, bint swap_row_column):
    """
    Change the saturation level of an array / pygame.Surface (compatible with 24-bit format image)
    Transform RGB model into HSL model and add <shift_> value to the saturation 
    
    :param swap_row_column: swap row and column values (only apply to array_) 
    :param array_: numpy.ndarray (w, h, 3) uint8 representing 24 bit format surface
    :param shift_: Value must be in range [-1.0 ... 1.0], negative values decrease saturation
    :return: a pygame.Surface 24-bit without per-pixel information 
    """

    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height
    try:
        if swap_row_column:
            height, width = (<object>array_).shape[:2]
        else:
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
                          unsigned char [:, :] alpha_, float shift_, bint swap_row_column):
    """
    Change the saturation level of an array/ pygame.Surface (compatible with 32-bit format image only)
    Transform RGB model into HSL model and add <shift_> value to the saturation 
        
    :param swap_row_column: swap row and column values (only apply to array_) 
    :param array_: numpy.ndarray shapes (w, h, 4) representing a pygame Surface 32 bit format
    :param alpha_: numpy.ndarray shapes (w, h) containing all alpha values
    :param shift_: Value must be in range [-1.0 ... 1.0], negative values decrease saturation  
    :return: a pygame.Surface 32-bit with per-pixel information 
    """

    assert -1.0 <= shift_ <= 1.0, 'Argument shift_ must be in range [-1.0 .. 1.0].'

    cdef int width, height, alpha_width, alpha_height

    try:
        if swap_row_column:
            height, width = (<object>array_).shape[:2]
        else:
            width, height = (<object>array_).shape[:2]
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
cdef saturation_buffer_mask_c(unsigned char [:] buffer_,
                              float shift_, float [:, :] mask_array):
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