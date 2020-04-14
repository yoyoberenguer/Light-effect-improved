
cdef extern from 'library.c' nogil:
    double * rgb_to_hsl(double r, double g, double b);
    double * hsl_to_rgb(double h, double s, double l);


# C-structure to store 3d array index values
cdef struct xyz:
    int x;
    int y;
    int z;


# APPLY SATURATION TO AN RGB BUFFER USING A MASK(COMPATIBLE SURFACE 24 BIT)
cdef saturation_buffer_mask_c(unsigned char [:] buffer_,
                              float shift_, float [:, :] mask_array)

# TODO: CREATE SATURATION_BUFFER_MASK FOR 32 BIT

# APPLY SATURATION TO AN RGB ARRAY USING A MASK(COMPATIBLE SURFACE 24 BIT)
cdef saturation_array24_mask_c(unsigned char [:, :, :] array_,
                               float shift_, float [:, :] mask_array, bint swap_row_column)

# APPLY SATURATION TO AN RGBA ARRAY USING A MASK(COMPATIBLE SURFACE 32 BIT)
cdef saturation_array32_mask_c(unsigned char [:, :, :] array_, unsigned char [:, :] alpha_,
                               float shift_, float [:, :] mask_array=*, bint swap_row_column=*)

# APPLY SATURATION TO AN RGB ARRAY
cdef saturation_array24_c(unsigned char [:, :, :] array_, float shift_, bint swap_row_column)

# APPLY SATURATION TO AN RGBA ARRAY
cdef saturation_array32_c(unsigned char [:, :, :] array_,
                          unsigned char [:, :] alpha_, float shift_, bint swap_row_column)
