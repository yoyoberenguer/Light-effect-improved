/* C implementation */


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <time.h>


// Calculate distance between two pixels (used by bilateral filters).
double distance (double x1, double y1, double x2, double y2);
// Gaussian function (used by bilateral filters).
double gaussian (double v, double sigma);

// From a given set of RGB values, determines min and max.
double fmax_rgb_value(double red, double green, double blue);
double fmin_rgb_value(double red, double green, double blue);
unsigned char max_rgb_value(unsigned char red, unsigned char green, unsigned char blue);
unsigned char min_rgb_value(unsigned char red, unsigned char green, unsigned char blue);

// Convert RGB color model into HSV and reciprocally
double * rgb_to_hsv(double r, double g, double b);
double * hsv_to_rgb(double h, double s, double v);

// Convert RGB color model into HSL and reciprocally
double hue_to_rgb(double m1, double m2, double hue);
double * rgb_to_hsl(double r, double g, double b);
double * hsl_to_rgb(double h, double s, double l);

// Quicksort algorithm
void swap(int* a, int* b);
int partition (int arr[], int low, int high);
int * quickSort(int arr[], int low, int high);


#define M_PI 3.14159265358979323846
#define ONE_SIX 1.0/6.0
#define ONE_THIRD 1.0 / 3.0
#define TWO_THIRD 2.0 / 3.0
#define ONE_255 1.0/255.0
#define ONE_360 1.0/360.0

#define cmax(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


unsigned char umax_ (unsigned char a, unsigned char b)
{
  if (a > b) {
  return a;
}
  else return b;
}


int imax_ (int a, int b)
{
  if (a > b) {
  return a;
}
  else return b;
}

float fmax_ (float a, float b)
{
  if (a > b) {
  return a;
}
  else return b;
}


double distance (double x1, double y1, double x2, double y2)
{
  return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
}

double gaussian (double v, double sigma)
{
  return (1.0 / (2.0 * M_PI * (sigma * sigma))) * exp(-(v * v ) / (2.0 * sigma * sigma));
}

// All inputs have to be double precision (python float) in range [0.0 ... 255.0]
// Output: return the maximum value from given RGB values (double precision).
inline double fmax_rgb_value(double red, double green, double blue)
{
    if (red>green){
        if (red>blue) {
		    return red;}
		else {
		    return blue;}
    }
    else if (green>blue){
	    return green;}
        else {
	        return blue;}
}

// All inputs have to be double precision (python float) in range [0.0 ... 255.0]
// Output: return the minimum value from given RGB values (double precision).
inline double fmin_rgb_value(double red, double green, double blue)
{
    if (red<green){
        if (red<blue){
            return red;}
    else{
	    return blue;}
    }
    else if (green<blue){
	    return green;}
    else{
	    return blue;}
}


// All inputs have to be double precision (python float) in range [0.0 ... 255.0]
// Output: return the maximum value from given RGB values (double precision).
inline unsigned char max_rgb_value(unsigned char red, unsigned char green, unsigned char blue)
{
    if (red>green){
        if (red>blue) {
		    return red;}
		else {
		    return blue;}
    }
    else if (green>blue){
	    return green;}
        else {
	        return blue;}
}

// All inputs have to be double precision (python float) in range [0.0 ... 255.0]
// Output: return the minimum value from given RGB values (double precision).
inline unsigned char min_rgb_value(unsigned char red, unsigned char green, unsigned char blue)
{
    if (red<green){
        if (red<blue){
            return red;}
    else{
	    return blue;}
    }
    else if (green<blue){
	    return green;}
    else{
	    return blue;}
}


// Convert RGB color model into HSV model (Hue, Saturation, Value)
// all colors inputs have to be double precision (RGB normalized values),
// (python float) in range [0.0 ... 1.0]
// outputs is a C array containing 3 values, HSV (double precision)
// to convert in % do the following:
// h = h * 360.0
// s = s * 100.0
// v = v * 100.0

inline double * rgb_to_hsv(double r, double g, double b)
{
    // check if all inputs are normalized
    assert ((0.0<=r) <= 1.0);
    assert ((0.0<=g) <= 1.0);
    assert ((0.0<=b) <= 1.0);

    double mx, mn;
    double h, df, s, v, df_;
    double *hsv = malloc (sizeof (double) * 3);
    // Check if the memory has been successfully
    // allocated by malloc or not
    if (hsv == NULL) {
        printf("Memory not allocated.\n");
        exit(0);
    }

    mx = fmax_rgb_value(r, g, b);
    mn = fmin_rgb_value(r, g, b);

    df = mx-mn;
    df_ = 1.0/df;
    if (mx == mn)
    {
        h = 0.0;}
    // The conversion to (int) approximate the final result
    else if (mx == r){
	    h = fmod(60.0 * ((g-b) * df_) + 360.0, 360);
	}
    else if (mx == g){
	    h = fmod(60.0 * ((b-r) * df_) + 120.0, 360);
	}
    else if (mx == b){
	    h = fmod(60.0 * ((r-g) * df_) + 240.0, 360);
    }
    if (mx == 0){
        s = 0.0;
    }
    else{
        s = df/mx;
    }
    v = mx;
    hsv[0] = h * ONE_360;
    hsv[1] = s;
    hsv[2] = v;
    return hsv;
}

// Convert HSV color model into RGB (red, green, blue)
// all inputs have to be double precision, (python float) in range [0.0 ... 1.0]
// outputs is a C array containing RGB values (double precision) normalized.
// to convert for a pixel colors
// r = r * 255.0
// g = g * 255.0
// b = b * 255.0

inline double * hsv_to_rgb(double h, double s, double v)
{
    // check if all inputs are normalized
    assert ((0.0<= h) <= 1.0);
    assert ((0.0<= s) <= 1.0);
    assert ((0.0<= v) <= 1.0);

    int i;
    double f, p, q, t;
    double *rgb = malloc (sizeof (double) * 3);
    // Check if the memory has been successfully
    // allocated by malloc or not
    if (rgb == NULL) {
        printf("Memory not allocated.\n");
        exit(0);
    }

    if (s == 0.0){
        rgb[0] = v;
        rgb[1] = v;
        rgb[2] = v;
        return rgb;
    }

    i = (int)(h*6.0);

    f = (h*6.0) - i;
    p = v*(1.0 - s);
    q = v*(1.0 - s*f);
    t = v*(1.0 - s*(1.0-f));
    i = i%6;

    if (i == 0){
        rgb[0] = v;
        rgb[1] = t;
        rgb[2] = p;
        return rgb;
    }
    else if (i == 1){
        rgb[0] = q;
        rgb[1] = v;
        rgb[2] = p;
        return rgb;
    }
    else if (i == 2){
        rgb[0] = p;
        rgb[1] = v;
        rgb[2] = t;
        return rgb;
    }
    else if (i == 3){
        rgb[0] = p;
        rgb[1] = q;
        rgb[2] = v;
        return rgb;
    }
    else if (i == 4){
        rgb[0] = t;
        rgb[1] = p;
        rgb[2] = v;
        return rgb;
    }
    else if (i == 5){
        rgb[0] = v;
        rgb[1] = p;
        rgb[2] = q;
        return rgb;
    }
    return rgb;
}



// HSL: Hue, Saturation, Luminance
// H: position in the spectrum
// L: color lightness
// S: color saturation
// all inputs have to be double precision, (python float) in range [0.0 ... 1.0]
// outputs is a C array containing HSL values (double precision) normalized.
// h (°) = h * 360
// s (%) = s * 100
// l (%) = l * 100
inline double * rgb_to_hsl(double r, double g, double b)
{
    // check if all inputs are normalized
    assert ((0.0<= r) <= 1.0);
    assert ((0.0<= g) <= 1.0);
    assert ((0.0<= b) <= 1.0);

    double *hsl = malloc (sizeof (double)* 3);
    // Check if the memory has been successfully
    // allocated by malloc or not
    if (hsl == NULL) {
        printf("Memory not allocated.\n");
        exit(0);
    }
    double cmax=0, cmin=0, delta=0, t;
    cmax = fmax_rgb_value(r, g, b);
    cmin = fmin_rgb_value(r, g, b);
    delta = (cmax - cmin);

    double h, l, s;
    l = (cmax + cmin) / 2.0;

    if (delta == 0) {
    h = 0;
    s = 0;
    }
    else {
    	  if (cmax == r){
    	        t = (g - b) / delta;
    	        if ((fabs(t) > 6.0) && (t > 0.0)) {
                  t = fmod(t, 6.0);
                }
                else if (t < 0.0){
                t = 6.0 - fabs(t);
                }

	            h = 60.0 * t;
          }
    	  else if (cmax == g){
                h = 60.0 * (((b - r) / delta) + 2.0);
          }

    	  else if (cmax == b){
    	        h = 60.0 * (((r - g) / delta) + 4.0);
          }

    	  if (l <=0.5) {
	            s=(delta/(cmax + cmin));
	      }
	  else {
	        s=(delta/(2.0 - cmax - cmin));
	  }
    }

    hsl[0] = h * ONE_360;
    hsl[1] = s;
    hsl[2] = l;
    // printf("\n %f, %f, %f", hsl[0], hsl[1], hsl[2]);
    return hsl;


}


inline double hue_to_rgb(double m1, double m2, double h)
{
    if ((fabs(h) > 1.0) && (h > 0.0)) {
      h = fmod(h, 1.0);
    }
    else if (h < 0.0){
    h = 1.0 - fabs(h);
    }

    if (h < ONE_SIX){
        return m1 + (m2 - m1) * h * 6.0;
    }
    if (h < 0.5){
        return m2;
    }
    if (h < TWO_THIRD){
        return m1 + ( m2 - m1 ) * (TWO_THIRD - h) * 6.0;
    }
    return m1;
}


// Convert HSL color model into RGB (red, green, blue)
// all inputs have to be double precision, (python float) in range [0.0 ... 1.0]
// outputs is a C array containing RGB values (double precision) normalized.
inline double * hsl_to_rgb(double h, double s, double l)
{
    double *rgb = malloc (sizeof (double) * 3);
    // Check if the memory has been successfully
    // allocated by malloc or not
    if (rgb == NULL) {
        printf("Memory not allocated.\n");
        exit(0);
    }

    double m2=0, m1=0;

    if (s == 0.0){
        rgb[0] = l;
        rgb[1] = l;
        rgb[2] = l;
        return rgb;
    }
    if (l <= 0.5){
        m2 = l * (1.0 + s);
    }
    else{
        m2 = l + s - (l * s);
    }
    m1 = 2.0 * l - m2;

    rgb[0] = hue_to_rgb(m1, m2, (h + ONE_THIRD));
    rgb[1] = hue_to_rgb(m1, m2, h);
    rgb[2] = hue_to_rgb(m1, m2, (h - ONE_THIRD));
    return rgb;
}

// A utility function to swap two elements
inline void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
	array, and places all smaller (smaller than pivot)
to left of pivot and all greater elements to right
of pivot */
inline int partition (int arr[], int low, int high)
{
	int pivot = arr[high]; // pivot
	int i = (low - 1); // Index of smaller element

	for (int j = low; j <= high- 1; j++)
	{
		// If current element is smaller than the pivot
		if (arr[j] < pivot)
		{
			i++; // increment index of smaller element
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	return (i + 1);
}

/* The main function that implements QuickSort
arr[] --> Array to be sorted,
low --> Starting index,
high --> Ending index */
int * quickSort(int arr[], int low, int high)
{
	if (low < high)
	{
		/* pi is partitioning index, arr[p] is now
		at right place */
		int pi = partition(arr, low, high);

		// Separately sort elements before
		// partition and after partition
		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
return arr;
}

/* Function to print an array */
void printArray(int arr[], int size)
{
	int i;
	for (i=0; i < size; i++)
		printf("%d ", arr[i]);
	printf("n");
}


int main(){
//double *array;
//double *arr;
//double h, l, s;
//double r, g, b;
//int i = 0, j = 0, k = 0;
//
//int n = 1000000;
//double *ptr;
//clock_t begin = clock();
//
///* here, do your time-consuming job */
//for (i=0; i<=n; ++i){
//ptr = rgb_to_hsl(25.0/255.0, 60.0/255.0, 128.0/255.0);
//}
//clock_t end = clock();
//double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//printf("\ntotal time %f :", time_spent);
//
//printf("\nTesting algorithm(s).");
//n = 0;
//
//for (i=0; i<256; i++){
//    for (j=0; j<256; j++){
//        for (k=0; k<256; k++){
//
//            array = rgb_to_hsl(i/255.0, j/255.0, k/255.0);
//            h = array[0];
//            s = array[1];
//            l = array[2];
//            free(array);
//            arr = hsl_to_rgb(h, s, l);
//            r = round(arr[0] * 255.0);
//            g = round(arr[1] * 255.0);
//            b = round(arr[2] * 255.0);
//	        free(arr);
//            // printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
//            // printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
//            // printf("\n %f, %f, %f ", h, l, s);
//
//            if (abs(i - r) > 0.1) {
//                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
//                    printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
//                printf("\n %f, %f, %f ", h, l, s);
//                        n+=1;
//                return -1;
//            }
//            if (abs(j - g) > 0.1){
//                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
//                    printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
//                printf("\n %f, %f, %f ", h, l, s);
//                        n+=1;
//                return -1;
//            }
//
//            if (abs(k - b) > 0.1){
//                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
//                printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
//                printf("\n %f, %f, %f ", h, l, s);
//                n+=1;
//		        return -1;
//
//            }
//
//            }
//
//        }
//    }
//
//
//printf("\nError(s) found n=%i", n);
return 0;
}

/*

int main ()
{
double *ar;
double *ar1;
int i, j, k;
double r, g, b;
double h, s, v;

int n = 1000000;
double *ptr;
clock_t begin = clock();


for (i=0; i<=n; ++i){
ptr = rgb_to_hsv(25.0/255.0, 60.0/255.0, 128.0/255.0);
}
clock_t end = clock();
double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
printf("\ntotal time %f :", time_spent);

printf("\nTesting algorithm(s).");
n = 0;
for (i=0; i<256; i++){
    for (j=0; j<256; j++){
        for (k=0; k<256; k++){
            ar = rgb_to_hsv((double)i/255, (double)j/255, (double)k/255);
            h=ar[0];
            s=ar[1];
            v=ar[2];
	        free(ar);
            ar1 = hsv_to_rgb(h, s, v);
            r = round(ar1[0] * 255.0);
            g = round(ar1[1] * 255.0);
            b = round(ar1[2] * 255.0);
   	        free(ar1);
            // printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
            // printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
            // printf("\n %f, %f, %f ", h, s, v);

            if (abs(i - r) > 0.1) {
                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
                    printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
                printf("\n %f, %f, %f ", h, s, v);
                        n+=1;
                return -1;
            }
            if (abs(j - g) > 0.1){
                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
                    printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
                printf("\n %f, %f, %f ", h, s, v);
                        n+=1;
                return -1;
            }

            if (abs(k - b) > 0.1){
                printf("\n\nRGB VALUES:R:%i G:%i B:%i ", i, j, k);
                printf("\nRGB VALUES:R:%f G:%f B:%f ", r, g, b);
                printf("\n %f, %f, %f ", h, s, v);
                n+=1;
		        return -1;

            }
        }
    }
}
printf("\nError(s) found. %i ", n);

return 0;
}

*/