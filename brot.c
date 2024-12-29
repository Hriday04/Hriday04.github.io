#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>

// Function to create a 3D array for the image
unsigned char ***create_base(int size) {
    unsigned char ***base = malloc(size * sizeof(unsigned char **));
    for (int i = 0; i < size; i++) {
        base[i] = malloc(size * sizeof(unsigned char *));
        for (int j = 0; j < size; j++) {
            base[i][j] = calloc(3, sizeof(unsigned char)); // Initialize to 0
        }
    }
    return base;
}

// Function to calculate the next value in the sequence
double complex m_seq(double complex z_n, double complex c) {
    return z_n * z_n + c;
}

// Function to convert a complex number to bitmap coordinates
void c2b(double complex c, int size, int *x, int *y) {
    double scale = 4.0 / size;
    *x = (creal(c) + 2) / scale;
    *y = (cimag(c) + 2) / scale;
}

// Function to convert bitmap coordinates to a complex number
double complex b2c(int size, int x, int y) {
    double scale = 4.0 / size;
    double a = x * scale - 2;
    double b = y * scale - 2;
    return a + b * I;
}

bool escapes(double complex c, int iters) {
    double complex z_n = 0; // Start with z_0 = 0
    for (int i = 0; i < iters; i++) {
        z_n = m_seq(z_n, c);
        if (cabs(z_n) > 2) { // Check if the magnitude exceeds 2
            return true;
        }
    }
    return false;
}

// Modified Function to track the trajectory and visualize paths of points that escape
void track_trajectory(unsigned char ***base, int size, double complex c, int max_iters) {
    if (!escapes(c, max_iters)) {
        // If the point does not escape, don't track its trajectory
        return;
    }

    double complex z = 0;
    int iter = 0;
    while (cabs(z) <= 2.0 && iter < max_iters) {
        z = m_seq(z, c);
        int x, y;
        c2b(z, size, &x, &y); // Convert each point in trajectory to bitmap coordinates
        if (x >= 0 && x < size && y >= 0 && y < size) {
            // Increment color channels to visualize the trajectory
            // Increase the increment to make the image brighter
            base[y][x][0] = fmin(255, base[y][x][0] + 20); // Increment red channel by a larger amount
            // Green Blue
            //base[y][x][1] = fmin(255, base[y][x][1] + 20); // Increment green channel by a larger amount
            base[y][x][2] = fmin(255, base[y][x][2] + 20); // Increment blue channel by a larger amount
        }
        iter++;
    }
}


// Function to fill the 3D array with colors based on trajectories
void get_colors(unsigned char ***base, int size, int iters) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double complex c = b2c(size, i, j);
            track_trajectory(base, size, c, iters);
        }
    }
}
// Main function to generate the Buddhabrot image
void make_brot(int size, int iters) {
    unsigned char ***base = create_base(size);
    get_colors(base, size, iters);

    FILE *fp = fopen("buddhabrot2ee4.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fwrite(base[i][j], 1, 3, fp);
        }
    }
    fclose(fp);

    // Free the allocated memory
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            free(base[i][j]);
        }
        free(base[i]);
    }
    free(base);
}

int main() {
    int size = 100000000000000000000000000000000000000000000000000000; // Image size, adjust as needed
    int iters = 10000000000; // Maximum iterations, adjust as needed
    make_brot(size, iters); // Generate the Buddhabrot image
    return 0;
}
