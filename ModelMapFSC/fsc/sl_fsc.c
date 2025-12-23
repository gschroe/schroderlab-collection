/* 
schroderlab-collection: Tool collection for the processing of Cryo-EM Datasets
Copyright (C) 2025 Gunnar Schröder

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>

/* MRC header structure.
   This structure covers the first 96 bytes of significant header data.
   The remaining bytes (to make up 1024) are stored in the extra field.
*/
typedef struct {
    int nx, ny, nz;      /* number of columns, rows, sections */
    int mode;            /* data type (2 for 32-bit floats) */
    int nxstart, nystart, nzstart;
    int mx, my, mz;
    float xlen, ylen, zlen; /* cell dimensions in Å */
    float alpha, beta, gamma;
    int mapc, mapr, maps;
    float dmin, dmax, dmean;
    int ispg, nsymbt;
    char extra[1024 - 96]; /* remaining bytes to complete 1024-byte header */
} MRCHeader;

/* Function to read an MRC file.
   Reads the 1024-byte header and then the density data (assumed to be floats).
   Returns a pointer to an array of floats with the density values.
*/
float* read_mrc(const char *filename, MRCHeader *header) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror(filename);
        return NULL;
    }
    if (fread(header, sizeof(MRCHeader), 1, fp) != 1) {
        fprintf(stderr, "Error reading header from %s\n", filename);
        fclose(fp);
        return NULL;
    }
    int nvoxels = header->nx * header->ny * header->nz;
    float *data = (float*) malloc(nvoxels * sizeof(float));
    if (!data) {
        perror("malloc");
        fclose(fp);
        return NULL;
    }
    if (fread(data, sizeof(float), nvoxels, fp) != nvoxels) {
        fprintf(stderr, "Error reading volume data from %s\n", filename);
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    return data;
}

void log_execution(int argc, char *argv[]) {
    FILE *logfile = fopen(".dxfsc.log", "a");
    if (!logfile) {
        return;
    }
    
    time_t rawtime;
    struct tm *timeinfo;
    char timestamp[80];
    
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    fprintf(logfile, "[%s] dxfsc executed with arguments: ", timestamp);
    for (int i = 0; i < argc; i++) {
        fprintf(logfile, "%s", argv[i]);
        if (i < argc - 1) fprintf(logfile, " ");
    }
    fprintf(logfile, "\n");
    fclose(logfile);
}

int main(int argc, char *argv[]) {
    log_execution(argc, argv);
    
    if (argc < 3) {
        printf("Usage: %s map1.mrc map2.mrc [-mask mask.mrc] [-o output_fsc.txt]\n", argv[0]);
        return 1;
    }

    const char *file1 = argv[1];
    const char *file2 = argv[2];
    const char *mask_file = NULL;
    const char *output_file = "fsc.txt"; // default output file name

    // Parse optional command-line arguments.
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-mask") == 0) {
            if (i + 1 < argc) {
                mask_file = argv[i+1];
                i++;
            } else {
                fprintf(stderr, "Error: Missing mask file after -mask\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                output_file = argv[i+1];
                i++;
            } else {
                fprintf(stderr, "Error: Missing output file name after -o\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Warning: Ignoring unrecognized argument %s\n", argv[i]);
        }
    }

    MRCHeader header1, header2;
    float *map1_f = read_mrc(file1, &header1);
    float *map2_f = read_mrc(file2, &header2);
    if (!map1_f || !map2_f) {
        fprintf(stderr, "Error reading MRC files.\n");
        return 1;
    }

    // Check that both maps have the same dimensions.
    if (header1.nx != header2.nx || header1.ny != header2.ny || header1.nz != header2.nz) {
        fprintf(stderr, "Error: Map dimensions do not match!\n");
        free(map1_f);
        free(map2_f);
        return 1;
    }

    int nx = header1.nx;
    int ny = header1.ny;
    int nz = header1.nz;
    int nvoxels = nx * ny * nz;

    /* Convert the float data to double for FFTW */
    double *map1 = (double*) malloc(nvoxels * sizeof(double));
    double *map2 = (double*) malloc(nvoxels * sizeof(double));
    if (!map1 || !map2) {
        perror("malloc");
        free(map1_f);
        free(map2_f);
        return 1;
    }
    for (int i = 0; i < nvoxels; i++) {
        map1[i] = (double) map1_f[i];
        map2[i] = (double) map2_f[i];
    }
    free(map1_f);
    free(map2_f);

    // If a mask file is provided, read it and multiply element-wise to map1 and map2.
    if (mask_file) {
        MRCHeader mask_header;
        float *mask_f = read_mrc(mask_file, &mask_header);
        if (!mask_f) {
            fprintf(stderr, "Error reading mask file %s\n", mask_file);
            free(map1);
            free(map2);
            return 1;
        }
        // Check dimensions of mask against map dimensions.
        if (mask_header.nx != nx || mask_header.ny != ny || mask_header.nz != nz) {
            fprintf(stderr, "Error: Mask dimensions do not match map dimensions!\n");
            free(mask_f);
            free(map1);
            free(map2);
            return 1;
        }
        for (int i = 0; i < nvoxels; i++) {
            double mval = (double) mask_f[i];
            map1[i] *= mval;
            map2[i] *= mval;
        }
        free(mask_f);
    }

    /* Set up FFTW for 3D Fourier transforms.
       For real-to-complex transforms the output dimensions are [nx][ny][nz/2+1]. */
    int n2c = nz / 2 + 1;
    fftw_complex *F1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * n2c);
    fftw_complex *F2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * n2c);
    if (!F1 || !F2) {
        fprintf(stderr, "Error allocating FFTW arrays.\n");
        free(map1);
        free(map2);
        return 1;
    }
    fftw_plan plan1 = fftw_plan_dft_r2c_3d(nx, ny, nz, map1, F1, FFTW_ESTIMATE);
    fftw_plan plan2 = fftw_plan_dft_r2c_3d(nx, ny, nz, map2, F2, FFTW_ESTIMATE);
    fftw_execute(plan1);
    fftw_execute(plan2);
    free(map1);
    free(map2);

    /* Determine spatial frequency resolution.
       Assume pixel sizes: dx = xlen/nx, etc. Frequencies in 1/Å are obtained by dividing
       the voxel index difference by the cell dimensions.
    */
    //double dx = header1.xlen / nx;
    //double dy = header1.ylen / ny;
    //double dz = header1.zlen / nz;

    /* Maximum frequency along each axis is roughly 1/(2*pixel_size). */
    double fx_max = (nx / 2) / header1.xlen;
    double fy_max = (ny / 2) / header1.ylen;
    double fz_max = (nz / 2) / header1.zlen;
    double f_max = sqrt(fx_max * fx_max + fy_max * fy_max + fz_max * fz_max);

    int num_bins = 200;
    double bin_width = f_max / num_bins;

    /* Allocate arrays for FSC accumulation */
    double *sum_num = (double*) calloc(num_bins, sizeof(double));
    double *sum_den1 = (double*) calloc(num_bins, sizeof(double));
    double *sum_den2 = (double*) calloc(num_bins, sizeof(double));
    int *count = (int*) calloc(num_bins, sizeof(int));
    if (!sum_num || !sum_den1 || !sum_den2 || !count) {
        fprintf(stderr, "Error allocating accumulation arrays.\n");
        fftw_free(F1);
        fftw_free(F2);
        fftw_destroy_plan(plan1);
        fftw_destroy_plan(plan2);
        return 1;
    }

    /* Loop over the Fourier space grid.
       For indices in the x and y dimensions we consider the full range [0, n-1],
       and for the z dimension we loop over [0, n2c-1] (resulting from the real-to-complex FFT).
       The Fourier coordinate for index i is given by:
         if (i <= n/2) then i, else (i - n)
       divided by the cell dimension.
    */
    for (int i = 0; i < nx; i++) {
        int ii = (i <= nx / 2) ? i : i - nx;
        double fx = ii / header1.xlen; /* frequency in 1/Å */
        for (int j = 0; j < ny; j++) {
            int jj = (j <= ny / 2) ? j : j - ny;
            double fy = jj / header1.ylen;
            for (int k = 0; k < n2c; k++) {
                double fz = k / header1.zlen; /* k is non-negative */
                double freq = sqrt(fx * fx + fy * fy + fz * fz);
                int bin = (int)(freq / bin_width);
                if (bin < num_bins) {
                    int index = (i * ny + j) * n2c + k;
                    double a_re = F1[index][0];
                    double a_im = F1[index][1];
                    double b_re = F2[index][0];
                    double b_im = F2[index][1];
                    double prod = a_re * b_re + a_im * b_im; // Real part of F1*conj(F2)
                    double power1 = a_re * a_re + a_im * a_im;
                    double power2 = b_re * b_re + b_im * b_im;
                    sum_num[bin] += prod;
                    sum_den1[bin] += power1;
                    sum_den2[bin] += power2;
                    count[bin]++;
                }
            }
        }
    }

    /* Write the FSC curve to the specified output file.
       Each line contains the center frequency of the bin and the FSC value.
    */
    FILE *outf = fopen(output_file, "w");
    if (!outf) {
        perror(output_file);
        fftw_free(F1);
        fftw_free(F2);
        fftw_destroy_plan(plan1);
        fftw_destroy_plan(plan2);
        free(sum_num);
        free(sum_den1);
        free(sum_den2);
        free(count);
        return 1;
    }
    for (int b = 0; b < num_bins; b++) {
        double f_center = (b + 0.5) * bin_width;
        double fsc = 0.0;
        if (sum_den1[b] > 0 && sum_den2[b] > 0)
            fsc = sum_num[b] / sqrt(sum_den1[b] * sum_den2[b]);
        fprintf(outf, "%lf\t%lf\n", f_center, fsc);
    }
    fclose(outf);

    /* Clean up FFTW plans and allocated memory */
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    fftw_free(F1);
    fftw_free(F2);
    free(sum_num);
    free(sum_den1);
    free(sum_den2);
    free(count);

    printf("FSC calculation complete. Results have been saved in %s\n", output_file);
    return 0;
}

