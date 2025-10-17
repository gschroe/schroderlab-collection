/*
 * visdem_density.c
 *
 * This program reads in PDB coordinates (ATOM and HETATM entries),
 * renders a density map using the Peng kernel as implemented in pDenRenderPeng
 * This code is based partially on code from PDen by Benjamin Falkner 
 *
 *
 * The program supports an optional template MRC file (to obtain grid parameters)
 * and additional command-line options: -angpix, -box, -center, and -res (resolution cutoff).
 *
 * The density is rendered by first parsing the PDB file into arrays of coordinates,
 * atom type indices (mapped from the atom type string) and weighting factors.
 * Then the density grid is represented via a simple PDen_t structure and
 * the pDenRenderPeng function deposits density on this grid using the Peng kernel.
 *
 * Compile with:
 *    gcc -O2 -o dxpdb2mrc dxpdb2mrc.c -lm -lfftw3f
 *
 * Author: Gunnar F. Schroeder
 * Date: Apr 1, 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>

#define HEADER_SIZE 1024
#define MAX_LINE 256

/* -------------------- MRC I/O Functions -------------------- */

/* Minimal MRC header structure */
typedef struct {
    int nx, ny, nz;       // grid dimensions
    int mode;             // data type: 2 means float
    int nxstart, nystart, nzstart;
    int mx, my, mz;
    float xlen, ylen, zlen;   // cell dimensions in Angstroms
    float alpha, beta, gamma;
    int mapc, mapr, maps;
    float amin, amax, amean;
    int ispg;
    int nsymbt;
    char extra[100];
    char map[4];
    int machst;
    float rms;
    int nlabels;
    char labels[10][80];
} MRCHeader;

int read_template_mrc(const char *filename, int *nx, int *ny, int *nz, double *angpix, double center[3]) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open template MRC file %s\n", filename);
        return -1;
    }
    unsigned char header[HEADER_SIZE];
    if (fread(header, 1, HEADER_SIZE, fp) != HEADER_SIZE) {
        fprintf(stderr, "Error: Cannot read header from %s\n", filename);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *nx = *(int*)(header);
    *ny = *(int*)(header + 4);
    *nz = *(int*)(header + 8);
    float xlen = *(float*)(header + 40);
    float ylen = *(float*)(header + 44);
    float zlen = *(float*)(header + 48);
    if (*nx > 0)
        *angpix = xlen / (*nx);
    else
        *angpix = 1.0;
    center[0] = xlen / 2.0;
    center[1] = ylen / 2.0;
    center[2] = zlen / 2.0;
    return 0;
}

int write_mrc(const char *filename, MRCHeader *header, float *data) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open output MRC file %s\n", filename);
        return -1;
    }
    unsigned char buffer[HEADER_SIZE];
    memset(buffer, 0, HEADER_SIZE);
    *(int*)(buffer) = header->nx;
    *(int*)(buffer + 4) = header->ny;
    *(int*)(buffer + 8) = header->nz;
    *(int*)(buffer + 12) = header->mode;
    *(int*)(buffer + 16) = header->nxstart;
    *(int*)(buffer + 20) = header->nystart;
    *(int*)(buffer + 24) = header->nzstart;
    *(int*)(buffer + 28) = header->mx;
    *(int*)(buffer + 32) = header->my;
    *(int*)(buffer + 36) = header->mz;
    *(float*)(buffer + 40) = header->xlen;
    *(float*)(buffer + 44) = header->ylen;
    *(float*)(buffer + 48) = header->zlen;
    *(float*)(buffer + 52) = header->alpha;
    *(float*)(buffer + 56) = header->beta;
    *(float*)(buffer + 60) = header->gamma;
    *(int*)(buffer + 64) = header->mapc;
    *(int*)(buffer + 68) = header->mapr;
    *(int*)(buffer + 72) = header->maps;
    *(float*)(buffer + 76) = header->amin;
    *(float*)(buffer + 80) = header->amax;
    *(float*)(buffer + 84) = header->amean;
    *(int*)(buffer + 88) = header->ispg;
    *(int*)(buffer + 92) = header->nsymbt;
    memcpy(buffer + 96, header->extra, sizeof(header->extra));
    memcpy(buffer + 196, header->map, 4);
    *(int*)(buffer + 200) = header->machst;
    *(float*)(buffer + 204) = header->rms;
    *(int*)(buffer + 208) = header->nlabels;
    memcpy(buffer + 212, header->labels, 10 * 80);
    
    if (fwrite(buffer, 1, HEADER_SIZE, fp) != HEADER_SIZE) {
        fprintf(stderr, "Error: Cannot write header to %s\n", filename);
        fclose(fp);
        return -1;
    }
    size_t nvox = (size_t)header->nx * header->ny * header->nz;
    if (fwrite(data, sizeof(float), nvox, fp) != nvox) {
        fprintf(stderr, "Error: Cannot write density data to %s\n", filename);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

/* Resolution filtering using FFTW */
void apply_resolution_filter(float *density, int nx, int ny, int nz, double angpix, double resolution) {
    int nz_complex = nz/2 + 1;
    int nvox = nx * ny * nz;
    fftwf_complex *fft_data = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nx * ny * nz_complex);
    fftwf_plan plan_forward = fftwf_plan_dft_r2c_3d(nx, ny, nz, density, fft_data, FFTW_ESTIMATE);
    fftwf_execute(plan_forward);
    
    double cutoff_freq = 1.0 / resolution;
    
    for (int i = 0; i < nx; i++) {
        double fx = (i <= nx/2) ? ((double)i/(nx*angpix)) : ((double)(i - nx)/(nx*angpix));
        for (int j = 0; j < ny; j++) {
            double fy = (j <= ny/2) ? ((double)j/(ny*angpix)) : ((double)(j - ny)/(ny*angpix));
            for (int k = 0; k < nz_complex; k++) {
                double fz = (double)k/(nz*angpix);
                double fmag = sqrt(fx*fx + fy*fy + fz*fz);
                int idx = i*(ny*nz_complex) + j*nz_complex + k;
                if (fmag > cutoff_freq) {
                    fft_data[idx][0] = 0.0f;
                    fft_data[idx][1] = 0.0f;
                }
            }
        }
    }
    
    fftwf_plan plan_backward = fftwf_plan_dft_c2r_3d(nx, ny, nz, fft_data, density, FFTW_ESTIMATE);
    fftwf_execute(plan_backward);
    for (int i = 0; i < nvox; i++) {
        density[i] /= nvox;
    }
    
    fftwf_destroy_plan(plan_forward);
    fftwf_destroy_plan(plan_backward);
    fftwf_free(fft_data);
}

/* -------------------- Peng Kernel and Rendering -------------------- */

/* Define our own real type as double */
typedef double real;

/* PDen_t structure representing our density grid */
typedef struct {
    double origin[3];
    struct { size_t x, y, z; } size;
    struct { double x, y, z; } apix;
    float *data;
} PDen_t;

/* Global Peng parameters (adapted from render.c from pDen */
static const real PengGFactor = 10.0;
static const real PengScatterFactor_A[][5] = {
    { 0.0489, 0.2091, 0.7537, 1.1420, 0.3555 }, // ANY
    { 0.0088, 0.0449, 0.1481, 0.2356, 0.0914 }, // H
    { 0.0489, 0.2091, 0.7537, 1.1420, 0.3555 }, // C
    { 0.0267, 0.1328, 0.5301, 1.1020, 0.4215 }, // N
    { 0.0365, 0.1729, 0.5805, 0.8814, 0.3121 }, // O
    { 0.0915, 0.4312, 1.0847, 2.4671, 1.0852 }, // S
    { 0.1005, 0.4615, 1.0663, 2.5854, 1.2725 }, // P
    { 0.5229, 2.2874, 4.7243, 5.0807, 5.6389 }, // BA
    { 0.1260, 0.6442, 0.8893, 1.8197, 1.2988 }, // NA
    { 0.0799, 0.3891, 1.0037, 2.3332, 1.0507 }, // CL
    { 0.2149, 0.8703, 2.4999, 2.3591, 3.0318 }, // K
    { 0.1130, 0.5575, 0.9046, 2.1580, 1.4735 }  // MG
};

static const real PengScatterFactor_B[][5] = {
    { 0.1140, 1.0825, 5.4281, 17.8811,  51.1341 }, // ANY
    { 0.1152, 1.0867, 4.9755, 16.5591,  43.2743 }, // H
    { 0.1140, 1.0825, 5.4281, 17.8811,  51.1341 }, // C
    { 0.0541, 0.5165, 2.8207, 10.6297,  34.3764 }, // N
    { 0.0652, 0.6184, 2.9449, 9.62980,  28.2194 }, // O
    { 0.0838, 0.7788, 4.3462, 15.5846,  44.6365 }, // S
    { 0.0977, 0.9084, 4.9654, 18.5471,  54.3648 }, // P
    { 0.1434, 1.6019, 9.4511, 42.7685, 148.4969 }, // BA
    { 0.1684, 1.7150, 8.8386, 50.8265, 147.2073 }, // NA
    { 0.0694, 0.6443, 3.5351, 12.5058,  35.8633 }, // CL
    { 0.1660, 1.6906, 8.7447, 46.7825, 165.6923 }, // K
    { 0.1356, 1.3579, 6.9255, 32.3165,  92.1138 }  // MG
};

/* Peng kernel: computes density contribution given r, atom type, and sigma.
   Here r is the (squared) distance and sigma is modified (sigma^2) as in render.c.
*/
double pDenPengKernel(double r, int type, double sigma) {
    double h0, h1, h2, h3, h4, v;
    double sf[5], sf_a[5];
    h0 = -9.8696 * r;
    
    sf[0] = PengScatterFactor_B[type][0];
    sf[1] = PengScatterFactor_B[type][1];
    sf[2] = PengScatterFactor_B[type][2];
    sf[3] = PengScatterFactor_B[type][3];
    sf[4] = PengScatterFactor_B[type][3];  // using index 3 for pass5 as in original code
    sf_a[0] = PengScatterFactor_A[type][0];
    sf_a[1] = PengScatterFactor_A[type][1];
    sf_a[2] = PengScatterFactor_A[type][2];
    sf_a[3] = PengScatterFactor_A[type][3];
    sf_a[4] = PengScatterFactor_A[type][4];
    
    // Pass 1
    h1 = sf[0];
    h2 = 1.0 / (h1 + PengGFactor);
    h3 = 1.0 / h1;
    h4 = sqrt(h3 * h3 * h3);
    v = sf_a[0] * h4 * exp(h0 * h2);
    // Pass 2
    h1 = sf[1];
    h2 = 1.0 / (h1 + PengGFactor);
    h3 = 1.0 / h1;
    h4 = sqrt(h2 * h2 * h2);
    v += sf_a[1] * h4 * exp(h0 * h2);
    // Pass 3
    h1 = sf[2];
    h2 = 1.0 / (h1 + PengGFactor);
    h3 = 1.0 / h1;
    h4 = sqrt(h2 * h2 * h2);
    v += sf_a[2] * h4 * exp(h0 * h2);
    // Pass 4
    h1 = sf[3];
    h2 = 1.0 / (h1 + PengGFactor);
    h3 = 1.0 / h1;
    h4 = sqrt(h2 * h2 * h2);
    v += sf_a[3] * h4 * exp(h0 * h2);
    // Pass 5
    h1 = sf[4];
    h2 = 1.0 / (h1 + PengGFactor);
    h3 = 1.0 / h1;
    h4 = sqrt(h2 * h2 * h2);
    v += sf_a[4] * h4 * exp(h0 * h2);
    
    return v;
}

/* pDenRenderPeng: Renders the density using the Peng kernel.
   This function is adapted from the render.c "render" macro from pDen.
   Parameters:
      result - pointer to a PDen_t structure with grid parameters and allocated data.
      width  - defines the walk radius (in Å).
      sigma  - kernel parameter (will be squared internally).
      n      - number of atoms.
      type   - array of atom type indices (integers corresponding to PengScatterFactor arrays).
      coords - array of atom coordinates (length 3*n).
      factor - array of per-atom weighting factors.
   Returns:
      pointer to the updated result.
*/
PDen_t * pDenRenderPeng(PDen_t * result, double width, double sigma, size_t n, int *type, double *coords, double *factor) {
    size_t atom;
    size_t idx, idxx;
    size_t i, j, k;
    struct { size_t x, y, z; } gridid, gridend, walk;
    struct { double x, y, z; } pos, r;
    double lenr;
    double density;
    double mass;
    size_t dim2, dim3;
    int type_;
    
    /* Square sigma as in original macro */
    sigma *= sigma;
    
    /* Determine walk radius (number of voxels) */
    walk.x = ((size_t)(0.5 * width / result->apix.x)) + 1;
    walk.y = ((size_t)(0.5 * width / result->apix.y)) + 1;
    walk.z = ((size_t)(0.5 * width / result->apix.z)) + 1;
    
    dim2 = result->size.x;
    dim3 = result->size.x * result->size.y;
    
    for (atom = 0; atom < n; atom++) {
        mass = 1.0;
        /* Get atom position (using Å units) */
        pos.x = coords[3 * atom] - result->origin[0];
        pos.y = coords[3 * atom + 1] - result->origin[1];
        pos.z = coords[3 * atom + 2] - result->origin[2];
        
        /* Get upper left grid point */
        gridid.x = (size_t)( pos.x / result->apix.x );
        gridid.y = (size_t)( pos.y / result->apix.y );
        gridid.z = (size_t)( pos.z / result->apix.z );
        
        /* Calculate end of walk */
        gridend.x = gridid.x + walk.x;
        gridend.y = gridid.y + walk.y;
        gridend.z = gridid.z + walk.z;
        
        gridid.x = (gridid.x > walk.x ? gridid.x - walk.x + 1 : 0);
        gridid.y = (gridid.y > walk.y ? gridid.y - walk.y + 1 : 0);
        gridid.z = (gridid.z > walk.z ? gridid.z - walk.z + 1 : 0);
        
        mass = factor[atom];
        type_ = type[atom];
        //printf("type = %i\n", type_);
        
        for (k = gridid.z; k < gridend.z; k++) {
            if (k < result->size.z) {
                idxx = k * dim3;
                for (j = gridid.y; j < gridend.y; j++) {
                    if (j < result->size.y) {
                        for (i = gridid.x; i < gridend.x; i++) {
                            if (i < result->size.x) {
                                idx = i + j * dim2 + idxx;
                                r.x = result->apix.x * i - pos.x;
                                r.y = result->apix.y * j - pos.y;
                                r.z = result->apix.z * k - pos.z;
                                lenr = r.x * r.x + r.y * r.y + r.z * r.z;
                                density = pDenPengKernel(lenr, type_, sigma);
                                density *= mass;
                                result->data[idx] += (float)density;
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

/* -------------------- PDB Parsing Utilities -------------------- */

/* Returns an integer index for the atom type based on its string.
   Mapping (example):
      "H" -> 1, "C" -> 2, "N" -> 3, "O" -> 4, "S" -> 5, "P" -> 6.
   All other types return 0 (ANY).
*/
int get_atom_index(const char *atype) {
    if (!atype || strlen(atype) < 1)
        return 0;
    char c = atype[0];
    if (c == 'H' || c == 'h')
        return 1;
    else if (c == 'C' || c == 'c')
        return 2;
    else if (c == 'N' || c == 'n')
        return 3;
    else if (c == 'O' || c == 'o')
        return 4;
    else if (c == 'S' || c == 's')
        return 5;
    else if (c == 'P' || c == 'p')
        return 6;
    else
        return 0;
}

/* First pass: count the number of ATOM/HETATM lines in the PDB file */
size_t count_atoms_in_pdb(const char *pdb_file) {
    FILE *fp = fopen(pdb_file, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open PDB file %s\n", pdb_file);
        return 0;
    }
    char line[MAX_LINE];
    size_t count = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "ATOM", 4) == 0 || strncmp(line, "HETATM", 6) == 0)
            count++;
    }
    fclose(fp);
    return count;
}

/* Second pass: parse the PDB file and fill arrays of coordinates, types, and factors.
   Arrays are allocated with size 3*n, n, and n respectively.
   Returns n (number of atoms) on success.
*/
size_t parse_pdb(const char *pdb_file, double **p_coords, int **p_type, double **p_factor) {
    size_t n = count_atoms_in_pdb(pdb_file);
    if (n == 0)
        return 0;
    double *coords = (double*)malloc(3 * n * sizeof(double));
    int *types = (int*)malloc(n * sizeof(int));
    double *factors = (double*)malloc(n * sizeof(double));
    if (!coords || !types || !factors) {
        fprintf(stderr, "Error: Memory allocation failed for PDB arrays.\n");
        free(coords); free(types); free(factors);
        return 0;
    }
    
    FILE *fp = fopen(pdb_file, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open PDB file %s\n", pdb_file);
        free(coords); free(types); free(factors);
        return 0;
    }
    char line[MAX_LINE];
    size_t atom = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "ATOM", 4) == 0 || strncmp(line, "HETATM", 6) == 0) {


/*
            char x_str[9], y_str[9], z_str[9], atype[5] = "";
            strncpy(x_str, line + 30, 8); x_str[8] = '\0';
            strncpy(y_str, line + 38, 8); y_str[8] = '\0';
            strncpy(z_str, line + 46, 8); z_str[8] = '\0';
            // Try to extract atom type from columns 77-78; if not, use columns 13-16.
            if (strlen(line) >= 78) {
                strncpy(atype, line + 76, 2);
                atype[2] = '\0';
            }
            if (strlen(atype) == 0) {
                strncpy(atype, line + 12, 2);
                atype[2] = '\0';
            }

*/

            char x_str[9], y_str[9], z_str[9], atype[5] = "";
            char trimmed_atype[5] = "";
            strncpy(x_str, line + 30, 8); 
            x_str[8] = '\0';
            strncpy(y_str, line + 38, 8); 
            y_str[8] = '\0';
            strncpy(z_str, line + 46, 8); 
            z_str[8] = '\0';
            // Try to extract atom type from columns 77-78; if not, use columns 13-16.
            if (strlen(line) >= 78) {
                strncpy(atype, line + 76, 2);
                atype[2] = '\0';
            }
            if (strlen(atype) == 0) {
                strncpy(atype, line + 12, 2);
                atype[2] = '\0';
            }
            // Trim leading spaces from atype:
            int i = 0, j = 0;
            while (atype[i] != '\0') {
                if (atype[i] != ' ')
                    trimmed_atype[j++] = atype[i];
                i++;
            }
            trimmed_atype[j] = '\0';



            coords[3 * atom] = atof(x_str);
            coords[3 * atom + 1] = atof(y_str);
            coords[3 * atom + 2] = atof(z_str);
//            types[atom] = get_atom_index(atype);
            types[atom] = get_atom_index(trimmed_atype);

            factors[atom] = 1.0;  // Default weight; adjust if needed.
            atom++;
        }
    }
    fclose(fp);
    *p_coords = coords;
    *p_type = types;
    *p_factor = factors;
    return n;
}

/* -------------------- Logging Function -------------------- */

void log_execution(int argc, char *argv[]) {
    FILE *log_fp = fopen(".dxpdb2mrc.log", "a");
    if (!log_fp) {
        fprintf(stderr, "Warning: Cannot open log file .dxpdb2mrc.log\n");
        return;
    }
    
    time_t now = time(NULL);
    char *time_str = ctime(&now);
    time_str[strlen(time_str) - 1] = '\0';
    
    fprintf(log_fp, "[%s] dxpdb2mrc executed with arguments: ", time_str);
    for (int i = 0; i < argc; i++) {
        fprintf(log_fp, "%s", argv[i]);
        if (i < argc - 1) fprintf(log_fp, " ");
    }
    fprintf(log_fp, "\n");
    fclose(log_fp);
}

/* -------------------- Main Function -------------------- */

int main(int argc, char *argv[]) {
    log_execution(argc, argv);
    
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.pdb output.mrc [template.mrc] [-angpix value] [-box nx ny nz] [-center cx cy cz] [-res resolution]\n", argv[0]);
        return 1;
    }
    
    char *pdb_file = argv[1];
    char *output_mrc = argv[2];
    char *template_mrc = NULL;
    
    /* Default grid parameters */
    double angpix = 1.0;
    int box[3] = {128, 128, 128};  // grid dimensions (voxels)
    double center[3] = { (box[0]*angpix)/2.0, (box[1]*angpix)/2.0, (box[2]*angpix)/2.0 };
    double resolution = 0.0;  // resolution cutoff; 0 means no filtering
    
    int arg_index = 3;
    /* Optional template MRC file */
    if (arg_index < argc && argv[arg_index][0] != '-') {
        template_mrc = argv[arg_index];
        arg_index++;
        int t_nx, t_ny, t_nz;
        double t_angpix;
        double t_center[3];
        if (read_template_mrc(template_mrc, &t_nx, &t_ny, &t_nz, &t_angpix, t_center) == 0) {
            box[0] = t_nx;
            box[1] = t_ny;
            box[2] = t_nz;
            angpix = t_angpix;
            center[0] = t_center[0];
            center[1] = t_center[1];
            center[2] = t_center[2];
            printf("Template MRC read: box = %d x %d x %d, angpix = %f, center = (%f, %f, %f)\n",
                   box[0], box[1], box[2], angpix, center[0], center[1], center[2]);
        } else {
            fprintf(stderr, "Error reading template MRC. Using default parameters.\n");
        }
    }
    
    /* Process additional command line options */
    while (arg_index < argc) {
        if (strcmp(argv[arg_index], "-angpix") == 0) {
            if (arg_index + 1 < argc) {
                angpix = atof(argv[arg_index + 1]);
                arg_index += 2;
            } else {
                fprintf(stderr, "Error: -angpix requires a value.\n");
                return 1;
            }
        } else if (strcmp(argv[arg_index], "-box") == 0) {
            if (arg_index + 3 < argc) {
                box[0] = atoi(argv[arg_index + 1]);
                box[1] = atoi(argv[arg_index + 2]);
                box[2] = atoi(argv[arg_index + 3]);
                arg_index += 4;
            } else {
                fprintf(stderr, "Error: -box requires three integer values.\n");
                return 1;
            }
        } else if (strcmp(argv[arg_index], "-center") == 0) {
            if (arg_index + 3 < argc) {
                center[0] = atof(argv[arg_index + 1]);
                center[1] = atof(argv[arg_index + 2]);
                center[2] = atof(argv[arg_index + 3]);
                arg_index += 4;
            } else {
                fprintf(stderr, "Error: -center requires three values.\n");
                return 1;
            }
        } else if (strcmp(argv[arg_index], "-res") == 0) {
            if (arg_index + 1 < argc) {
                resolution = atof(argv[arg_index + 1]);
                arg_index += 2;
            } else {
                fprintf(stderr, "Error: -res requires a value.\n");
                return 1;
            }
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[arg_index]);
            return 1;
        }
    }
    
    /* Compute grid origin from center and box dimensions */
    double origin[3];
    origin[0] = center[0] - (box[0] * angpix) / 2.0;
    origin[1] = center[1] - (box[1] * angpix) / 2.0;
    origin[2] = center[2] - (box[2] * angpix) / 2.0;
    printf("Using grid origin: (%f, %f, %f)\n", origin[0], origin[1], origin[2]);
    printf("Grid dimensions: %d x %d x %d, angpix = %f\n", box[0], box[1], box[2], angpix);
    if (resolution > 0)
        printf("Resolution cutoff: %f Å\n", resolution);
    
    /* Parse the PDB file into coordinate, type, and factor arrays */
    double *coords = NULL;
    int *atom_types = NULL;
    double *factors = NULL;
    size_t natoms = parse_pdb(pdb_file, &coords, &atom_types, &factors);
    if (natoms == 0) {
        fprintf(stderr, "Error: No atoms parsed from PDB file.\n");
        return 1;
    }
    printf("Parsed %zu atoms from PDB file.\n", natoms);
    
    /* Allocate and initialize density grid via a PDen_t structure */
    PDen_t density_map;
    density_map.origin[0] = origin[0];
    density_map.origin[1] = origin[1];
    density_map.origin[2] = origin[2];
    density_map.size.x = box[0];
    density_map.size.y = box[1];
    density_map.size.z = box[2];
    density_map.apix.x = angpix;
    density_map.apix.y = angpix;
    density_map.apix.z = angpix;
    size_t nvox = box[0] * box[1] * box[2];
    density_map.data = (float*)calloc(nvox, sizeof(float));
    if (!density_map.data) {
        fprintf(stderr, "Error: Cannot allocate memory for density grid.\n");
        free(coords); free(atom_types); free(factors);
        return 1;
    }
    
    /* Render density map using pDenRenderPeng.
       Here we set width and sigma to default values; adjust as needed.
       For example, width = 10.0 Å and sigma = 2.0 Å.
    */
    double width = 18.0;
    double sigma = 4.0;
    pDenRenderPeng(&density_map, width, sigma, natoms, atom_types, coords, factors);
    
    /* Optionally, apply resolution filtering */
    if (resolution > 0) {
        printf("Applying resolution filter...\n");
        apply_resolution_filter(density_map.data, box[0], box[1], box[2], angpix, resolution);
    }
    
    /* Prepare MRC header */
    MRCHeader header;
    header.nx = box[0];
    header.ny = box[1];
    header.nz = box[2];
    header.mode = 2;  // float data
    header.nxstart = 0;
    header.nystart = 0;
    header.nzstart = 0;
    header.mx = box[0];
    header.my = box[1];
    header.mz = box[2];
    header.xlen = box[0] * angpix;
    header.ylen = box[1] * angpix;
    header.zlen = box[2] * angpix;
    header.alpha = 90.0;
    header.beta = 90.0;
    header.gamma = 90.0;
    header.mapc = 1;
    header.mapr = 2;
    header.maps = 3;
    float min_val = density_map.data[0], max_val = density_map.data[0];
    double sum = 0.0;
    for (size_t i = 0; i < nvox; i++) {
        if (density_map.data[i] < min_val) min_val = density_map.data[i];
        if (density_map.data[i] > max_val) max_val = density_map.data[i];
        sum += density_map.data[i];
    }
    header.amin = min_val;
    header.amax = max_val;
    header.amean = (float)(sum / nvox);
    header.ispg = 0;
    header.nsymbt = 0;
    memset(header.extra, 0, sizeof(header.extra));
    memcpy(header.map, "MAP ", 4);
    header.machst = 0x44440000;
    header.rms = 0.0;
    header.nlabels = 0;
    memset(header.labels, 0, sizeof(header.labels));
    
    if (write_mrc(output_mrc, &header, density_map.data) != 0) {
        fprintf(stderr, "Error writing output MRC file.\n");
        free(density_map.data);
        free(coords); free(atom_types); free(factors);
        return 1;
    }
    
    free(density_map.data);
    free(coords);
    free(atom_types);
    free(factors);
    printf("Density map written to %s\n", output_mrc);
    return 0;
}

