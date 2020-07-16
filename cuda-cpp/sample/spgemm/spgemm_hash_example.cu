#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <nsparse.hpp>
#include <CSR.hpp>
#include <SpGEMM.hpp>
#include <HashSpGEMM_volta.hpp>

typedef int IT;
#ifdef FLOAT
typedef float VT;
#else
typedef double VT;
#endif

template <class idType, class valType>
void spgemm_hash(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{

    idType i;
  
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;
  
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Memcpy A and B from Host to Device */
    a.memcpyHtD();
    b.memcpyHtD();
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, flop_count);

    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            c.release_csr();
        }
        cudaEventRecord(event[0], 0);
        SpGEMM_Hash(a, b, c);
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
    
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (Hash): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    /* Numeric Only */
    ave_msec = 0;
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        cudaEventRecord(event[0], 0);
        SpGEMM_Hash_Numeric(a, b, c);
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
    
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;

    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (Hash, only numeric phase): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    c.memcpyDtH();
    c.release_csr();

#ifdef sfDEBUG
    CSR<IT, VT> cusparse_c;
    SpGEMM_cuSPARSE(a, b, cusparse_c);
    if (c == cusparse_c) {
        std::cout << "HashSpGEMM is correctly executed" << std::endl;
    }
    std::cout << "Nnz of A: " << a.nnz << std::endl; 
    std::cout << "Number of intermediate products: " << flop_count / 2 << std::endl; 
    std::cout << "Nnz of C: " << c.nnz << std::endl; 
    cusparse_c.release_cpu_csr();
#endif

    a.release_csr();
    b.release_csr();

    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
}

template <typename T, typename index_type>
void print_mtx(const std::string& fname, const CSR<index_type, T>& a);

/*Main Function*/
int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cerr << "usage: ./spgemm_hash [A matrix] [B matrix]" << std::endl;
        return 1;
    }
    CSR<IT, VT> a, b, c;

    /* Set CSR reding from MM file or generating random matrix */
    std::cout << "Initialize Matrix A" << std::endl;
    std::cout << "Read matrix data from " << argv[1] << std::endl;
    a.init_data_from_mtx(argv[1]);

    std::cout << "Initialize Matrix B" << std::endl;
    std::cout << "Read matrix data from " << argv[2] << std::endl;
    b.init_data_from_mtx(argv[2]);

    std::cout << "Multiplying matrices..." << std::endl;

    SpGEMM_Hash(a, b, c);

    print_mtx("/gpfs/alpine/scratch/b2v/bif115/nsparse_c_output.mtx", c);

    a.release_cpu_csr();
    b.release_cpu_csr();
    c.release_cpu_csr();

    return 0;

}

template <typename T, typename index_type>
void print_mtx(const std::string& fname, const CSR<index_type, T>& a) {
  fprintf(stderr, "Printing to file %s\n", fname.c_str());
  FILE* f = fopen(fname.c_str(), "w");
  assert(f != nullptr);
  T* d_a_vals = a.d_values;
  index_type* d_a_row_ptr = a.d_rpt;
  index_type* d_a_col_ind = a.d_colids;
  size_t a_nnz = a.nnz;
  size_t a_m = a.nrow;
  size_t a_n = a.ncolumn;

  std::vector<T> a_vals(a_nnz);
  std::vector<index_type> a_col_ind(a_nnz);
  std::vector<index_type> a_row_ptr(a_m+1);

  fprintf(stderr, "cudaMemcpy'ing\n");
  cudaMemcpy(a_vals.data(), d_a_vals, a_vals.size() * sizeof(T), cudaMemcpyDefault);
  cudaMemcpy(a_col_ind.data(), d_a_col_ind, a_col_ind.size() * sizeof(index_type), cudaMemcpyDefault);
  cudaMemcpy(a_row_ptr.data(), d_a_row_ptr, a_row_ptr.size() * sizeof(index_type), cudaMemcpyDefault);

  size_t counted_nnz = 0;
  fprintf(stderr, "printing...\n");

  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%%\n");
  fprintf(f, "%lu %lu %lu\n", a_m, a_n, a_nnz);

  for (index_type i = 0; i < a_m; i++) {
    size_t col_num = 0;
    for (size_t j_ptr = a_row_ptr[i]; j_ptr < a_row_ptr[i+1]; j_ptr++) {
      T val = a_vals[j_ptr];
      index_type j = a_col_ind[j_ptr];
      fprintf(f, "%d %d %lf", i+1, j+1, val);
      counted_nnz++;
      if (col_num != 0) {
        if (a_col_ind[j_ptr] <= a_col_ind[j_ptr-1]) {
          fprintf(f, " [unsorted]");
        }
      }
      fprintf(f, "\n");
      col_num++;
    }
  }

  printf("Printing matrix to %s (%lu x %lu), %lu nnz (counted %lu nnz)\n",
         fname.c_str(), a_m, a_n, a_nnz, counted_nnz);
  fclose(f);
}

