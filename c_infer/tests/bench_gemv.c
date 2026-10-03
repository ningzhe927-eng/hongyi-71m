#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
extern void openblas_set_num_threads(int);
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc,char**argv){
  int nth = argc>1?atoi(argv[1]):8;
  openblas_set_num_threads(nth);
  int K=608; int Ns[]={608,1621,3242,6484,12968,30008};
  int M=1;
  float *x=malloc(sizeof(float)*K);
  for(int i=0;i<K;i++) x[i]=0.01f*(i%13-6);
  for(unsigned t=0;t<sizeof(Ns)/sizeof(Ns[0]);t++){
    int N=Ns[t];
    float *W=malloc(sizeof(float)*(size_t)N*K), *y=malloc(sizeof(float)*N);
    for(size_t i=0;i<(size_t)N*K;i++) W[i]=0.001f*(i%7);
    for(int r=0;r<3;r++) cblas_sgemv(CblasRowMajor,CblasNoTrans,N,K,1.0f,W,K,x,1,0.0f,y,1);
    int rep = 200000/N + 5;
    double t0=now();
    for(int r=0;r<rep;r++) cblas_sgemv(CblasRowMajor,CblasNoTrans,N,K,1.0f,W,K,x,1,0.0f,y,1);
    double dt=(now()-t0)/rep;
    printf("  threads=%d  N=%6d K=%d : %8.3f us/call  %6.1f GB/s\n", nth, N, K, dt*1e6,
           (double)N*K*4/(dt*1e9));
    free(W);free(y);
  }
  (void)M; return 0;
}
