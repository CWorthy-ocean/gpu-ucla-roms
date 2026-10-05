#ifdef USE_CUDA
#define CUF_DO1 !$cuf kernel do(1) <<<*,*,0,work_stream>>>
#define CUF_DO2 !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#define CUF_DO3 !$cuf kernel do(3) <<<*,*,0,work_stream>>>
! One thread, for corner points: keeps scalar stores on the device.
#define CUF_SERIAL !$cuf kernel do(1) <<<1,1,0,work_stream>>>
#else
#define CUF_DO1
#define CUF_DO2
#define CUF_DO3
#define CUF_SERIAL
#endif
