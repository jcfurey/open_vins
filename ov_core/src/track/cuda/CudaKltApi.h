/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef OV_CUDA_KLT_API_H
#define OV_CUDA_KLT_API_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct OvCudaKlt OvCudaKlt;
typedef struct { float x,y; } OvCudaPoint;
int ov_cuda_klt_device_count(void);
int ov_cuda_klt_runtime_version(void);
int ov_cuda_klt_compiled_version(void);
int ov_cuda_klt_device_name(int device,char *name,size_t size);
OvCudaKlt *ov_cuda_klt_create(int device,int levels,int window,char *message,size_t size);
void ov_cuda_klt_destroy(OvCudaKlt *context);
int ov_cuda_klt_track(OvCudaKlt *context,const uint8_t *previous,size_t previous_step,
                      const uint8_t *current,size_t current_step,int width,int height,
                      const OvCudaPoint *points_previous,OvCudaPoint *points_current,
                      uint8_t *status,float *error,int count,char *message,size_t size);
#ifdef __cplusplus
}
#endif
#endif
