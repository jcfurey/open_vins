/* SPDX-License-Identifier: GPL-3.0-or-later */
// Pyramidal inverse-compositional sparse LK. No dependency on an OpenCV CUDA build.
#include "CudaKltApi.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void checked(cudaError_t result) {
  if (result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void message(char *out,size_t size,const char *value) {if(out && size) std::snprintf(out,size,"%s",value);}
struct Pyramid {
  float *previous[9]{},*current[9]{},*gx[9]{},*gy[9]{};
  int width[9]{},height[9]{},levels=0;
};
__device__ int reflected(int p,int n) {
  if(n<=1) return 0;
  const int period=2*n-2;p%=period;if(p<0)p+=period;return min(p,period-p);
}
__device__ float at(const float *image,int w,int h,int x,int y) {
  return image[reflected(y,h)*w+reflected(x,w)];
}
__device__ float bilinear(const float *image,int w,int h,float x,float y) {
  const int ix=static_cast<int>(floorf(x)),iy=static_cast<int>(floorf(y));const float a=x-ix,b=y-iy;
  return (1-a)*((1-b)*at(image,w,h,ix,iy)+b*at(image,w,h,ix,iy+1))+
              a*((1-b)*at(image,w,h,ix+1,iy)+b*at(image,w,h,ix+1,iy+1));
}
__global__ void grayscale(const uint8_t *input,size_t stride,float *output,int w,int h) {
  const int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x<w && y<h) output[y*w+x]=input[y*stride+x];
}
__global__ void downsample(const float *input,int w,int h,float *output,int ow,int oh) {
  const int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=ow || y>=oh)return;
  const float kernel[5]={1,4,6,4,1};float value=0;
  for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx)
    value+=kernel[dy+2]*kernel[dx+2]*at(input,w,h,2*x+dx,2*y+dy);
  output[y*ow+x]=value/256.f;
}
__global__ void derivatives(const float *input,float *gx,float *gy,int w,int h) {
  const int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=w || y>=h)return;
  gx[y*w+x]=(3*(at(input,w,h,x+1,y-1)-at(input,w,h,x-1,y-1))+
               10*(at(input,w,h,x+1,y)-at(input,w,h,x-1,y))+
                3*(at(input,w,h,x+1,y+1)-at(input,w,h,x-1,y+1)))/32.f;
  gy[y*w+x]=(3*(at(input,w,h,x-1,y+1)-at(input,w,h,x-1,y-1))+
               10*(at(input,w,h,x,y+1)-at(input,w,h,x,y-1))+
                3*(at(input,w,h,x+1,y+1)-at(input,w,h,x+1,y-1)))/32.f;
}
__device__ void reduce(float &a,float &b,float &c) {
  __shared__ float sums[3][8];const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(int offset=16;offset>0;offset/=2) {
    a+=__shfl_down_sync(0xffffffff,a,offset);b+=__shfl_down_sync(0xffffffff,b,offset);c+=__shfl_down_sync(0xffffffff,c,offset);
  }
  if(lane==0){sums[0][warp]=a;sums[1][warp]=b;sums[2][warp]=c;}
  __syncthreads();
  if(warp==0) {
    a=lane<8?sums[0][lane]:0;b=lane<8?sums[1][lane]:0;c=lane<8?sums[2][lane]:0;
    for(int offset=16;offset>0;offset/=2) {
      a+=__shfl_down_sync(0xffffffff,a,offset);b+=__shfl_down_sync(0xffffffff,b,offset);c+=__shfl_down_sync(0xffffffff,c,offset);
    }
    if(lane==0){sums[0][0]=a;sums[1][0]=b;sums[2][0]=c;}
  }
  __syncthreads();a=sums[0][0];b=sums[1][0];c=sums[2][0];
  // The next invocation may reuse this shared storage immediately.
  __syncthreads();
}
__global__ void sparse_lk(Pyramid pyramid,int window,const OvCudaPoint *previous,OvCudaPoint *current,
                           uint8_t *status,float *error) {
  const int feature=blockIdx.x,half=window/2,area=window*window;
  const auto origin=previous[feature],guess=current[feature];
  bool good=isfinite(origin.x)&&isfinite(origin.y)&&isfinite(guess.x)&&isfinite(guess.y);
  float x=guess.x/(1<<pyramid.levels),y=guess.y/(1<<pyramid.levels);
  float values[4]{},dx[4]{},dy[4]{};
  for(int level=pyramid.levels;level>=0 && good;--level) {
    if(level!=pyramid.levels){x*=2;y*=2;}
    const int w=pyramid.width[level],h=pyramid.height[level];
    const float px=origin.x/(1<<level),py=origin.y/(1<<level);
    float H00=0,H01=0,H11=0;
    #pragma unroll
    for(int j=0;j<4;++j) {
      const int index=threadIdx.x+256*j;
      if(index<area) {
        const float tx=px+index%window-half,ty=py+index/window-half;
        values[j]=bilinear(pyramid.previous[level],w,h,tx,ty);
        dx[j]=bilinear(pyramid.gx[level],w,h,tx,ty);dy[j]=bilinear(pyramid.gy[level],w,h,tx,ty);
        H00+=dx[j]*dx[j];H01+=dx[j]*dy[j];H11+=dy[j]*dy[j];
      }
    }
    reduce(H00,H01,H11);
    const float determinant=H00*H11-H01*H01;
    // Scharr/32 derivatives in 8-bit intensity units; equivalent scale to CPU LK's default minEigThreshold.
    const float eigen=.5f*(H00+H11-sqrtf((H00-H11)*(H00-H11)+4*H01*H01))/area;
    if(!isfinite(determinant)||determinant<1e-8f||eigen<.1024f) {
      if(level==0)good=false;
      continue;
    }
    float last_x=0,last_y=0;
    for(int iteration=0;iteration<30;++iteration) {
      if(!isfinite(x)||!isfinite(y)||x< -half||y< -half||x>=w+half||y>=h+half){good=false;break;}
      float b0=0,b1=0,cost=0;
      #pragma unroll
      for(int j=0;j<4;++j) {
        const int index=threadIdx.x+256*j;
        if(index<area) {
          const float residual=bilinear(pyramid.current[level],w,h,x+index%window-half,y+index/window-half)-values[j];
          b0+=dx[j]*residual;b1+=dy[j]*residual;cost+=fabsf(residual);
        }
      }
      reduce(b0,b1,cost);
      const float sx=(-H11*b0+H01*b1)/determinant,sy=(H01*b0-H00*b1)/determinant;
      x+=sx;y+=sy;
      if(sx*sx+sy*sy<.0001f)break;
      if(iteration>0 && fabsf(sx+last_x)<.01f && fabsf(sy+last_y)<.01f){x-=.5f*sx;y-=.5f*sy;break;}
      last_x=sx;last_y=sy;
    }
  }
  good=good&&isfinite(x)&&isfinite(y)&&x>=0&&y>=0&&x<pyramid.width[0]&&y<pyramid.height[0];
  float cost=0,dummy0=0,dummy1=0;
  if(good) {
    #pragma unroll
    for(int j=0;j<4;++j) {
      const int index=threadIdx.x+256*j;
      if(index<area)cost+=fabsf(bilinear(pyramid.current[0],pyramid.width[0],pyramid.height[0],
                                       x+index%window-half,y+index/window-half)-values[j]);
    }
  }
  reduce(cost,dummy0,dummy1);
  if(threadIdx.x==0) {
    current[feature]=good?OvCudaPoint{x,y}:origin;status[feature]=good?1:0;error[feature]=good?cost/area:0;
  }
}
} // namespace

struct OvCudaKlt {
  int device=0,max_levels=0,window=0,width=0,height=0,capacity=0;
  cudaStream_t stream=nullptr;uint8_t *image=nullptr,*status=nullptr;size_t stride=0;
  float *error=nullptr;OvCudaPoint *previous=nullptr,*current=nullptr;
  Pyramid pyramid;std::vector<float *> images;
  ~OvCudaKlt() {
    cudaSetDevice(device);if(stream)cudaStreamSynchronize(stream);
    for(auto pointer:images)cudaFree(pointer);
    cudaFree(image);cudaFree(status);cudaFree(error);cudaFree(previous);cudaFree(current);
    if(stream)cudaStreamDestroy(stream);
  }
  void size(int w,int h,int count) {
    if(w!=width||h!=height) {
      for(auto pointer:images)checked(cudaFree(pointer));images.clear();
      checked(cudaFree(image));image=nullptr;checked(cudaMallocPitch(&image,&stride,w,h));
      width=w;height=h;pyramid=Pyramid();
      int level=0;
      while(true) {
        pyramid.width[level]=w;pyramid.height[level]=h;
        for(auto target:{&pyramid.previous[level],&pyramid.current[level],&pyramid.gx[level],&pyramid.gy[level]}) {
          checked(cudaMalloc(target,static_cast<size_t>(w)*h*sizeof(float)));images.push_back(*target);
        }
        if(level>=max_levels || (w+1)/2<=window || (h+1)/2<=window)break;
        w=(w+1)/2;h=(h+1)/2;++level;
      }
      pyramid.levels=level;
    }
    if(count>capacity) {
      checked(cudaFree(status));checked(cudaFree(error));checked(cudaFree(previous));checked(cudaFree(current));
      status=nullptr;error=nullptr;previous=nullptr;current=nullptr;capacity=0;
      const int n=std::max(512,count);
      checked(cudaMalloc(&status,n));checked(cudaMalloc(&error,n*sizeof(float)));
      checked(cudaMalloc(&previous,n*sizeof(OvCudaPoint)));checked(cudaMalloc(&current,n*sizeof(OvCudaPoint)));capacity=n;
    }
  }
  void build(const uint8_t *host,size_t step,bool first) {
    checked(cudaMemcpy2DAsync(image,stride,host,step,width,height,cudaMemcpyHostToDevice,stream));
    auto *targets=first?pyramid.previous:pyramid.current;
    const dim3 block(16,16),grid((width+15)/16,(height+15)/16);
    grayscale<<<grid,block,0,stream>>>(image,stride,targets[0],width,height);
    for(int level=0;level<=pyramid.levels;++level) {
      const int w=pyramid.width[level],h=pyramid.height[level];const dim3 size((w+15)/16,(h+15)/16);
      if(level>0)downsample<<<size,block,0,stream>>>(targets[level-1],pyramid.width[level-1],pyramid.height[level-1],targets[level],w,h);
      if(first)derivatives<<<size,block,0,stream>>>(targets[level],pyramid.gx[level],pyramid.gy[level],w,h);
    }
  }
};
extern "C" int ov_cuda_klt_device_count() {int count=0;return cudaGetDeviceCount(&count)==cudaSuccess?count:0;}
extern "C" int ov_cuda_klt_device_name(int device,char *name,size_t size) {
  cudaDeviceProp properties{};if(cudaGetDeviceProperties(&properties,device)!=cudaSuccess)return 0;
  message(name,size,properties.name);return 1;
}
extern "C" OvCudaKlt *ov_cuda_klt_create(int device,int levels,int window,char *out,size_t n) {
  try {
    if(device<0||levels<0||levels>8||window<3||window>31||window%2==0)throw std::invalid_argument("Invalid CUDA KLT device/levels/window");
    auto context=std::make_unique<OvCudaKlt>();context->device=device;context->max_levels=levels;context->window=window;
    checked(cudaSetDevice(device));checked(cudaStreamCreateWithFlags(&context->stream,cudaStreamNonBlocking));
    return context.release();
  }catch(const std::exception &error){message(out,n,error.what());return nullptr;}
}
extern "C" void ov_cuda_klt_destroy(OvCudaKlt *context) {delete context;}
extern "C" int ov_cuda_klt_track(OvCudaKlt *context,const uint8_t *before,size_t before_step,
                                  const uint8_t *after,size_t after_step,int width,int height,
                                  const OvCudaPoint *previous,OvCudaPoint *current,uint8_t *status,float *error,int count,char *out,size_t n) {
  try {
    if(!context||!before||!after||width<3||height<3||count<0||count>1000000)throw std::invalid_argument("Invalid CUDA KLT input");
    if(count==0)return 1;
    checked(cudaSetDevice(context->device));context->size(width,height,count);
    context->build(before,before_step,true);context->build(after,after_step,false);
    checked(cudaMemcpyAsync(context->previous,previous,count*sizeof(OvCudaPoint),cudaMemcpyHostToDevice,context->stream));
    checked(cudaMemcpyAsync(context->current,current,count*sizeof(OvCudaPoint),cudaMemcpyHostToDevice,context->stream));
    sparse_lk<<<count,256,0,context->stream>>>(context->pyramid,context->window,context->previous,context->current,context->status,context->error);
    checked(cudaGetLastError());
    checked(cudaMemcpyAsync(current,context->current,count*sizeof(OvCudaPoint),cudaMemcpyDeviceToHost,context->stream));
    checked(cudaMemcpyAsync(status,context->status,count,cudaMemcpyDeviceToHost,context->stream));
    checked(cudaMemcpyAsync(error,context->error,count*sizeof(float),cudaMemcpyDeviceToHost,context->stream));
    checked(cudaStreamSynchronize(context->stream));return 1;
  }catch(const std::exception &failure){message(out,n,failure.what());return 0;}
}
