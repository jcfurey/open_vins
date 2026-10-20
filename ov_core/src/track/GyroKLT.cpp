/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "GyroKLT.h"
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace ov_core {
namespace {
float sample(const cv::Mat &image, float x, float y) {
  const int ix = static_cast<int>(x), iy = static_cast<int>(y);
  const float a = x - ix, b = y - iy;
  const auto *p = image.ptr<uchar>(iy), *q = image.ptr<uchar>(iy + 1);
  return ((1-a)*((1-b)*p[ix]+b*q[ix]) + a*((1-b)*p[ix+1]+b*q[ix+1])) / 255.f;
}
Eigen::Vector2d gradient(const cv::Mat &image, float x, float y) {
  const int ix = static_cast<int>(x), iy = static_cast<int>(y);
  const float a = x - ix, b = y - iy;
  const auto *p = image.ptr<cv::Vec2s>(iy), *q = image.ptr<cv::Vec2s>(iy + 1);
  const cv::Vec2f g = (1-a)*((1-b)*cv::Vec2f(p[ix])+b*cv::Vec2f(q[ix])) +
                         a*((1-b)*cv::Vec2f(p[ix+1])+b*cv::Vec2f(q[ix+1]));
  return Eigen::Vector2d(g[0], g[1]) / (32. * 255.);
}
} // namespace

// A bounded, reusable pool also works inside OpenCV's camera-parallel region,
// where nested cv::parallel_for_ would execute serially. Only independent pixel
// fits run here; references, IDs, counters and database commits remain serial.
struct GyroKLT::Workers {
  std::vector<std::thread> threads;
  std::mutex mutex;
  std::condition_variable ready,done;
  std::function<void(size_t)> job;
  std::atomic<size_t> next{0};
  size_t count=0,generation=0,remaining=0;
  bool stopping=false;
  std::exception_ptr exception;
  explicit Workers(int number) {
    try {
      for(int i=1;i<number;++i) threads.emplace_back([this] {
        size_t seen=0;
        std::unique_lock<std::mutex> lock(mutex);
        while(true) {
          ready.wait(lock,[&]{return stopping || generation!=seen;});
          if(stopping)return;
          seen=generation;lock.unlock();execute();lock.lock();
          if(--remaining==0)done.notify_one();
        }
      });
    } catch(...) {stop();throw;}
  }
  ~Workers(){stop();}
  void stop() {
    {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
    ready.notify_all();for(auto &thread:threads)if(thread.joinable())thread.join();
  }
  void execute() {
    try {
      while(true){const size_t i=next.fetch_add(1,std::memory_order_relaxed);if(i>=count)break;job(i);}
    } catch(...) {
      std::lock_guard<std::mutex> lock(mutex);
      if(!exception)exception=std::current_exception();
      next.store(count,std::memory_order_relaxed);
    }
  }
  void run(size_t n,const std::function<void(size_t)> &function) {
    if(threads.empty() || n<8){for(size_t i=0;i<n;++i)function(i);return;}
    {
      std::lock_guard<std::mutex> lock(mutex);
      job=function;count=n;next.store(0,std::memory_order_relaxed);exception=nullptr;remaining=threads.size();++generation;
    }
    ready.notify_all();execute();
    std::unique_lock<std::mutex> lock(mutex);done.wait(lock,[&]{return remaining==0;});job={};
    if(exception)std::rethrow_exception(exception);
  }
};

GyroKLT::GyroKLT(std::shared_ptr<CamBase> camera, std::string mode, int window, int reference_age, int threads)
    : camera(std::move(camera)), mode(std::move(mode)), window(window), reference_age(reference_age) {
  if (!this->camera || (this->mode != "seed" && this->mode != "affine" && this->mode != "reference") ||
      window < 3 || window > 63 || window % 2 == 0 || reference_age < 1 || reference_age > 100 || threads<1 || threads>16)
    throw std::invalid_argument("Invalid gyro KLT mode/window/reference age");
  workers=std::make_unique<Workers>(this->mode=="seed" ? 1 : threads);
}
GyroKLT::~GyroKLT()=default;

void GyroKLT::begin_frame(const Eigen::Matrix3d *rotation) {
  ++frame;
  previous_orientation = orientation;
  prediction_valid = rotation && rotation->allFinite() &&
      (rotation->transpose()*(*rotation)-Eigen::Matrix3d::Identity()).norm() < 1e-5 && std::abs(rotation->determinant()-1) < 1e-5;
  if (!prediction_valid) {
    ++stats.missing_frames;
    references.clear();
    previous_orientation.setIdentity();
    orientation.setIdentity();
    step.setIdentity();
    return;
  }
  step = *rotation;
  orientation = Eigen::Quaterniond(step * previous_orientation).normalized().toRotationMatrix();
}

bool GyroKLT::project(CamBase &camera, const Eigen::Matrix3d &rotation, const cv::Point2f &point,
                      cv::Point2f &predicted, Eigen::Matrix2d *affine) {
  auto map = [&](const cv::Point2f &p, cv::Point2f &out) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
    const Eigen::Vector2d norm = camera.undistort_d(Eigen::Vector2d(p.x,p.y));
    const Eigen::Vector3d ray = rotation * Eigen::Vector3d(norm.x(),norm.y(),1);
    if (!ray.allFinite() || ray.z() <= 1e-6) return false;
    const Eigen::Vector2d pixel = camera.distort_d(ray.head<2>()/ray.z());
    if (!pixel.allFinite()) return false;
    out = cv::Point2f(pixel.x(),pixel.y());
    return true;
  };
  if (!rotation.allFinite() || !map(point,predicted)) return false;
  if (affine) {
    cv::Point2f a,b,c,d;
    if (!map(point+cv::Point2f(.5,0),a) || !map(point-cv::Point2f(.5,0),b) ||
        !map(point+cv::Point2f(0,.5),c) || !map(point-cv::Point2f(0,.5),d)) return false;
    *affine << a.x-b.x,c.x-d.x,a.y-b.y,c.y-d.y;
    if (!affine->allFinite() || affine->determinant() < .2 || affine->determinant() > 5 || affine->norm() > 5) return false;
  }
  return true;
}

void GyroKLT::seed(const std::vector<cv::Point2f> &previous, std::vector<cv::Point2f> &current) {
  if (!prediction_valid) return;
  for (size_t i=0;i<previous.size();++i) {
    cv::Point2f prediction;
    if (project(*camera,step,previous[i],prediction) && prediction.x>=0 && prediction.y>=0 &&
        prediction.x<camera->w() && prediction.y<camera->h()) {
      current[i]=prediction;
      ++stats.predictions;
    }
  }
}

GyroKLT::Reference GyroKLT::make_reference(const std::vector<cv::Mat> &pyramid, const cv::Point2f &point,
                                          const Eigen::Matrix3d &rotation) const {
  Reference result;
  result.point=point; result.orientation=rotation; result.frame=frame-1;
  // Refinement uses the two finest levels. Coarse search remains native pyramidal LK.
  for (size_t level=0;level<2 && 2*level<pyramid.size();++level) {
    const auto &image=pyramid[2*level];
    const cv::Point2f center=point/static_cast<float>(1<<level);
    const float half=window/2.f+1;
    cv::Mat patch;
    if (image.type()==CV_8UC1 && center.x>=half && center.y>=half && center.x<image.cols-half && center.y<image.rows-half) {
      cv::getRectSubPix(image,cv::Size(window,window),center,patch,CV_32F);
      patch*=1.f/255.f;
    }
    result.patches.push_back(patch);
  }
  return result;
}

bool GyroKLT::align(const Reference &reference, const std::vector<cv::Mat> &current, const Eigen::Matrix2d &affine,
                    cv::Point2f &point, float &error) const {
  const cv::Point2f initial=point;
  const int half=window/2;
  bool aligned=false;
  for (int level=static_cast<int>(reference.patches.size())-1;level>=0;--level) {
    if (reference.patches[level].empty() || 2*level+1>=static_cast<int>(current.size())) continue;
    const auto &image=current[2*level], &derivative=current[2*level+1], &patch=reference.patches[level];
    if (image.type()!=CV_8UC1 || derivative.type()!=CV_16SC2) return false;
    const float scale=static_cast<float>(1<<level);
    Eigen::Vector2d location(point.x/scale,point.y/scale);
    double gain=1,offset=0,rmse=0;
    for (int iteration=0;iteration<12;++iteration) {
      Eigen::Matrix4d H=Eigen::Matrix4d::Zero();
      Eigen::Vector4d b=Eigen::Vector4d::Zero();
      double cost=0;
      for (int y=-half;y<=half;++y) for (int x=-half;x<=half;++x) {
        const Eigen::Vector2d at=location+affine*Eigen::Vector2d(x,y);
        if (at.x()<1 || at.y()<1 || at.x()>=image.cols-2 || at.y()>=image.rows-2) return false;
        const double value=patch.at<float>(y+half,x+half);
        const double residual=sample(image,at.x(),at.y())-gain*value-offset;
        const auto g=gradient(derivative,at.x(),at.y());
        Eigen::Vector4d J; J<<g.x(),g.y(),-value,-1;
        const double weight=std::min(1.,.05/std::max(1e-9,std::abs(residual)));
        H.noalias()+=weight*J*J.transpose(); b.noalias()+=weight*J*residual;
        cost+=residual*residual;
      }
      // Eliminate brightness rather than hiding a singular patch behind damping.
      Eigen::LDLT<Eigen::Matrix2d> photo(H.bottomRightCorner<2,2>());
      if (!photo.isPositive() || photo.vectorD().minCoeff()<1e-7) return false;
      const Eigen::Matrix2d S=H.topLeftCorner<2,2>()-H.topRightCorner<2,2>()*photo.solve(H.bottomLeftCorner<2,2>());
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen(S);
      if (eigen.info()!=Eigen::Success || eigen.eigenvalues()(0)<1e-5 || eigen.eigenvalues()(1)/eigen.eigenvalues()(0)>1e4) return false;
      const Eigen::Vector4d delta=H.ldlt().solve(-b);
      if (!delta.allFinite() || delta.head<2>().norm()>3) return false;
      location+=delta.head<2>(); gain+=delta(2);offset+=delta(3);
      if (gain<.25 || gain>4 || std::abs(offset)>.75) return false;
      rmse=std::sqrt(cost/(window*window));
      if (delta.head<2>().norm()<.005 && std::abs(delta(2))<.001 && std::abs(delta(3))<.001) break;
    }
    point=cv::Point2f(location.x()*scale,location.y()*scale);
    error=static_cast<float>(255*rmse);
    aligned=true;
  }
  // Prevent a reference patch jumping to another instance of repetitive texture.
  return aligned && cv::norm(point-initial)<=2.0 && std::isfinite(error) && error<=25;
}

void GyroKLT::refine(const std::vector<cv::Mat> &previous_raw, const std::vector<cv::Mat> &current_raw,
                     const std::vector<cv::Point2f> &previous, const std::vector<size_t> &ids,
                     std::vector<cv::Point2f> &current, std::vector<uchar> &status, std::vector<float> &error) {
  if (!prediction_valid || mode=="seed" || previous_raw.empty() || current_raw.empty()) return;
  if (previous.size()!=ids.size() || current.size()!=ids.size() || status.size()!=ids.size() || error.size()!=ids.size())
    throw std::invalid_argument("Gyro KLT point/id/status/error sizes differ");
  std::vector<Reference> local(ids.size());
  std::vector<const Reference *> selected(ids.size(),nullptr);
  std::vector<uchar> retained(ids.size(),0),success(ids.size(),0);
  for(size_t i=0;i<ids.size();++i) {
    if(!status[i])continue;
    auto found=references.find(ids[i]);
    if(mode=="reference" && found!=references.end() && frame-found->second.frame<=static_cast<size_t>(reference_age)) {
      selected[i]=&found->second;retained[i]=1;
    } else {
      local[i]=make_reference(previous_raw,previous[i],previous_orientation);selected[i]=&local[i];
    }
  }
  workers->run(ids.size(),[&](size_t i) {
    const Reference *reference=selected[i];if(!reference)return;
    cv::Point2f prediction,result=current[i]; float residual=error[i]; Eigen::Matrix2d affine;
    const Eigen::Matrix3d rotation=orientation*reference->orientation.transpose();
    success[i]=project(*camera,rotation,reference->point,prediction,&affine) && align(*reference,current_raw,affine,result,residual);
    if(success[i]) {current[i]=result;error[i]=residual;}
  });
  for(size_t i=0;i<ids.size();++i) {
    if(!selected[i])continue;
    if (success[i]) {
      ++stats.refinements;
      if (retained[i]) ++stats.references;
      if (mode=="reference" && !retained[i]) references[ids[i]]=std::move(local[i]);
    } else {
      ++stats.rejected;
      // A changed viewpoint may invalidate the old reference. Preserve the independently
      // successful frame-to-frame LK result and re-anchor, rather than killing the track.
      if (retained[i]) references.erase(ids[i]);
    }
  }
}
void GyroKLT::rekey(size_t previous, size_t current) {
  if (previous==current) return;
  auto entry=references.extract(previous);
  if (!entry.empty()) { entry.key()=current; references.insert(std::move(entry)); }
}
void GyroKLT::retain(const std::vector<size_t> &ids) {
  const std::unordered_set<size_t> keep(ids.begin(),ids.end());
  for (auto it=references.begin();it!=references.end();) {
    if (!keep.count(it->first)) it=references.erase(it); else ++it;
  }
}
} // namespace ov_core
