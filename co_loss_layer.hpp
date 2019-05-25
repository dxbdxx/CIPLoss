#ifndef CAFFE_CO_LOSS_LAYER_HPP_
#define CAFFE_CO_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
	//************************************************************************/
	// bottom[0] is the data of one layer
	// top[0] output the ClusterLoss                                                          
	// top[1] output the OrthoLoss
	//
	// protoformat:
	//	layer {
	//	  name: "coloss"
	//    type: "COLoss"
	//    bottom: "data"
	//    bottom: "label"
	//    top: "closs"
	//	  top: "oloss"
	//    loss_weight: 1
	//	  loss_weight: 0.01
	//	  co_loss_param {
	//		num_output: 10
	//		cutoff: 0
	//		delta: 2
	// }
	//************************************************************************/

	template <typename Dtype>
	class COLossLayer : public Layer<Dtype>
	{
	public:
		explicit COLossLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "COLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		// virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		// virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int num_vec_;
		int dim_vec_;
		int num_output_;

		Dtype cutoff_;
		Dtype delta_;

		Blob<Dtype> ip_vec_;
		Blob<Dtype> trigger_;

		Blob<Dtype> cluster_loss_;
		Blob<Dtype> ortho_loss_;
	};
}  // namespace caffe

#endif  // CAFFE_CO_LOSS_LAYER_HPP_
