#include <vector>
#include <string>

#include "caffe/filler.hpp"
#include "caffe/layers/co_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
	template <typename Dtype>
	void COLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		num_vec_ = bottom[0]->shape(0);
		dim_vec_ = 1;
		for (int i = 1; i < bottom[0]->num_axes(); i++)
			dim_vec_ *= bottom[0]->shape(i);

		num_output_ = this->layer_param_.co_loss_param().num_output();
		cutoff_ = this->layer_param_.co_loss_param().cutoff();
		delta_ = this->layer_param_.co_loss_param().delta();

		CHECK_EQ(num_vec_, bottom[1]->shape(0));

		// Intialize the center
		if (this->blobs_.size() > 0)
			LOG(INFO) << "Skipping parameter initialization";
		else
		{
			this->blobs_.resize(1);
			vector<int> center_shape = { num_output_, dim_vec_ };
			this->blobs_[0].reset(new Blob<Dtype>(center_shape));
			shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(this->layer_param_.js_loss_param().center_filler()));
			center_filler->Fill(this->blobs_[0].get());
		}
		this->param_propagate_down_.resize(this->blobs_.size(), true);

		// Allocate memory
		vector<int> cluster_loss_shape = { num_vec_ };
		cluster_loss_.Reshape(cluster_loss_shape);

		vector<int> ip_vec_shape = { num_vec_, num_output_ };
		ip_vec_.Reshape(ip_vec_shape);
		trigger_.Reshape(ip_vec_shape);
		ortho_loss_.Reshape(ip_vec_shape);
	}

	template <typename Dtype>
	void COLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		vector<int> scalar_shape(0);
		top[0]->Reshape(scalar_shape);
		top[1]->Reshape(scalar_shape);
	}

	template <typename Dtype>
	void COLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		const Dtype* data = bottom[0]->cpu_data();
		const Dtype* label = bottom[1]->cpu_data();
		Dtype* center = this->blobs_[0]->mutable_cpu_data();

		// Valuation
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, num_vec_, num_output_, dim_vec_, (Dtype)1, data, center, (Dtype)0, ip_vec_.mutable_cpu_data());

		// Calculate ortho_loss 
		for (int i = 0; i < num_vec_; i++)
		{
			const int label_value = static_cast<int>(label[i]);
			int offset = i*num_output_;
			for (int j = 0; j < num_output_; j++)
			{
				if (j != label_value)
				{
					if (ip_vec_.cpu_data()[offset + j] > 0)
					{
						trigger_.mutable_cpu_data()[offset + j] = 1;
						ortho_loss_.mutable_cpu_data()[offset + j] = ip_vec_.cpu_data()[offset + j];
					}
					else
					{
						trigger_.mutable_cpu_data()[offset + j] = 0;
						ortho_loss_.mutable_cpu_data()[offset + j] = 0;
					}
				}
				else
				{
					trigger_.mutable_cpu_data()[offset + j] = 0;
					ortho_loss_.mutable_cpu_data()[offset + j] = 0;
				}
			}
		}
		Dtype ortho_loss = caffe_cpu_asum(num_vec_*num_output_, ortho_loss_.cpu_data());
		
		// Calculate cluster_loss 
		for (int i = 0; i < num_vec_; i++)
		{
			const int label_value = static_cast<int>(label[i]);

			Dtype dot = ip_vec_.cpu_data()[i*num_output_ + label_value];
			if (dot <= cutoff_)
				dot = cutoff_;

			cluster_loss_.mutable_cpu_data()[i] = (Dtype)1 / (dot + delta_);
		}
		Dtype cluster_loss = caffe_cpu_asum(num_vec_, cluster_loss_.cpu_data());

		top[0]->mutable_cpu_data()[0] = cluster_loss / (Dtype)num_vec_;
		top[1]->mutable_cpu_data()[0] = ortho_loss / ((Dtype)2 * num_vec_);
	}

	template <typename Dtype>
	void COLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		const Dtype* data = bottom[0]->cpu_data();
		const Dtype* label = bottom[1]->cpu_data();
		Dtype* data_diff = bottom[0]->mutable_cpu_diff();
		Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
		const Dtype* center = this->blobs_[0]->cpu_data();

		// Gradient with respect to center
		if (this->param_propagate_down_[0])
		{
			caffe_set(num_output_*dim_vec_, (Dtype)0, center_diff);

			// ortho_loss
			Dtype alpha1 = top[1]->cpu_diff()[0] / (Dtype)num_vec_;
			caffe_cpu_gemm(CblasTrans, CblasNoTrans, num_output_, dim_vec_, num_vec_, alpha1, trigger_.cpu_data(), data, (Dtype)1, center_diff);
			for (int j = 0; j < num_output_; j++)
			{
				Dtype count = 1;
				for (int k = 0; k < num_vec_; k++)
					count += trigger_.cpu_data()[k*num_output_ + j];
				caffe_cpu_scale(dim_vec_, Dtype(1) / count, center_diff + j*dim_vec_, center_diff + j*dim_vec_);
			}

			// cluster_loss
			Dtype alpha2 = top[0]->cpu_diff()[0] / (Dtype)num_vec_;
			for (int j = 0; j < num_output_; j++)
			{
				for (int i = 0; i < num_vec_; i++)
				{
					const int label_value = static_cast<int>(label[i]);
					if (label_value == j)
					{
						Dtype alpha = -alpha2 * cluster_loss_.cpu_data()[i] * cluster_loss_.cpu_data()[i];
						caffe_cpu_axpby(dim_vec_, alpha, data + i*dim_vec_, (Dtype)1, center_diff + j*dim_vec_);
					}
				}
			}
		}
		
		// Gradient with respect to bottom
		if (propagate_down[0])
		{
			// ortho_loss
			Dtype alpha1 = top[1]->cpu_diff()[0] / (Dtype)num_vec_;
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_vec_, dim_vec_, num_output_, alpha1, trigger_.cpu_data(), center, (Dtype)0, data_diff);
			
			// cluster_loss
			Dtype alpha2 = top[0]->cpu_diff()[0] / (Dtype)num_vec_;
			for (int i = 0; i < num_vec_; i++)
			{
				const int label_value = static_cast<int>(label[i]);
				Dtype alpha = -alpha2*cluster_loss_.cpu_data()[i] * cluster_loss_.cpu_data()[i];
				caffe_cpu_axpby(dim_vec_, alpha, center + label_value*dim_vec_, (Dtype)1, data_diff + i*dim_vec_);
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(COLossLayer);
#endif
	INSTANTIATE_CLASS(COLossLayer);
	REGISTER_LAYER_CLASS(COLoss);
}  // namespace caffe
