#ifndef NN_MULTITHREADTRAINER_H
#define NN_MULTITHREADTRAINER_H

#include <Trainer.hpp>

namespace NN
{
	using namespace std;

	enum class JobCode{
		NOP,
		Compute,
		Exit
	};
	enum class ReplyCode{
		NOP,
		Completed
	};

	inline ostream& operator<<(ostream &os, const enum JobCode j) {
		string str;
		switch (j){
			case JobCode::NOP: str = "NOP"; break;
			case JobCode::Compute: str = "ComputeGradient"; break;
			case JobCode::Exit: str = "Exit"; break;
		}
		return os<<"JobCode::"+str;
	}

	inline ostream& operator<<(ostream &os, const enum ReplyCode r) {
		string str;
		switch (r){
			case ReplyCode::NOP: str = "NOP"; break;
			case ReplyCode::Completed: str = "Completed"; break;
		}
		return os<<"ReplyCode::"+str;
	}

	template<class First, class Second>
	inline ostream& operator<<(ostream &os, const pair<First, Second> &p) {
		return os<<"("<<p.first<<", "<<p.second<<")";
	}

	template<class T>
	class sync_channel{
	private:
		condition_variable cv;
		mutex cv_m;
		string name;
		vector<T> v;
	public:
		sync_channel(const string _name, const size_t _n): name(_name), v(_n) {
			assert("Must not be empty!" && _n);
		}
		void set(size_t k, T t = T()) {
			unique_lock<mutex> l(cv_m);
			v[k] = t;
			cv.notify_all();
		}
		void set_all(T t = T()) {
			unique_lock<mutex> l(cv_m);
			fill(v.begin(), v.end(), t);
			cv.notify_all();
		}
		T get(size_t k) {
			unique_lock<mutex> l(cv_m);
			cv.wait(l, [this, k](){return this->v[k] != T();});
			T t = v[k];
			v[k] = (T)0;
			return t;
		}
		vector<T> get_all() {
			unique_lock<mutex> l(cv_m);
			cv.wait(l, [this](){return count(this->v.begin(), this->v.end(), T()) == 0;});
			vector<T> ret = v;
			fill(v.begin(), v.end(), T());
			return ret;
		}
	};

	template <class ANN, class DataRange>
	class MultithreadTrainer 
	{
	private:
		typedef typename ANN::data_type data_type;
		typedef vector<data_type> V1D;
		typedef vector<V1D> V2D;
		typedef vector<V2D> V3D;
		typedef sub_range<DataRange> range;
		typedef Trainer<ANN, range> single_trainer;
		
		ANN &ann;
		const DataRange& full_range;
		V3D prev_dw;
		V2D prev_db;

		data_type learning_rate;
		data_type momentum;

		vector<unique_ptr<range>> ranges; // need heap storage to prevent worker threads accessing corrupted (out-dated) stack frames
		vector<unique_ptr<single_trainer>> trainers; // same heap storage
		vector<thread> threads;

		vector<unique_ptr<range>> test_ranges;
		size_t test_size;

		typedef pair<ReplyCode, data_type> reply;
		sync_channel<JobCode> job_chan;
		sync_channel<reply> reply_chan;

		const ErrFn err_fn;

		void worker_thread_loop(single_trainer *trainer, size_t id) {
			while (1){
				switch(job_chan.get(id)){
					case JobCode::NOP: assert("received NOP" && false);
				case JobCode::Compute:
						reply_chan.set(id, reply(ReplyCode::Completed, trainer->train_multi_thread()));
						break;
					case JobCode::Exit:
						return;
				}
			}
		}

		void apply_dw() 
		{
			data_type c = learning_rate / full_range.size();
			size_t l = 0;
			typedef typename single_trainer::V3D_iter V3D_iter;
			typedef typename single_trainer::V2D_iter V2D_iter;
			vector<V3D_iter> dw_iters;
			vector<V2D_iter> db_iters;
			transform(trainers.begin(), trainers.end(), back_inserter(dw_iters), [](const auto& t){return t->dw_begin();});
			transform(trainers.begin(), trainers.end(), back_inserter(db_iters), [](const auto& t){return t->db_begin();});
			for (auto &layer: ann){
				size_t j = 0;
				auto bj_iter = layer->bias_begin();
				vector<V2D_iter> dwj_iters;
				transform(dw_iters.begin(), dw_iters.end(), back_inserter(dwj_iters), [j](const auto& it){ return it->cbegin(); });
				for (auto &wj: *layer){
					size_t i = 0;
					for (auto & wij: wj){
						data_type sum_dw = 0;
						for (auto & it: dwj_iters) sum_dw += it->at(i);
						wij -= (prev_dw[l][j][i] = c * sum_dw + momentum * prev_dw[l][j][i]);
						++i;
					}
					data_type sum_db = 0;
					for (auto & it: db_iters) sum_db += it->at(j);
					*bj_iter -= (prev_db[l][j] = c * sum_db + momentum * prev_db[l][j]);
					for (auto & it: dwj_iters) ++it;
					++j, ++bj_iter;
				}
				for (auto & it: dw_iters) ++it;
				for (auto & it: db_iters) ++it;
				++l;
			}
		}
		data_type compute_error(data_type err, size_t sz) {
			switch (err_fn) {
				case ErrFn::SquareError:  return sqrt(err/sz);
				case ErrFn::CrossEntropy: return err/sz;
			}
		}
		void print_progress(data_type err, size_t i, size_t epochs, size_t p = 100) {
			size_t segment = epochs / p;
			if (segment && i % segment && i != epochs) return;
			double percentage = 100.0 * i / epochs;
			cout<<percentage<<"% epoch "<<i<<" training error: "<<compute_error(err, full_range.size())<<"\n";
			if (test_size){	
				err = 0;
				for (auto &t : trainers) err += t->get_error_multi_thread(true);
				cout<<"         testing error: "<<compute_error(err, test_size)<<"\n";
			}
			if (i == epochs) ann.print();
		}
	public:
		MultithreadTrainer(ANN &_ann, const DataRange &data_range, size_t n_t = 4,
			data_type alpha = (data_type)0.01, data_type beta = (data_type)0.9, ErrFn _err_fn = ErrFn::SquareError)
		:	ann(_ann), full_range(data_range), prev_dw(ann.size()), prev_db(ann.size()),
			learning_rate(alpha), momentum(beta), ranges(n_t), trainers(n_t), threads(n_t), test_ranges(n_t), test_size(0),
			job_chan("{main->worker}", n_t), reply_chan("{worker->main}", n_t), err_fn(_err_fn)
		{
			size_t n_hw_t = thread::hardware_concurrency();
			if (n_hw_t < n_t) cerr<<"WARNING: requesting more threads("<<n_t<<") than the hardware supports("<<n_hw_t<<").\n";
			size_t sub_range_sz = data_range.size() / n_t;
			assert("more threads than data samples" && sub_range_sz);
			
			size_t l = 0;
			for (auto &layer : ann){
				prev_db[l].resize(layer->size());
				prev_dw[l].resize(layer->size(), V1D(layer->input_size()));
				++l;
			}
			for (size_t i = 0; i < n_t; ++i) {
				ranges[i] = make_unique<range>(full_range, i * sub_range_sz, min(sub_range_sz, full_range.size() - i * sub_range_sz));
				trainers[i] = make_unique<single_trainer>(true, ann, *(ranges[i].get()), err_fn);
				threads[i] = thread(&MultithreadTrainer::worker_thread_loop, this, trainers[i].get(), i);
				// pointers in ranges / thrainers will outlive threads
				// because they are deleted after the destructor ~MultithreadTrainer which joins the threads
			}
		}
		void set_test_data(const DataRange &test_range) { 
			test_size = test_range.size();
			size_t sub_range_sz = test_size / trainers.size();
			size_t i = 0;
			for (auto &t : trainers) {
				test_ranges[i] = make_unique<range>(test_range, i * sub_range_sz, min(sub_range_sz, test_size - i * sub_range_sz));
				t->set_test_data(test_ranges[i].get());
				++i;
			} 
		}

		void train(size_t epochs, size_t p = 100) 
		{
			using namespace chrono;
			auto dispatch_time = high_resolution_clock::duration::zero();
			auto gather_time = high_resolution_clock::duration::zero();

			data_type sum_err = 0;
			for (size_t i = 0; i < epochs; ++i) {
				auto tp = high_resolution_clock::now();
				job_chan.set_all(JobCode::Compute);
				vector<reply> r = reply_chan.get_all();
				dispatch_time += high_resolution_clock::now() - tp;
				sum_err = 0;
				for (auto& rep : r){
					assert("bad job" && rep.first == ReplyCode::Completed);
					sum_err += rep.second;
				}
				print_progress(sum_err, i, epochs, p);
				tp = high_resolution_clock::now();
				apply_dw();
				gather_time += high_resolution_clock::now() - tp;
			}
			sum_err = 0;
			for (auto &t : trainers) sum_err += t->get_error_multi_thread(false);
			print_progress(sum_err, epochs, epochs);


			cout<<"dispatch_time:"<<duration_cast<seconds>(dispatch_time).count()<<" gather_time:"<<duration_cast<seconds>(gather_time).count()<<"\n";
		}
		~MultithreadTrainer() 
		{
			job_chan.set_all(JobCode::Exit);
			for (thread &worker : threads) worker.join();
		}
	};
}

#endif /*NN_MULTITHREADTRAINER_H*/
