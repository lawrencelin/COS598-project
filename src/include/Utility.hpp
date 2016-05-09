#ifndef NN_UTILITY_H
#define NN_UTILITY_H


#include <initializer_list>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <fstream>
#include <iterator>
#include <iostream>
#include <utility>
#include <cassert>
#include <tuple>
#include <numeric>
#include <thread>
#include <chrono>
#include <functional>

namespace NN
{
	using namespace std;

	enum class ErrFn
	{
		SquareError,
		CrossEntropy
	};
	template <class T>
	void print_V2D(ostream& os, const vector<vector<T>> & v2d){
		for (auto & v1d : v2d){
			for (auto & t: v1d) os<<t<<" ";
			cerr<<"\n";
		}
	}
	template <class T>
	void print_V3D(ostream& os, const vector<vector<vector<T>>> & v3d){
		for (auto & v2d : v3d){
			for (auto & v1d : v2d){
				for (auto & t: v1d) os<<t<<" ";
				os<<"\n";
			}
			os<<"==================\n";
		}
	}

	template <class Ty>
	class DataSet
	{
	private:
		size_t in_sz;
		size_t out_sz;
		typedef pair<vector<Ty>, vector<Ty>> sample;
		vector<sample> data;
	public:
		typedef typename vector<sample>::const_iterator sample_iter;
		DataSet(const char *f, size_t n, size_t _in_sz, size_t _out_sz)  // if n * (in_sz + out_sz) > #of data in file, this function crashes.
		:	in_sz(_in_sz), out_sz(_out_sz), 
			data(n, sample(piecewise_construct, forward_as_tuple(in_sz), forward_as_tuple(out_sz)))
		{
			ifstream fs(f);
			assert("Cannot open file" && fs.is_open());
			istream_iterator<Ty> file_reader(fs);
			for (sample &dp : data) {
				for (Ty &ref : dp.first)
					ref = *(file_reader++);
				for (Ty &ref : dp.second)
					ref = *(file_reader++);
			}
		}
		DataSet(bool mnist_placeholder, const char *f_img, const char *f_label, size_t n)
		:	in_sz(784), out_sz(10), 
			data(n, sample(piecewise_construct, forward_as_tuple(in_sz), forward_as_tuple(out_sz)))
		{
			(void)mnist_placeholder;
			ifstream fs_img(f_img);
			ifstream fs_label(f_label);
			fs_img.seekg(16);
			fs_label.seekg(16);
			for (sample &dp : data) {
				for (Ty &pixel: dp.first) pixel = fs_img.get() / (Ty)255.0;
				dp.second[(size_t)fs_label.get()] = 1; // one-hot
			}
		}
		sample_iter begin() const { return data.cbegin(); }
		sample_iter end() const { return data.cend(); }
		size_t size() const {return data.size(); }
	};

	template<class T>
	class sub_range {
	private:
		typedef typename T::sample_iter iter;
		const T &t;
		size_t start, sz;
	public:
		sub_range(const T &_t, size_t _start, size_t _sz)
		: t(_t), start(_start), sz(_sz) {}
		sub_range(const sub_range&) = delete;
		iter begin() const { return t.begin() + (long)start; }
		iter end() const { return t.begin() + (long)start + (long)sz;}
		size_t size() const { return sz; }
	};
}

#endif /*NN_UTILITY_H*/
