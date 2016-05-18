#ifndef NN_CONNECTION_HPP
#define NN_CONNECTION_HPP

#include <boost/asio.hpp>
#include <functional>
#include <vector>
#include <iostream>

#include <NN.hpp>


namespace NN 
{

using namespace std;

enum class MsgCode
{
	// base Message
	NONE,
	Exit_H2W,
	Success_W2H,
	Fail_W2H,
	// derived InitMessage
	InitNetwork_H2W,
	// derived DataMessage
	TrainingData_H2W,
	// derived DeltaMessage
	ApplyAndTrain_H2W,
	Delta_W2H
	// H2W: Host to Worker, W2H: Worker to Host
};
template <class S>
S &operator<<(S &s, const MsgCode& code) {
	string str;
	switch (code) {
		case MsgCode::NONE:              str = "NONE";              break;
		case MsgCode::Exit_H2W:          str = "Exit_H2W";          break;
		case MsgCode::Success_W2H:       str = "Success_W2H";       break;
		case MsgCode::Fail_W2H:          str = "Fail_W2H";          break;
		case MsgCode::InitNetwork_H2W:   str = "InitNetwork_H2W";   break;
		case MsgCode::TrainingData_H2W:  str = "TrainingData_H2W";  break;
		case MsgCode::ApplyAndTrain_H2W: str = "ApplyAndTrain_H2W"; break;
		case MsgCode::Delta_W2H:         str = "Delta_W2H";         break;
	}
	return s<<"MsgCode::"+str;
}

class Message
{
public:
	MsgCode code;
	Message(MsgCode _code = MsgCode::NONE): code(_code), data(new vector<char>()) {}
	virtual size_t data_size() { return data->size(); }
	virtual shared_ptr<vector<char>> to_data() { return data; }
	virtual void from_data(vector<char> *buf) { data.reset(buf); }
	virtual ~Message() {}
	// virtual string what() { return "This is a message."; }
protected:
	shared_ptr<vector<char>> data;
};
class InitMessage: public Message
{
public:
	vector<size_t> sizes;
	ActFn hidden_type;
	ActFn output_type;
	
	InitMessage(): Message(MsgCode::InitNetwork_H2W) {}
	
	template <class ANN>
	InitMessage(const ANN& ann): InitMessage() {
		size_t l = 0;
		for (auto& layer: ann) {
			if (!l) sizes.push_back(layer->input_size());
			sizes.push_back(layer->size());
			++l;
		}
		hidden_type = ann.hidden_type;
		output_type = ann.output_type;
		data->resize(sizeof(size_t) * (sizes.size() + 2));
		size_t *ptr = copy(sizes.begin(), sizes.end(), (size_t*)(data->data()));
		*ptr = (size_t)hidden_type;
		*(ptr+1) = (size_t)output_type;
	}

	void from_data(vector<char> *buf) override {
		// TODO verification
		Message::from_data(buf);
		size_t *begin = (size_t *)data->data();
		size_t *end = begin + data->size() / sizeof(size_t) - 2;
		sizes.assign(begin, end);
		hidden_type = (ActFn)*end;
		output_type = (ActFn)*(end+1);
	}


};
class DataMessage: public Message
{
private:
	size_t n_samples;
	size_t in_sz;
	size_t out_sz;
	float* val;

	static const size_t data_offset = 4 * sizeof(size_t);

	class pointer_wrapper {
	private:
		const float *base;
		const size_t sz;
	public:
		pointer_wrapper(const float *_base, const size_t& _sz): base(_base), sz(_sz) {}
		const float *cbegin() const { return base; }
		const float *cend() const { return base + sz; }
		const size_t size() const { return sz; }
		const float &operator[](size_t idx) const { return base[idx]; }
	};
	class sample {
	private:
		const float *base;
		const size_t in_sz, out_sz;
	public:
		pointer_wrapper first, second;
		sample(const float *_base, const size_t& _in_sz, const size_t& _out_sz)
		:	base(_base), in_sz(_in_sz), out_sz(_out_sz), 
			first(base, in_sz), second(base + in_sz, out_sz) {}
	};
public:
	class sample_iter : public iterator<input_iterator_tag, sample>{
	private:
		const DataMessage &base;
		size_t offset;
	public:
		sample_iter(const DataMessage& _base, size_t _offset) : base(_base), offset(_offset) {}

		sample_iter(const sample_iter& rhs) : base(rhs.base), offset(rhs.offset) {}
		sample_iter& operator++() { ++offset; return *this; }
		sample_iter& operator+(long diff) { offset += diff; return *this; }
		sample_iter operator++(int) { sample_iter tmp = sample_iter(*this); operator++(); return tmp; }
		bool operator==(const sample_iter& rhs) { return &base == &(rhs.base) && offset == rhs.offset; }
		bool operator!=(const sample_iter& rhs) { return !operator==(rhs); }
		const sample operator*() { return sample(base.base_ptr(), base.input_size(), base.output_size()); }
	};
	DataMessage() : Message(MsgCode::TrainingData_H2W), val(nullptr) {}
	
	template <class Range>
	DataMessage(const Range& range): Message(MsgCode::TrainingData_H2W),
		n_samples(range.size()), in_sz(range.input_size()), out_sz(range.output_size()) 
	{
		data->resize(data_offset + n_samples * (in_sz + out_sz) * sizeof(float));
		size_t *sz_ptr = (size_t *)(data->data());
		
		*(sz_ptr++) = n_samples;
		*(sz_ptr++) = in_sz;
		*(sz_ptr++) = out_sz;
		
		float *f_ptr = val = (float *)(data->data() + data_offset);
		for (auto& sample : range) {
			for (auto& v : sample.first) *(f_ptr++) = v;
			for (auto& v : sample.second) *(f_ptr++) = v;
		}
	}

	void from_data(vector<char> *buf) override{
		Message::from_data(buf);
		size_t *sz_ptr = (size_t *)(data->data());
		n_samples = *(sz_ptr++);
		in_sz = *(sz_ptr++);
		out_sz = *(sz_ptr++);
		val = (float *)(data->data() + data_offset);
	}

	sample_iter begin() const { return {*this, 0}; }
	sample_iter end() const { return {*this, n_samples}; }

	float *base_ptr() const { return val; }
	size_t size() const { return n_samples; }
	size_t input_size() const { return in_sz; }
	size_t output_size() const { return out_sz; }

};

class DeltaMessage: public Message
{
private:
	float *val;
	size_t n_params;
public:

	DeltaMessage(const MsgCode& _code = MsgCode::ApplyAndTrain_H2W): Message(_code), val(nullptr), n_params(0) {}

	template <class ANN>
	void from_ann(const ANN& ann) {
		n_params = 0;
		for (auto& layer: ann)
			n_params += (layer->input_size() + 1) * layer->size();
		data->resize(n_params * sizeof(float));

		float *ptr = val = (float *)(data->data());

		for (auto &layer: ann) {
			for (auto &w_j: *layer)
				for (auto & w_ij: w_j) *(ptr++) = w_ij;
			size_t j = layer->size();
			auto b_j = layer->bias_begin();
			while (j--) *(ptr++) = *(b_j++);
		}
	}
	
	template <class Trainer>
	DeltaMessage(const Trainer& t) : Message(MsgCode::Delta_W2H) {
		n_params = 0;
		for (auto &w : t.get_dw())
			n_params += w.size() * (w[0].size() + 1);
		data->resize(n_params * sizeof(float));
		float *ptr = val = (float *)(data->data());

		size_t l = 0;
		auto b = t.get_db().cbegin();
		for (auto &w : t.get_dw()) {
			for (auto &w_j: w)
				for (auto & w_ij: w_j) 
					*(ptr++) = w_ij;
			for (auto &b_j : *b) 
				*(ptr++) = b_j;
			++l, ++b;
		}
	}

	void from_data(vector<char> *buf) override {
		Message::from_data(buf);
		val = (float *)(data->data());
		n_params = data->size() / sizeof(float);
	}

	template <class ANN>
	void apply_dw(ANN & ann) {
		float *ptr = val;
		for (auto &layer: ann) {
			for (auto &w_j: *layer) 
				for (auto & w_ij: w_j) w_ij -= *(ptr++);
			auto b_j = layer->bias_begin();
			size_t j = layer->size();
			while (j--) *(b_j++) -= *(ptr++);
		}
	}

	void accumulate(const vector<shared_ptr<DeltaMessage>> & partial_delta) {
		vector<float *> partial_ptrs;
		transform(partial_delta.begin(), partial_delta.end(), 
			back_inserter(partial_ptrs), [](auto&ptr) { return ptr->val; });
		float *ptr = val;
		for (size_t i = 0; i < n_params; ++i) {
			ptr[i] = 0;
			for (auto& it : partial_ptrs) 
				ptr[i] += *(it++);
		}
	}
};

typedef shared_ptr<Message> message_ptr;

class Connection
{
public:
	const size_t id;
	/// Constructor.
	Connection(boost::asio::io_service& io_service, size_t _id = 0) 
	:	id(_id), socket_(io_service) {}

	boost::asio::ip::tcp::socket& socket() { return socket_; }

	template <typename Handler>
	void async_write(message_ptr msg, Handler handler)
	{
		if (!msg) {
			socket_.get_io_service().post(
				bind(handler, boost::system::error_code(boost::asio::error::invalid_argument), 0));
			return;
		}
		
		*(size_t*)out_header = (size_t)msg->code;
		*(size_t*)(out_header+sizeof(size_t)) = msg->data_size();
		out_data = msg->to_data();
		vector<boost::asio::const_buffer> buffers = {
			boost::asio::buffer(out_header),
			boost::asio::buffer(*out_data)
		};
		boost::asio::async_write(socket_, buffers, handler);
	}

	/// Asynchronously read a data structure from the socket.
	template <typename Handler>
	void async_read(Handler handler)
	{
		boost::asio::async_read(socket_, boost::asio::buffer(in_header),
			[=](auto& e, size_t sz) {
				if (e) { handler(e, sz, message_ptr()); return; }
				size_t code = *(size_t*)in_header;
				size_t in_sz = *(size_t*)(in_header+sizeof(size_t));

				vector<char> *in_data = new vector<char>(in_sz);
				message_ptr msg;
				switch ((MsgCode)code) {
					case MsgCode::NONE:
					case MsgCode::Exit_H2W:
					case MsgCode::Success_W2H:
					case MsgCode::Fail_W2H:
						msg = make_shared<Message>((MsgCode)code);
						break;
					case MsgCode::InitNetwork_H2W:
						msg = make_shared<InitMessage>();
						break;
					case MsgCode::TrainingData_H2W:
						msg = make_shared<DataMessage>();
						break;
					case MsgCode::ApplyAndTrain_H2W:
					case MsgCode::Delta_W2H:
						msg = make_shared<DeltaMessage>((MsgCode)code);
						break;
				}
				boost::asio::async_read(socket_, boost::asio::buffer(*in_data),
					[=](auto& e, size_t sz) {
						if (e) { handler(e, sz, message_ptr()); return; }
						msg->from_data(in_data);
						handler(e, sz, msg);
					});
				in_data = nullptr;
			});
	}

	void close() {
		socket_.close();
	}

	~Connection() {
		cout<<"Connection closed\n";
	}
private:
	/// The underlying socket.
	boost::asio::ip::tcp::socket socket_;
	static const size_t header_length = 2 * sizeof(size_t);
	char out_header[header_length];
	char in_header[header_length];
	/// Holds the outbound data.
	shared_ptr<vector<char>> out_data;
	// unique_ptr<vector<char>> in_data;
};

typedef shared_ptr<Connection> conn_ptr;

} // namespace NN

#endif // NN_CONNECTION_HPP
