#include <boost/asio.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <tuple>
#include <exception>
#include <Connection.hpp>
#include <NN.hpp>
#include <chrono>

using namespace std;

namespace NN {

/// Serves stock quote information to any client that connects to it.
template <class ANN, class DataRange>
class Host
{
private:
	/// The acceptor object used to accept incoming socket Connections.
	boost::asio::ip::tcp::acceptor acceptor;

	typedef sub_range<DataRange> range;
	typedef unique_ptr<range> range_ptr;
	typedef shared_ptr<DeltaMessage> delta_msg_ptr; // should be unique_ptr
	typedef chrono::high_resolution_clock::time_point time_point;
	ANN &ann;
	const DataRange &full_range;
	size_t n_worker;
	vector<range_ptr> ranges;
	vector<conn_ptr> workers;
	vector<delta_msg_ptr> partial_delta;
	vector<bool> ready;

	shared_ptr<InitMessage> init_msg;
	delta_msg_ptr delta_msg;

	time_point start_time;
	vector<time_point> worker_start_time;

	size_t total_epochs;
	size_t trained_epochs;
	
	template <class Continuation>
	static void send_receive(message_ptr out_msg, conn_ptr conn, const MsgCode& exp_code, Continuation c) {
		auto handle_receive = [=](const auto& e, size_t sz, message_ptr in_msg) {
			if (e || !in_msg) { cerr<<"Receive failed: "<<e.message()<<"\n"; c(false, message_ptr()); }
			else if (in_msg->code != exp_code) { cerr<<"Received bad reply: "<<in_msg->code<<"\n"; c(false, in_msg); }
			else c(true, in_msg);
		};
		auto handle_send = [=](const auto& e, size_t sz) {
			if (e) { cerr<<"Send failed: "<<e.message()<<"\n"; c(false, message_ptr()); return; }
			conn->async_read(handle_receive);
		};
		conn->async_write(out_msg, handle_send);
	}

	void close() {
		cout<<"Drop all connections\n"; 
		acceptor.close();
		for (auto & c: workers) c->close();
		workers.clear();
		return;
	}
	void epilogue() {
		cout<<"[host end] "<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count()<<"\n";
		close();
	}
	void print_progress() {
		cout<<"Epoch "<<trained_epochs<<"\n";
	}
	void start_training(bool result, size_t id, message_ptr in_msg) {
		if (!result) { close(); return; }
		ready[id] = true;
		partial_delta[id] = dynamic_pointer_cast<DeltaMessage>(in_msg);

		if (trained_epochs) {
			cout<<"[host receive] "<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - worker_start_time[id]).count()<<"\n";
		}

		for (bool r : ready) if (!r) return;

		if (trained_epochs) {	
			delta_msg->accumulate(partial_delta);
			delta_msg->apply_dw(ann);
		} else {
			// first epoch
			cout<<"[host start]\n";
			start_time = chrono::high_resolution_clock::now();
		}

		if (trained_epochs == total_epochs) { epilogue(); return; }
		ready.flip();
		for (auto& conn : workers) {
			worker_start_time[conn->id] = chrono::high_resolution_clock::now();
			send_receive(delta_msg, conn, MsgCode::Delta_W2H, 
				bind(&Host::start_training, this, placeholders::_1, conn->id, placeholders::_2));
		}
		trained_epochs++;
		print_progress();

	}
	/// Handle completion of a accept operation.
	void handle_accept(const boost::system::error_code& e, conn_ptr conn)
	{	
		if (e) cerr<<"Accept failed: "<<e.message()<<"\n";
		else {
			cout<<"Accepted Connection: "<<workers.size()<<"\n";
			workers.push_back(conn);

			auto send_receive_data = [=](bool result, message_ptr in_msg){
				if (!result) { close(); return; }
				message_ptr data_msg = make_shared<DataMessage>(*ranges[conn->id]);
				send_receive(data_msg, conn, MsgCode::Success_W2H,
					bind(&Host::start_training, this, placeholders::_1, conn->id, placeholders::_2));
			};
			send_receive(init_msg, conn, MsgCode::Success_W2H, send_receive_data);
		}
		if (workers.size() == n_worker) return;
		conn_ptr new_conn(new Connection(acceptor.get_io_service(), workers.size()));
		acceptor.async_accept(new_conn->socket(), 
			bind(&Host::handle_accept, this, placeholders::_1, new_conn));
	}
public:
	/// Constructor opens the acceptor and starts waiting for the first incoming
	/// Connection.
	Host(boost::asio::io_service& io_service, unsigned short port, 
		ANN& _ann, const DataRange & _full_range, size_t _n_worker = 1, size_t _epochs = 100)
	:	acceptor(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)), 
		ann(_ann), full_range(_full_range), n_worker(_n_worker), ranges(n_worker),
		workers(), partial_delta(n_worker), ready(n_worker), init_msg(new InitMessage(ann)),
		delta_msg(new DeltaMessage()), total_epochs(_epochs), trained_epochs(0)
	{
		size_t sub_range_sz = full_range.size() / n_worker;
		for (size_t i = 0; i < n_worker; ++i)
			ranges[i] = make_unique<range>(full_range, i * sub_range_sz,
				min(sub_range_sz, full_range.size() - i * sub_range_sz));
		delta_msg->from_ann(ann);
		// Start an accept operation for a new Connection.
		conn_ptr new_conn(new Connection(acceptor.get_io_service()));
		acceptor.async_accept(new_conn->socket(),
				bind(&Host::handle_accept, this, placeholders::_1, new_conn));
	}
};

} // namespace NN

using namespace NN;
int main(int argc, char* argv[])
{
	try
	{
		// Check command line arguments.
		if (argc != 2)
		{
			std::cerr << "Usage: host <port>" << std::endl;
			return 1;
		}
		unsigned short port = boost::lexical_cast<unsigned short>(argv[1]);

		boost::asio::io_service io_service;

		typedef float data_type;
		typedef NeuralNetwork<data_type> float_network;
		typedef DataSet<data_type> float_dataset;
		typedef Host<float_network, float_dataset> float_server;

        
		float_network ann({784, 800, 400, 10}, ActFn::ReLU, ActFn::Softmax);

		float_dataset ds(true, "data/img_train", "data/label_train", 3000);
		float_server server(io_service, port, ann, ds, 2, 10);
		io_service.run();
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
