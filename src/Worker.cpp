#include <boost/asio.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <Connection.hpp>
#include <NN.hpp>

using namespace std;

namespace NN {

/// Downloads stock quote information from a server.
class Worker
{
private:
	typedef NeuralNetwork<float> ANN;
	typedef MultithreadTrainer<ANN, DataMessage> MT;
	typedef unique_ptr<ANN> ann_ptr;
	typedef shared_ptr<DataMessage> data_ptr;
	typedef unique_ptr<MT> trainer_ptr;
	conn_ptr conn;
	ann_ptr ann;
	data_ptr data;
	trainer_ptr trainer;
	// unique_ptr<NeuralNetwork<float>> ann_ptr;
	template <class Operation, class Continuation>
	static void receive_send(const MsgCode& exp_code, Operation op, conn_ptr conn, Continuation c) {
		auto handle_send = [=](const auto& e, size_t sz) {
			if (e) { cerr<<"Send failed: "<<e.message()<<"\n"; c(false, message_ptr()); return; }
			c(true, message_ptr());
		};
		auto handle_receive = [=](const auto& e, size_t sz, message_ptr in_msg) {
			if (e || !in_msg) { cerr<<"Receive failed: "<<e.message()<<"\n"; c(false, message_ptr()); }
			else if (in_msg->code != exp_code) { cerr<<"Received bad reply: "<<in_msg->code<<" expecting: "<<exp_code<<"\n"; c(false, in_msg); }
			else {
				message_ptr out_msg = op(in_msg);
				conn->async_write(out_msg, handle_send);
			}
		};
		conn->async_read(handle_receive);
	}
	void handle_connect(const boost::system::error_code& e)
	{
		if (e) { std::cerr << e.message() << std::endl; return; }
		
		auto init_handler = [=](message_ptr in_msg) {
			auto init_msg = dynamic_pointer_cast<InitMessage>(in_msg);
			ann = make_unique<ANN>(init_msg->sizes, init_msg->hidden_type, init_msg->output_type);
			return make_shared<Message>(ann? MsgCode::Success_W2H: MsgCode::Fail_W2H);
			// return make_shared<Message>(MsgCode::Success_W2H);
		};
		auto data_handler = [=](message_ptr in_msg) {
			data = dynamic_pointer_cast<DataMessage>(in_msg);
			return make_shared<Message>(data? MsgCode::Success_W2H: MsgCode::Fail_W2H);
		};
		auto continuation = [=](bool success, message_ptr in_msg) {
			if (!success) conn->close();
			else receive_send(MsgCode::TrainingData_H2W, data_handler, conn,
					bind(&Worker::train_loop, this, placeholders::_1, placeholders::_2));
		};
		receive_send(MsgCode::InitNetwork_H2W, init_handler, conn, continuation);
	}

	void train_loop(bool success, message_ptr in_msg) {
		if (!success) { conn->close(); return; }
		if (!trainer) trainer = make_unique<MT>(*ann, *data);
		auto train_once = [=](message_ptr in_msg) {
			dynamic_pointer_cast<DeltaMessage>(in_msg)->apply_dw(*ann);
			trainer->train_once();
			return dynamic_pointer_cast<Message>(make_shared<DeltaMessage>(*trainer));
		};
		receive_send(MsgCode::ApplyAndTrain_H2W, train_once, conn,
			bind(&Worker::train_loop, this, placeholders::_1, placeholders::_2));
	}

public:
	/// Constructor starts the asynchronous connect operation.
	Worker(boost::asio::io_service& io_service,
			const std::string& host, const std::string& service)
		: conn(new Connection(io_service))
	{
		// Resolve the host name into an IP address.
		boost::asio::ip::tcp::resolver resolver(io_service);
		boost::asio::ip::tcp::resolver::query query(host, service);
		boost::asio::ip::tcp::resolver::iterator endpoint_iterator =
			resolver.resolve(query);

		// Start an asynchronous connect operation.
		boost::asio::async_connect(conn->socket(), endpoint_iterator,
				bind(&Worker::handle_connect, this, placeholders::_1));
	}
};

} 
using namespace NN;

int main(int argc, char* argv[])
{
	try
	{
		// Check command line arguments.
		if (argc != 3)
		{
			std::cerr << "Usage: worker <host> <port>" << std::endl;
			return 1;
		}

		boost::asio::io_service io_service;
		Worker worker(io_service, argv[1], argv[2]);
		io_service.run();
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
