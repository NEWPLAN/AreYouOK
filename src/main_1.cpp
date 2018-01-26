#include <thread>
#include <iostream>
#include <future>
#include <chrono>

struct _data
{
	int32_t value;
};

_data data = { 0 };

int main_1()
{
	std::promise<_data> data_promise;      //创建一个承诺
	std::future<_data> data_future = data_promise.get_future();     //得到这个承诺封装好的期望

	std::thread prepare_data_thread([](std::promise<_data> &data_promise)
	{
		std::this_thread::sleep_for(std::chrono::seconds(2));    //模拟生产过程

		data_promise.set_value({ 1 });       //通过set_value()反馈结果
	}, std::ref(data_promise));

	std::thread process_data_thread([](std::future<_data> &data_future)
	{
		std::cout << data_future.get().value << std::endl;    //通过get()获取结果
	}, std::ref(data_future));
	std::cout << "main thread is waiting for subthread." << std::endl;
	std::this_thread::get_id();
	prepare_data_thread.join();
	process_data_thread.join();

	system("pause");
	return 0;
}