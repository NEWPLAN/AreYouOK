#include "threadpool.h"
#include <iostream>
#include <mutex>
#include <cmath>


void fun1(int slp)
{
	std::cout << "  hello, fun1 !  " << std::this_thread::get_id() << std::endl;
	if (slp > 0)
	{
		std::cout << "======= fun1 sleep " << slp << "  =========  " << std::this_thread::get_id() << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(slp));
	}
}

struct gfun
{
	int operator()(int n)
	{
		std::cout << n << "  hello, gfun !  " << std::this_thread::get_id();
		return 42;
	}
};

class A      //'static' decorate is must for threadpool
{
public:
	static int Afun(int n = 0)
	{
		std::cout << n << "  hello, Afun !  " << std::this_thread::get_id() << std::endl;
		return n;
	}

	static std::string Bfun(int n, std::string str, char c)
	{
		std::cout << n << "  hello, Bfun !  " << str.c_str() << "  " << (int)c << "  " << std::this_thread::get_id() << std::endl;
		return str;
	}
};

/*a log show info orderly*/
static void log_info(int index, bool show_res = false)
{
	return;
	static std::mutex mymutex;
	if (!show_res)
	{
		//std::this_thread::sleep_for(std::chrono::milliseconds(1));

		std::lock_guard<std::mutex> mylock(mymutex);
		std::cout << "hello, " << index << std::endl;
	}
	else
	{
		std::lock_guard<std::mutex> mylock(mymutex);
		std::cout << "the sqare of " << sqrt(index) << " is : " << index << std::endl;
	}
}

static void func_thread_test(void)
{
	std::threadpool executor{ 50 };
	A a;
	std::future<void> ff = executor.commit(fun1, 0);
	std::future<int> fg = executor.commit(gfun{}, 0);
	std::future<int> gg = executor.commit(&(a.Afun), 9999);
	std::future<std::string> gh = executor.commit(A::Bfun, 9998, "mult args", 123);
	std::future<std::string> fh = executor.commit([]()->std::string { std::cout << "hello, fh !  " << std::this_thread::get_id() << std::endl; return "hello,fh ret !"; });

	std::cout << " =======  sleep ========= " << std::this_thread::get_id() << std::endl;
	std::this_thread::sleep_for(std::chrono::microseconds(900));

	for (int i = 0; i < 50; i++)
	{
		executor.commit(fun1, i * 100);
	}
	std::cout << " =======  commit all ========= " << std::this_thread::get_id() << " idlsize=" << executor.idlCount() << std::endl;

	std::cout << " =======  sleep ========= " << std::this_thread::get_id() << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(3));

	ff.get(); //get() is called to get result after execution of some thread is done
	std::cout << fg.get() << "  " << fh.get().c_str() << "  " << std::this_thread::get_id() << std::endl;

	std::cout << " =======  sleep ========= " << std::this_thread::get_id() << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(3));

	std::cout << " =======  fun1,55 ========= " << std::this_thread::get_id() << std::endl;
	executor.commit(fun1, 55).get();   //get() is called to get result after execution of some thread is done

	std::cout << "end... " << std::this_thread::get_id() << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(2));
}

int main(int argc, char** argv)
{
	try
	{
		//func_thread_test();


		std::threadpool pool(4);/*N-size threadpool is created*/
		std::vector< std::future<int> > results;


		for (int i = 0; i < 2000000; ++i)
		{
			results.emplace_back(
				pool.commit([i]
			{
				log_info(i);
				return i*i;
			})
			);
		}
		std::cout << "All " << results.size() << " tasks are committed to " << pool.thrCount() << " thread(s) done!" << std::endl;

		for (auto && result : results)log_info(result.get(), true);
		return 0;
	}
	catch (std::exception& e)
	{
		std::cout << "some unhappy happened...  " << std::this_thread::get_id() << e.what() << std::endl;
	}
}