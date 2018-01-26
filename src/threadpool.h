#pragma once
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <atomic>
#include <future>
//#include <condition_variable>
//#include <thread>
//#include <functional>
#include <stdexcept>

namespace std
{
	//max num of threads, should not to huge
#define  THREADPOOL_MAX_NUM 16
	//#define  THREADPOOL_AUTO_GROW

	//thread pool, submitted a task by a variable-parameters or lambda expression, then get the results
	//member in class is not supported, excepted static, global or Operator() decorative expression
	class threadpool
	{
		using Task = function<void()>;	//	type define
		vector<thread> _pool;     		//	thread pool
		queue<Task> _tasks;            	//	a task queue
		mutex _lock;                   	//	for synchronization
		condition_variable _task_cv;   	//	condition variable to block
		atomic<bool> _run{ true };     	//	thread pool is executable
		atomic<int>  _idlThrNum{ 0 };  	//	number of idle threads

	public:
		inline threadpool(unsigned short size = 4) { addThread(size); }
		inline ~threadpool()
		{
			_run = false;
			_task_cv.notify_all(); 		// wakeup all threads to work
			for (thread& thread : _pool)
			{
				//thread.detach(); // detach threads from master thread
				if (thread.joinable())
					thread.join(); // waiting for exit, before all task submitted before is done
			}
		}

	public:
		// submit a task
		// get() will be called to get the return value after the task is done.
		// two ways to call the member expression of class:
		// -bind	： .commit(std::bind(&Dog::sayHello, &dog));
		// -mem_fn	： .commit(std::mem_fn(&Dog::sayHello), this);
		// a decorative template for variable parameters
		template<class F, class... Args>
		auto commit(F&& f, Args&&... args) ->future<decltype(f(args...))>
		{
			if (!_run)    // stopped
				throw runtime_error("commit on ThreadPool is stopped.");

			using RetType = decltype(f(args...)); // typename std::result_of<F(Args...)>::type, prototype and return type of 'f' expression
			auto task = make_shared<packaged_task<RetType()>>(
				bind(forward<F>(f), forward<Args>(args)...)
				); // bind the entery and parameter for sepcific expression
			future<RetType> future = task->get_future();
			{
				// add to task queue
				lock_guard<mutex> lock{ _lock }; //lock_guard is necessary to protect the context
				_tasks.emplace([task]()  // push a task to the tail of task_queue
				{
					(*task)();
				});
			}
#ifdef THREADPOOL_AUTO_GROW
			if (_idlThrNum < 1 && _pool.size() < THREADPOOL_MAX_NUM)
				addThread(1);
#endif // !THREADPOOL_AUTO_GROW
			_task_cv.notify_one(); // wake one thread to work

			return future;
		}

		//the idle threads number
		int idlCount() { return _idlThrNum; }
		//all threads number
		int thrCount() { return _pool.size(); }
#ifndef THREADPOOL_AUTO_GROW
	private:
#endif // !THREADPOOL_AUTO_GROW
		//add specific number threads to backgrounds
		void addThread(unsigned short size)
		{
			for (; _pool.size() < THREADPOOL_MAX_NUM && size > 0; --size)
			{
				//create threads no more than THREADPOOL_MAX_NUM
				_pool.emplace_back([this]  //threads to work
				{
					while (_run)
					{
						Task task; //get a task from task queue
						{
							// unique_lock to protect context
							unique_lock<mutex> lock{ _lock };
							_task_cv.wait(lock, [this]
							{
								return !_run || !_tasks.empty();
							}); // waiting and blocked until a task is available
							if (!_run && _tasks.empty())
								return;
							task = move(_tasks.front()); // get task by FIFO
							_tasks.pop();
						}
						_idlThrNum--;
						task();//task to execute
						_idlThrNum++;
					}
				});
				_idlThrNum++;
			}
		}
	};

}

#endif
