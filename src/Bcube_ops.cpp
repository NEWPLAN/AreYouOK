#include "Bcube_ops.h"
#include <iostream>
#include <atomic>
#include <thread>
#include <string>
#include <assert.h>
#include <cstring>


extern int TYPE_SIZE[];

bcube_global_struct bcube_gs;
void bcube_do_steps(bcube_global_struct&);
void bg_loops(bcube_global_struct& bgs)
{

	bcube_init(bgs.bcube_s, bgs);
	bgs.unfinished_tensor.resize(4);
	bgs.is_inited_done = true;
	std::cout << "all init done, now we are going to send msg in bgthread..." << std::endl;
	while (true)
	{
		//std::cout << "run in bg_main_threads, now we are going to send some message" << std::endl;
		try
		{
			bcube_do_steps(bgs);
		}
		catch (exception& e)
		{
			std::cout << "error: " << e.what();
			while (1);
		}
		//std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}


/*
全局init，开启一个后台线程.
*/

void bcube_all_init_onice(bcube_global_struct& gs)
{
	static int print_count = 0;
	if (!bcube_gs.bgthread_start.test_and_set())
	{
		std::cout << "start the bgthread" << std::endl;
		bcube_gs.bg_thread.push_back(std::thread(bg_loops, std::ref(bcube_gs)));
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	while (!bcube_gs.is_inited_done)
	{
		if((print_count++)%5==0)
			std::cout << "bcube is not inited successfully, waiting for other connecting" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}


bool bcube_reduce(bcube_global_struct& bgs, tensor_table_entry& e, bool is_scatter)
{
	auto tensor_name = e.tensor_name;
	std::vector<received_tensor_entry> rcv_tensor;
	{
		std::lock_guard<std::mutex> rece_lock(bgs.bcube_mutex);
		auto& tensor_receive = bgs.receiv_tensor;
		auto find_tensor = tensor_receive.find(tensor_name);
		if (find_tensor == tensor_receive.end())return false;/*没有收到,下一次再取*/
		rcv_tensor = std::move(find_tensor->second);
		tensor_receive.erase(find_tensor);
	}
	assert(rcv_tensor.size()==4);
	//return true;
	if (e.tensor_ops == ALLREDUCE)
	{
		if (is_scatter)/*for scatter, sum these tensor*/
		{

			/*下面做一次加和操作*/
			for (auto it = rcv_tensor.begin(); it != rcv_tensor.end(); it++)
			{
				auto tensor_ptr = (int*)it->receive_ptr;
				auto tensor_counts = it->tensor_nums;
				auto start_position = it->start_position;
				auto e_tensor_ptr = e.tensor_data;
				auto type_size = TYPE_SIZE[e.tensor_type];
				auto block_size = e.block_size;
				auto add_pos = (int*)((char*)e_tensor_ptr + start_position*type_size*block_size);
				for (size_t addnum = 0; addnum < tensor_counts; addnum++)
				{
					//printf("%d,%d,%d\n", add_pos[addnum], tensor_ptr[addnum], add_pos[addnum]+ tensor_ptr[addnum]);
					add_pos[addnum] += tensor_ptr[addnum];

				}
				{/*release reources*/
					std::free(it->receive_ptr);
					//printf("in allreduce: free %p\n", it->receive_ptr);
					it->receive_ptr = nullptr;
				}
			}
			return true;
		}
		else/*gather, replace these tensor*/
		{
			/*下面做一次替换操作*/
			for (auto it = rcv_tensor.begin(); it != rcv_tensor.end(); it++)
			{
				auto tensor_ptr = (int*)it->receive_ptr;
				auto tensor_counts = it->tensor_nums;
				auto start_position = it->start_position;
				auto e_tensor_ptr = e.tensor_data;
				auto type_size = TYPE_SIZE[e.tensor_type];
				auto block_size = e.block_size;
				auto add_pos = (int*)((char*)e_tensor_ptr + start_position*type_size*block_size);
				for (size_t addnum = 0; addnum < tensor_counts; addnum++)
				{
					//printf("%d,%d,%d\n", add_pos[addnum], tensor_ptr[addnum], tensor_ptr[addnum]);
					add_pos[addnum] = tensor_ptr[addnum];
				}
				{/*release reources*/
					std::free(it->receive_ptr);
					//printf("in allreduce: free %p\n", it->receive_ptr);
					it->receive_ptr = nullptr;
				}
			}
			return true;
		}
	}
	else if(e.tensor_ops == ALLGATHER|| e.tensor_ops==BROADCAST)
	{
		for (auto it = rcv_tensor.begin(); it != rcv_tensor.end(); it++)
		{
			for (size_t index = 0; index < it->gather_ptr.size(); index++)
			{
				auto& _a_tensor = e.gather_tensor[it->start_position+index];
				/*first to do:
					release the old resource
					std::free(_a_tensor.tensor_ptr);
					_a_tensor.tensor_ptr=nullptr;
					,but now, we will release them last.
				*/
				std::free(_a_tensor.tensor_ptr);
				//printf("in allgather reduce: free %p\n", _a_tensor.tensor_ptr);
				_a_tensor.tensor_ptr = it->gather_ptr[index].tensor_ptr;
				_a_tensor.tensor_shape = it->gather_ptr[index].tensor_shape;
			}
		}
	}
	return true;
}
#include <algorithm>

void release_src(tensor_table_entry& e)
{
	std::vector<void*> free_ptr;
	free_ptr.push_back(e.tensor_data);
	free_ptr.push_back(nullptr);
	
	for (auto& it : e.gather_tensor)
	{
		if (find(free_ptr.begin(), free_ptr.end(), it.tensor_ptr) == free_ptr.end())
		{
			std::free(it.tensor_ptr);
			//printf("in release: free %p\n", it.tensor_ptr);
			free_ptr.push_back(it.tensor_ptr);
			it.tensor_ptr = nullptr;
		}
		else if(it.tensor_ptr != e.tensor_data)
		{
			perror("error in free data");
		}
	}
	std::free(e.tensor_data);
	//printf("in allreduce tensor data: free %p\n", e.tensor_data);
	e.tensor_data = nullptr;

	return;
}

static int reduce_loop = 0;
#define _DEBUG_TENSOR_GEN_
static void show_tensor(tensor_table_entry& e,int status=ALLREDUCE)
{
	int loops = 0;
	int node_id = 0;
	auto tensor_ptr = e.tensor_data;

#ifdef _DEBUG_TENSOR_GEN_
	if (e.tensor_ops == ALLREDUCE)
	{
		sscanf(e.tensor_name.c_str(), "_%d_allreduce_%d", &loops, &node_id);
		reduce_loop = loops;
		if ((loops % 100) || node_id != 100123)return;
	}
	else if (e.tensor_ops == ALLGATHER)
	{
		sscanf(e.tensor_name.c_str(), "_%d_allgather_%d", &loops, &node_id);
		reduce_loop = loops;
		if ((loops % 100) || node_id != 1003)return;
	}
	else if (e.tensor_ops == BROADCAST)
	{
		sscanf(e.tensor_name.c_str(), "_%d_broadcast_%d", &loops, &node_id);
		reduce_loop = loops;
		if ((loops % 100) || node_id != 1003)return;
	}
#endif // !_DEBUG_TENSOR_GEN_
	printf("%s ", e.tensor_name.c_str());
	if (e.tensor_ops == ALLREDUCE)
	{
		printf("[ ");
		for (int tensor_index = 0; tensor_index < e.tensor_size; tensor_index++)
			printf("%4d", ((int*)tensor_ptr)[tensor_index]);
		printf("]\n");
		if (0)
		{
			std::free(e.tensor_data);
			//printf("in allreduce: free %p\n", e.tensor_data);
			e.tensor_data = nullptr;
		}
	}
	else if (e.tensor_ops == ALLGATHER)
	{
		printf("\n[ ");
		for (int nums = 0; nums < 9; nums++)
		{
			auto& gather_tensor = e.gather_tensor[nums];
			auto& tensor_ptr = gather_tensor.tensor_ptr;
			printf("\n\t[");
			for (int tensor_index = 0; tensor_index < gather_tensor.tensor_shape; tensor_index++)
				printf("%4d", ((int*)tensor_ptr)[tensor_index]);
			printf("]\n");
		}
		printf("\n]\n");
	}
	else if (e.tensor_ops == BROADCAST)
	{
		printf("[ ");
		auto& broadcast_tensor = e.gather_tensor[0];
		auto& tensor_ptr = broadcast_tensor.tensor_ptr;
		for (int tensor_index = 0; tensor_index < broadcast_tensor.tensor_shape; tensor_index++)
			printf("%4d", ((int*)tensor_ptr)[tensor_index]);
		printf("]\n");
	}
	else
	{
		perror("not ready to handle");
		exit(-1);
	}
}
//void bcube_allreduce(char* src, char* dst, int block_size, int block_num, int block_type)
void bcube_do_steps(bcube_global_struct& bgs)
{
	auto& unfinished_vect = bgs.unfinished_tensor;
	auto unfin_size = (int)bgs.unfinished_tensor.size();
	//std::cout<<"unfinished tensor size: "<< unfin_size <<std::endl;
	{/*last stage*/
		std::vector<tensor_table_entry> tmp_tensor_table;
		auto& last_stage_tensor = unfinished_vect[unfin_size - 1];
		for (auto it = last_stage_tensor.begin();it != last_stage_tensor.end(); it++)
		{
			if (bcube_reduce(bgs, *it, false))
			{
				/*in here done all the ops and execute a callback*/
				//std::cout<<"this tensor "<<it->tensor_name<<" is done..."<<std::endl;
				show_tensor(*it);
				release_src(*it);
			}
			else 
			{
				tmp_tensor_table.push_back(std::move(*it));
			}			
		}
		last_stage_tensor = std::move(tmp_tensor_table);
	}
	for (int unfin_index = unfin_size - 2; unfin_index >= 0; unfin_index--)
	{/*从step3->step2->step1...step0*/
		std::vector<tensor_table_entry> tmp_tensor_table;
		auto& step_it = unfinished_vect[unfin_index];
		for (auto it = step_it.begin(); it != step_it.end(); it++)
		{
			bool is_reduce = bcube_reduce(bgs, *it, (unfin_index < (unfin_size / 2)) ? true : false);
			if (is_reduce)
			{
				/*copy to the next stage*/
				it->tensor_name += std::to_string(unfin_index + 1);
				bcube_send(*it, bgs.bcube_s, unfin_index + 1);
				//printf("...in the %d stage of %s...\n", unfin_index, it->tensor_name.c_str());
				unfinished_vect[unfin_index + 1].push_back(std::move(*it));
			}
			else
			{
				tmp_tensor_table.push_back(std::move(*it));
			}
		}
		step_it = std::move(tmp_tensor_table);
	}
	{/*from buf copy to unfinished.*/
		int count = 5;
		std::vector<tensor_table_entry> tmp_table;
		{
			std::lock_guard<std::mutex> gene_tensor_lock(bgs.bcube_mutex);
			auto & undo = bgs.tensor_table;
			while (!undo.empty() && count)
			{
				auto& it = undo.front();
				tmp_table.push_back(std::move(it));
				undo.pop();
			}
		}
		auto& unfin = bgs.unfinished_tensor;
		for (auto it = tmp_table.begin(); it != tmp_table.end(); it++)
		{
			it->tensor_name += "0";/*add exchange name*/
			if (it->tensor_ops == ALLREDUCE)
			{
				//printf("in allreduce\n");
				/*send out*/
				bcube_send((*it), bgs.bcube_s, 0);
				/*move to unfinished vector*/
				unfin[0].push_back(std::move(*it));
			}
			else if(it->tensor_ops == BROADCAST|| it->tensor_ops == ALLGATHER)
			{/*enter a gather stage directly*/
				//printf("in allgather or broadcast, enter stage %d\n", unfin_size / 2);
				bcube_send((*it), bgs.bcube_s, unfin_size / 2);
				unfin[unfin_size/2].push_back(std::move(*it));
			}
			else
			{
				std::cerr << "error in tensor..." << std::endl;
				exit(-1);
			}
		}
	}
}

#include <random>

void allreduce_enque(bcube_global_struct& bgs, int unique_id,int loops)
{
	std::random_device rd;
	int init_num = rd() % 83;
	int* a_ = new int[18]();
	for (int nums = 0; nums < 18; nums++)a_[nums] = init_num++;
	tensor_table_entry e;
	e.block_size = 1;
	e.tensor_name = "_"+std::to_string(loops)+"_allreduce_" + std::to_string(unique_id);
	e.available_nums = 18;
	e.tensor_size = 18;
	e.tensor_data = (void*)a_;
	e.tensor_type = T_INT32;
	e.tensor_ops = ALLREDUCE;
#ifdef _DEBUG_TENSOR_GEN_show_
	{
		std::lock_guard<std::mutex> print_tensor(bgs.bcube_mutex);
		printf("creates %s : [", e.tensor_name.c_str());
		for (int i = 0; i < 18; i++)printf("%5d", a_[i]);
		printf(" ]\n");
	}
#endif
	{
		std::lock_guard<std::mutex> enque_lock(bgs.bcube_mutex);
		auto& tensor_table = bgs.tensor_table;
		tensor_table.push(std::move(e));
	}
	while (bgs.tensor_table.size() > 100)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	return;
}

void allgather_enqueue(bcube_global_struct& bgs, int unique_id, int loops)
{
	std::random_device rd;
	int init_num = rd() % 83;
	int* a_ = new int[18]();
	for (int nums = 0; nums < 18; nums++)a_[nums] = init_num++;
	tensor_table_entry e;
	e.block_size = 1;
	e.tensor_name = "_" + std::to_string(loops) + "_allgather_" + std::to_string(unique_id);
	e.available_nums = 18;
	e.tensor_size = 18;
	e.tensor_data = (void*)a_;
	e.gather_tensor.resize(e.tensor_size);
	{
		for (auto& it : e.gather_tensor)
		{
			it.tensor_shape = e.tensor_size;
			it.tensor_ptr = (void*)std::malloc(18*sizeof(int));
			//printf("in allgather_enqueue: malloc %p\n", it.tensor_ptr);
			std::memcpy(it.tensor_ptr, (void*)a_,18*sizeof(int));
		}
	}
	e.tensor_type = T_INT32;
	e.tensor_ops = ALLGATHER;
	if (0)
	{
		std::lock_guard<std::mutex> print_tensor(bgs.bcube_mutex);
		printf("creates %s : [", e.tensor_name.c_str());
		for (int i = 0; i < 18; i++)printf("%5d", a_[i]);
		printf(" ]\n");
	}
	//e.tensor_ops = ALLREDUCE;
	//show_tensor(e,ALLGATHER);
	//e.tensor_ops = ALLGATHER;
	{
		std::lock_guard<std::mutex> enque_lock(bgs.bcube_mutex);
		auto& tensor_table = bgs.tensor_table;
		tensor_table.push(std::move(e));
	}
	while (bgs.tensor_table.size() > 100)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	return;
}

void broadcast_enqueue(bcube_global_struct& bgs, int unique_id, int loops)
{
	std::random_device rd;
	int init_num = rd() % 83;
	int* a_ = new int[18]();
	for (int nums = 0; nums < 18; nums++)a_[nums] = init_num++;
	tensor_table_entry e;
	e.block_size = 1;
	e.tensor_name = "_" + std::to_string(loops) + "_broadcast_" + std::to_string(unique_id);
	e.available_nums = 18;
	e.tensor_size = 18;
	e.gather_tensor.resize(e.tensor_size);
	{
		for (auto& it : e.gather_tensor)
		{
			it.tensor_shape = e.tensor_size;
			it.tensor_ptr = (void*)std::malloc(18 * sizeof(int));
			//printf("in broadcast_enqueue: malloc %p\n", it.tensor_ptr);
			std::memcpy(it.tensor_ptr, (void*)a_, 18 * sizeof(int));
		}
	}
	e.tensor_data = (void*)a_;
	e.tensor_type = T_INT32;
	e.tensor_ops = BROADCAST;
	if (0)
	{
		std::lock_guard<std::mutex> print_tensor(bgs.bcube_mutex);
		printf("creates %s : [", e.tensor_name.c_str());
		for (int i = 0; i < 18; i++)printf("%5d", a_[i]);
		printf(" ]\n");
	}
	{
		std::lock_guard<std::mutex> enque_lock(bgs.bcube_mutex);
		auto& tensor_table = bgs.tensor_table;
		tensor_table.push(std::move(e));
	}
	while (bgs.tensor_table.size() > 100)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	return;
}

void allreduce_test(bcube_global_struct& bcube_gs)
{
	int loops = 0;
	while (1)
	{
		std::vector<std::thread> thread_handle;
		int threadnum = 60;
		while (threadnum--)
		{
			thread_handle.push_back(std::thread(allreduce_enque, std::ref(bcube_gs), threadnum, loops));
		}
		for (auto& thread_id : thread_handle)
			thread_id.join();
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		loops++;
	}
}
void allgather_test(bcube_global_struct& bcube_gs)
{
	int loops = 0;
	while (1)
	{
		std::vector<std::thread> thread_handle;
		int threadnum = 12;
		while (threadnum--)
		{
			thread_handle.push_back(std::thread(allgather_enqueue, std::ref(bcube_gs), threadnum, loops));
		}
		for (auto& thread_id : thread_handle)
			thread_id.join();
		//std::cout << "create tensors done" << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		int cycle = 0;
		int receive_size = 0;
		while (loops>(reduce_loop+20))
		{
			cycle++;
			int rrr = (int)bcube_gs.receiv_tmp_tensor.size();
			if (receive_size != rrr)
			{
				receive_size = rrr;
				cycle = 0;
			}
			//printf("in %d loops, receiv_tmp_tensor.size() = %d\n", loops, rrr);
			std::this_thread::sleep_for(std::chrono::seconds(1));
			if ((cycle > 10) && rrr == receive_size)
				exit(-1);
		}
		//std::this_thread::sleep_for(std::chrono::seconds(10));
		loops++;
	}
}
void broadcast_test(bcube_global_struct& bcube_gs)
{
	int loops = 0;
	while (1)
	{
		std::vector<std::thread> thread_handle;
		int threadnum = 12;
		while (threadnum--)
		{
			thread_handle.push_back(std::thread(broadcast_enqueue, std::ref(bcube_gs), threadnum, loops));
		}
		for (auto& thread_id : thread_handle)
			thread_id.join();
		//std::cout << "create tensors done" << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		//std::this_thread::sleep_for(std::chrono::seconds(10));
		loops++;
	}
}


void bcube_ops_test(void)
{
	bcube_all_init_onice(bcube_gs);
	
	allreduce_test(bcube_gs);
	//allgather_test(bcube_gs);
	broadcast_test(bcube_gs);
	
	return;
}