/*==============================================================================
# Copyright 2017 NEWPLAN, Tsinghua University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
*/

#ifndef __TENSOTFLOW_BCUBE_OPS__
#define __TENSOTFLOW_BCUBE_OPS__
#include "bcube_utils.h"
#include <vector>
#include <iostream>
#include <atomic>
#include <thread>
#include <string>
#include <mutex>
#include "bcube_comm.h"
#include "bcube_message.h"
#include <unordered_map>
#include <queue>
#include <chrono>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define EIGEN_USE_THREADS

#if HAVE_CUDA
#include "tensorflow/stream_executor/stream.h"
#include <cuda_runtime.h>
#endif

//typedef char* tensor;
/*
*include all tensor infos
*/

#define CPU_DEVICE_ID -1

typedef std::unordered_map<std::string, std::vector<received_tensor_entry>> Received_tensor;

struct bcube_global_struct
{
	std::atomic_flag bgthread_start = ATOMIC_FLAG_INIT;
	std::vector<std::thread> bg_thread;/*two background thread,main thread is for data sending, and a receive loops thread to reaceive data*/
	std::vector<step> all_send_strategy;/*send strategy*/

	bool is_inited_done = false; /*flag indicating inited done*/

	std::mutex bcube_mutex;/*multi thread for mutex*/
	std::mutex tensor_gene_mutex;/*muthex between tensor gen and fetch*/
	std::mutex tensor_recv_mutex;/*mutex between tensor receive and reduce*/

	std::queue<tensor_table_entry> tensor_table;/*add new tensor into table*/
	std::vector<std::vector<tensor_table_entry>> unfinished_tensor;/*2D for different stage tensor.*/
	Received_tensor receiv_tensor;/*this is storing ready to reduce tensor*/
	Received_tensor receiv_tmp_tensor;/*first insert into tmp buf, if collected all other, copy to receiv_msg.*/

	int rank;/*my rank*/
	bcube_struct bcube_s;/*bcube structs*/


	bool shut_down = false;
	// The CUDA stream used for data transfers and within-allreduce operations.
	// A naive implementation would use the TensorFlow StreamExecutor CUDA
	// stream. However, the allreduce and allgather require doing memory copies
	// and kernel executions (for accumulation of values on the GPU). However,
	// the subsequent operations must wait for those operations to complete,
	// otherwise MPI (which uses its own stream internally) will begin the data
	// transfers before the CUDA calls are complete. In order to wait for those
	// CUDA operations, if we were using the TensorFlow stream, we would have to
	// synchronize that stream; however, other TensorFlow threads may be
	// submitting more work to that stream, so synchronizing on it can cause the
	// allreduce to be delayed, waiting for compute totally unrelated to it in
	// other parts of the graph. Overlaying memory transfers and compute during
	// backpropagation is crucial for good performance, so we cannot use the
	// TensorFlow stream, and must use our own stream.
#if HAVE_CUDA
	std::unordered_map<int, cudaStream_t> streams;
#endif


	~bcube_global_struct()
	{
		std::this_thread::sleep_for(std::chrono::seconds(10));
		for (auto& thread_id : bg_thread)
		{
			if (thread_id.joinable())
			{
				shut_down = true;
				thread_id.join();
			}
		}
		std::cout << "end in global states" << std::endl;
	}
};

#endif
