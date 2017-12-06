/************************************below is belong to tensorflow ************************/
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2017 NEWPLAN Tsinghua University.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

/*
* Allreduce, Allgather and Broadcast Ops for TensorFlow.
*
* TensorFlow natively provides inter-device communication through send and
* receive ops and inter-node communication through Distributed TensorFlow,
* based on the same send and receive abstractions. These end up being
* insufficient for synchronous data-parallel training on HPC clusters where
* Infiniband or other high-speed interconnects are available.  This module
* implements BCUBE ops for allgather, allreduce and broadcast, which do
* optimized gathers, reductions and broadcasts and can take advantage of topology
* of bcube.
*
* The primary logic of the allreduce, allgather and broadcast are in TCP and
* RDMA implementations. The background thread which facilitates BCUBE operations
* is run in BackgroundThreadLoop(). The provided ops are:
*      - BcubeAllreduce:
*          Perform an allreduce on a Tensor, returning the sum
*          across all BCUBE nodes in the global topology.
*      - BcubeAllgather:
*          Perform an allgather on a Tensor, returning the concatenation of
*          the tensor on the first dimension across allBCUBE nodes in the
*          global Topology.
*      - BcubeBroadcast:
*          Perform a broadcast on a Tensor, broadcasting Tensor
*          value from root rank(RANK_0) to all other ranks.
*
* Additionally, this library provides C APIs to initialize Bcube and query
* rank, and world size.  These are used in Python directly through ctypes.
*/
#include "bcube_ops.h"
#include <iostream>
#include <atomic>
#include <thread>
#include <string>
#include <assert.h>
#include <cstring>

using namespace tensorflow;

namespace bcube
{
	namespace tensorflow
	{
		namespace
		{
			static  int TYPE_SIZE[] =
			{
				4,				sizeof(bool),
				sizeof(uint8_t), sizeof(int8_t),
				sizeof(uint16_t),sizeof(int16_t),
				sizeof(uint32_t),sizeof(int32_t),
				sizeof(uint64_t),sizeof(int64_t),
				sizeof(float_t),sizeof(double_t)
			};

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
					bcube_do_steps(bgs);
				}
			}


			/*
			global init, launch a background thread.
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
					if ((print_count++) % 5 == 0)
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
					if (find_tensor == tensor_receive.end())return false;/*û���յ�,��һ����ȡ*/
					rcv_tensor = std::move(find_tensor->second);
					tensor_receive.erase(find_tensor);
				}
				assert(rcv_tensor.size() == 4);
				//return true;
				if (e.tensor_ops == ALLREDUCE)
				{
					if (is_scatter)/*for scatter, sum these tensor*/
					{

						/*������һ�μӺͲ���*/
						for (auto it = rcv_tensor.begin(); it != rcv_tensor.end(); it++)
						{
							auto tensor_ptr = (int*)it->receive_ptr;
							auto tensor_counts = it->tensor_nums;
							auto start_position = it->start_position;
							auto e_tensor_ptr = e.tensor_data;
							auto type_size = TYPE_SIZE[e.tensor_type];
							auto block_size = e.block_size;
							auto add_pos = (int*)((char*)e_tensor_ptr + start_position * type_size * block_size);
							for (size_t addnum = 0; addnum < tensor_counts; addnum++)
							{
								//printf("%d,%d,%d\n", add_pos[addnum], tensor_ptr[addnum], add_pos[addnum]+ tensor_ptr[addnum]);
								add_pos[addnum] += tensor_ptr[addnum];

							}
							{
								/*release reources*/
								std::free(it->receive_ptr);
								//printf("in allreduce: free %p\n", it->receive_ptr);
								it->receive_ptr = nullptr;
							}
						}
						return true;
					}
					else/*gather, replace these tensor*/
					{
						/*������һ���滻����*/
						for (auto it = rcv_tensor.begin(); it != rcv_tensor.end(); it++)
						{
							auto tensor_ptr = (int*)it->receive_ptr;
							auto tensor_counts = it->tensor_nums;
							auto start_position = it->start_position;
							auto e_tensor_ptr = e.tensor_data;
							auto type_size = TYPE_SIZE[e.tensor_type];
							auto block_size = e.block_size;
							auto add_pos = (int*)((char*)e_tensor_ptr + start_position * type_size * block_size);
							for (size_t addnum = 0; addnum < tensor_counts; addnum++)
							{
								//printf("%d,%d,%d\n", add_pos[addnum], tensor_ptr[addnum], tensor_ptr[addnum]);
								add_pos[addnum] = tensor_ptr[addnum];
							}
							{
								/*release reources*/
								std::free(it->receive_ptr);
								//printf("in allreduce: free %p\n", it->receive_ptr);
								it->receive_ptr = nullptr;
							}
						}
						return true;
					}
				}
				else if (e.tensor_ops == ALLGATHER || e.tensor_ops == BROADCAST)
				{
					for (auto it = rcv_tensor.begin(); it != rcv_tensor.end(); it++)
					{
						for (size_t index = 0; index < it->gather_ptr.size(); index++)
						{
							auto& _a_tensor = e.gather_tensor[it->start_position + index];
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
					else if (it.tensor_ptr != e.tensor_data)
					{
						perror("error in free data");
					}
				}
				std::free(e.tensor_data);
				//printf("in allreduce tensor data: free %p\n", e.tensor_data);
				e.tensor_data = nullptr;

				return;
			}
			static void show_tensor(tensor_table_entry& e, int status = ALLREDUCE)
			{
				printf("%s ", e.tensor_name.c_str());
			}
			//void bcube_allreduce(char* src, char* dst, int block_size, int block_num, int block_type)
			void bcube_do_steps(bcube_global_struct& bgs)
			{
				auto& unfinished_vect = bgs.unfinished_tensor;
				auto unfin_size = (int)bgs.unfinished_tensor.size();
				{
					/*last stage*/
					std::vector<tensor_table_entry> tmp_tensor_table;
					auto& last_stage_tensor = unfinished_vect[unfin_size - 1];
					for (auto it = last_stage_tensor.begin(); it != last_stage_tensor.end(); it++)
					{
						if (bcube_reduce(bgs, *it, false))
						{
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
				{
					/*from step3->step2->step1...step0*/
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
							unfinished_vect[unfin_index + 1].push_back(std::move(*it));
						}
						else
						{
							tmp_tensor_table.push_back(std::move(*it));
						}
					}
					step_it = std::move(tmp_tensor_table);
				}
				{
					/*from buf copy to unfinished.*/
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
						else if (it->tensor_ops == BROADCAST || it->tensor_ops == ALLGATHER)
						{
							/*enter a gather stage directly*/
							//printf("in allgather or broadcast, enter stage %d\n", unfin_size / 2);
							bcube_send((*it), bgs.bcube_s, unfin_size / 2);
							unfin[unfin_size / 2].push_back(std::move(*it));
						}
						else
						{
							std::cerr << "error in tensor..." << std::endl;
							exit(-1);
						}
					}
				}
			}
			// Convert a TensorFlow DataType to our MPIDataType.
			Status DataTypeToBcubeType(DataType tf_dtype, BCUBE_TYPE* bcube_dtype)
			{
				switch (tf_dtype)
				{
				case DT_UINT8:
					*bcube_dtype = T_UINIT8;
					return Status::OK();
				case DT_INT8:
					*bcube_dtype = T_INIT8;
					return Status::OK();
				case DT_UINT16:
					*bcube_dtype = T_UINT16;
					return Status::OK();
				case DT_INT16:
					*bcube_dtype = T_INT16;
					return Status::OK();
				case DT_INT32:
					*bcube_dtype = T_INT32;
					return Status::OK();
				case DT_INT64:
					*bcube_dtype = T_INT64;
					return Status::OK();
				case DT_FLOAT:
					*bcube_dtype = T_FLOAT32;
					return Status::OK();
				case DT_DOUBLE:
					*bcube_dtype = T_FLOAT64;
					return Status::OK();
				case DT_BOOL:
					*bcube_dtype = T_BOOL;
					return Status::OK();
				default:
					return errors::Internal("Invalid tensor type.");
				}
			}

			// bcube must be initialized and the background thread must be running before this function is called.
			void bcube_allreduce_queue(OpKernelContext* context, const Tensor& tensor,
				Tensor* output, GPU_EVENT_IF_CUDA ready_event,
				const std::string name, const int device,
				StatusCallback callback)
			{
				BCUBE_TYPE dtype;
				Status status = DataTypeToBcubeType(tensor.dtype(), &dtype);
				if (!status.ok())
				{
					callback(status);
					return;
				}

				std::vector<int64_t> _tensor_shape;
				for (int i = 0; i < tensor.shape().dims(); i++)
				{
					_tensor_shape.push_back(tensor.shape().dim_size(i));
				}

				tensor_table_entry e;

				e.tensor_name = name;
				e.context = context;
				e.tensor = tensor;
				e.output = output;
				e.ready_event = ready_event;
				e.device = device;
				e.callback = callback;
				e.tensor_shape = std::move(_tensor_shape);

				{
					e.available_nums = tensor.NumElements();
					e.block_size = (e.available_nums + 17) / 18;
					e.tensor_size = 18 * e.block_size;
					auto _type_size = TYPE_SIZE[dtype];
					e.tensor_data = std::malloc(e.tensor_size * _type_size);
					assert(e.tensor_data != nullptr);
					std::memcpy(e.tensor_data, (const void*)tensor.tensor_data().data(),
						_type_size*e.available_nums);
				}
				e.tensor_type = dtype;
				e.tensor_ops = ALLREDUCE;
				{
					std::lock_guard<std::mutex> enque_lock(bcube_gs.bcube_mutex);
					auto& tensor_table = bcube_gs.tensor_table;
					tensor_table.push(std::move(e));
				}
				return;
			}

			// bcube must be initialized and the background thread must be running before this function is called.
			void bcube_allgather_queue(OpKernelContext* context, const Tensor& tensor,
				GPU_EVENT_IF_CUDA ready_event,
				const std::string name, const int device,
				StatusCallback callback)
			{

				BCUBE_TYPE dtype;
				Status status = DataTypeToBcubeType(tensor.dtype(), &dtype);
				if (!status.ok())
				{
					callback(status);
					return;
				}

				std::vector<int64_t> _tensor_shape;
				for (int i = 0; i < tensor.shape().dims(); i++)
				{
					_tensor_shape.push_back(tensor.shape().dim_size(i));
				}

				tensor_table_entry e;

				e.tensor_name = name;
				e.context = context;
				e.tensor = tensor;
				e.ready_event = ready_event;
				e.device = device;
				e.callback = callback;
				e.tensor_shape = std::move(_tensor_shape);

				{
					e.block_size = 1;
					e.available_nums = 18;
					e.tensor_size = 18;
					int element_nums = tensor.NumElements();
					e.tensor_data = nullptr;
					auto _type_size = TYPE_SIZE[dtype];

					e.gather_tensor.resize(e.available_nums);
					{
						for (auto& it : e.gather_tensor)
						{
							it.tensor_shape = element_nums;
							it.tensor_ptr = (void*)std::malloc(it.tensor_shape * _type_size);
							assert(it.tensor_ptr != nullptr);
							std::memcpy(it.tensor_ptr, (const void*)tensor.tensor_data().data(),
								it.tensor_shape * _type_size);
						}
					}
				}
				e.tensor_type = dtype;
				e.tensor_ops = ALLGATHER;
				{
					std::lock_guard<std::mutex> enque_lock(bcube_gs.bcube_mutex);
					auto& tensor_table = bcube_gs.tensor_table;
					tensor_table.push(std::move(e));
				}
				return;
			}

			// bcube must be initialized and the background thread must be running before this function is called.
			void bcube_broadcast_queue(OpKernelContext* context, const Tensor& tensor,
				Tensor* output, int root_rank,
				GPU_EVENT_IF_CUDA ready_event,
				const std::string name, const int device,
				StatusCallback callback)
			{
				BCUBE_TYPE dtype;
				Status status = DataTypeToBcubeType(tensor.dtype(), &dtype);
				if (!status.ok())
				{
					callback(status);
					return;
				}

				std::vector<int64_t> _tensor_shape;
				for (int i = 0; i < tensor.shape().dims(); i++)
				{
					_tensor_shape.push_back(tensor.shape().dim_size(i));
				}

				tensor_table_entry e;

				e.tensor_name = name;
				e.context = context;
				e.tensor = tensor;
				e.ready_event = ready_event;
				e.device = device;
				e.callback = callback;
				e.tensor_shape = std::move(_tensor_shape);

				{
					e.block_size = 1;
					e.available_nums = 18;
					e.tensor_size = 18;
					int element_nums = tensor.NumElements();
					e.tensor_data = nullptr;
					auto _type_size = TYPE_SIZE[dtype];

					e.gather_tensor.resize(e.available_nums);
					{
						for (auto& it : e.gather_tensor)
						{
							it.tensor_shape = element_nums;
							it.tensor_ptr = (void*)std::malloc(it.tensor_shape * _type_size);
							assert(it.tensor_ptr != nullptr);
							std::memcpy(it.tensor_ptr, (const void*)tensor.tensor_data().data(),
								it.tensor_shape * _type_size);
						}
					}
				}
				e.tensor_type = dtype;
				e.tensor_ops = BROADCAST;
				{
					std::lock_guard<std::mutex> enque_lock(bcube_gs.bcube_mutex);
					auto& tensor_table = bcube_gs.tensor_table;
					tensor_table.push(std::move(e));
				}
				return;
			}

			// Check that Bcube is initialized.
			Status CheckInitialized()
			{
				if (!bcube_gs.is_inited_done)
				{
					return errors::FailedPrecondition(
						"Bcube has not been initialized; use bcube.tensorflow.init().");
				}
				return Status::OK();
			}

			// C interface to initialize Bcube.
			extern "C" void bcube_tensorflow_init()
			{
				auto &bgs = bcube_gs;
				bcube_all_init_onice(bgs);
			}

			// C interface to get index of current Bcube process.
			// Returns -1 if Bcube is not initialized.
			extern "C" int bcube_tensorflow_rank()
			{
				if (!bcube_gs.is_inited_done)
				{
					return -1;
				}
				return bcube_gs.bcube_s.rank;
			}

			// C interface to get index of current Bcube process in the node it is on..
			// Returns -1 if Bcube is not initialized.
			extern "C" int bcube_tensorflow_local_rank()
			{
				return 0;
			}

			// C interface to return number of Bcube processes.
			// Returns -1 if Bcube is not initialized.
			extern "C" int bcube_tensorflow_size()
			{
				if (!bcube_gs.is_inited_done)
				{
					return -1;
				}
				return bcube_gs.bcube_s.bcube_node_count;
			}
			int GetDeviceID(OpKernelContext* context)
			{
				int device = CPU_DEVICE_ID;
				if (context->device() != nullptr &&
					context->device()->tensorflow_gpu_device_info() != nullptr)
				{
					device = context->device()->tensorflow_gpu_device_info()->gpu_id;
				}
				return device;
			}

			// On GPU this event will signal that data is ready, and tensors are
			// allocated.
			GPU_EVENT_IF_CUDA RecordReadyEvent(OpKernelContext* context)
			{
#if HAVE_CUDA
				auto device_context = context->op_device_context();
				if (device_context != nullptr)
				{
					auto executor = device_context->stream()->parent();
					GPU_EVENT_IF_CUDA ready_event = new perftools::gputools::Event(executor);
					ready_event->Init();
					device_context->stream()->ThenRecordEvent(ready_event);
					return ready_event;
				}
#endif
				return nullptr;
			}
		}//namespace tensorflow

			class BcubeAllreduceOp : public AsyncOpKernel
			{
			public:
				explicit BcubeAllreduceOp(OpKernelConstruction* context)
					: AsyncOpKernel(context) {}

				void ComputeAsync(OpKernelContext* context, DoneCallback done) override
				{
					OP_REQUIRES_OK(context, CheckInitialized());

					auto node_name = name();
					auto device = GetDeviceID(context);
					auto tensor = context->input(0);
					Tensor* output;
					OP_REQUIRES_OK(context,
						context->allocate_output(0, tensor.shape(), &output));
					GPU_EVENT_IF_CUDA ready_event = RecordReadyEvent(context);
					bcube_allreduce_queue(context, tensor, output, ready_event, node_name,
						device, [context, done](const Status & status)
					{
						context->SetStatus(status);
						done();
					});
				}
			};

			REGISTER_KERNEL_BUILDER(Name("BcubeAllreduce").Device(DEVICE_CPU),
				BcubeAllreduceOp);
#if BCUBE_GPU_ALLREDUCE
			REGISTER_KERNEL_BUILDER(Name("BcubeAllreduce").Device(DEVICE_GPU),
				BcubeAllreduceOp);
#endif

			REGISTER_OP("BcubeAllreduce")
				.Attr("T: {int32, int64, float32, float64}")
				.Input("tensor: T")
				.Output("sum: T")
				.SetShapeFn([](shape_inference::InferenceContext* c)
							{
								c->set_output(0, c->input(0));
								return Status::OK();
							})
				.Doc(R"doc(
					Perform an Bcube Allreduce on a tensor. All other nodes that do a reduction
					on a tensor with the same name must have the same dimension for that tensor.
					Tensors are reduced with other tensors that have the same node name for the
					allreduce.

					Arguments
						tensor:     A tensor to reduce.

					Output
						sum:    A tensor with the same shape as `tensor`, summed across all Bcube nodes.
					)doc");

			class BcubeAllgatherOp : public AsyncOpKernel 
			{
			public:
				explicit BcubeAllgatherOp(OpKernelConstruction* context): AsyncOpKernel(context) {}

				void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
					OP_REQUIRES_OK(context, CheckInitialized());

					auto node_name = name();
					auto device = GetDeviceID(context);
					auto tensor = context->input(0);
					// We cannot pre-allocate output for allgather, since shape of result
					// is only known after all nodes make a notice.
					GPU_EVENT_IF_CUDA ready_event = RecordReadyEvent(context);
					bcube_allgather_queue(context, tensor, ready_event, node_name, device,
						[context, done](const Status& status) 
						{
							context->SetStatus(status);
							done();
						});
				}
			}; 

			REGISTER_KERNEL_BUILDER(Name("BcubeAllgather").Device(DEVICE_CPU),
				BcubeAllgatherOp);
#if BCUBE_GPU_ALLGATHER
			REGISTER_KERNEL_BUILDER(Name("BcubeAllgather").Device(DEVICE_GPU),
				BcubeAllgatherOp);
#endif

			REGISTER_OP("BcubeAllgather")
				.Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64, bool}")
				.Input("tensor: T")
				.Output("output: T")
				.SetShapeFn([](shape_inference::InferenceContext* c) 
							{
								shape_inference::ShapeHandle output;
								TF_RETURN_IF_ERROR(
									c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
								c->set_output(0, output);
								return Status::OK();
							})
				.Doc(R"doc(
					Perform an Bcube Allgather on a tensor. All other processes that do a gather on a
					tensor with the same name must have the same rank for that tensor, and have the
					same dimension on all but the first dimension.

					Arguments
						tensor:     A tensor to gather.

					Output
						gathered:    A tensor with the same shape as `tensor` except for the first dimension.
					)doc");

			class BcubeBroadcastOp : public AsyncOpKernel 
			{
			public:
				explicit BcubeBroadcastOp(OpKernelConstruction* context): AsyncOpKernel(context) 
				{
					OP_REQUIRES_OK(context, context->GetAttr("root_rank", &root_rank_));
				}

				void ComputeAsync(OpKernelContext* context, DoneCallback done) override 
				{
					OP_REQUIRES_OK(context, CheckInitialized());

					auto node_name = name();
					auto device = GetDeviceID(context);
					auto tensor = context->input(0);
					Tensor* output = nullptr;
					if (bcube_gs.bcube_s.rank == root_rank_) 
					{
						context->set_output(0, tensor);
					}
					else 
					{
						OP_REQUIRES_OK(context, context->allocate_output(0, tensor.shape(), &output));
					}
					GPU_EVENT_IF_CUDA ready_event = RecordReadyEvent(context);
					bcube_broadcast_queue(context, tensor, output, root_rank_, ready_event, node_name, device,
						[context, done](const Status& status) 
						{
							context->SetStatus(status);
							done();
						});
				}

			private:
				int root_rank_;
			};

			REGISTER_KERNEL_BUILDER(Name("BcubeBroadcast").Device(DEVICE_CPU),
				BcubeBroadcastOp);
#if BCUBE_GPU_BROADCAST
			REGISTER_KERNEL_BUILDER(Name("BcubeBroadcast").Device(DEVICE_GPU),
				BcubeBroadcastOp);
#endif

			REGISTER_OP("BcubeBroadcast")
				.Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64, bool}")
				.Attr("root_rank: int")
				.Input("tensor: T")
				.Output("output: T")
				.SetShapeFn([](shape_inference::InferenceContext* c) {
				c->set_output(0, c->input(0));
				return Status::OK();
			})
				.Doc(R"doc(
					Perform an Bcube Broadcast on a tensor. All other processes that do a broadcast
					on a tensor with the same name must have the same dimension for that tensor.

					Arguments
						tensor:     A tensor to broadcast.
						root_rank:  Rank that will send data, other ranks will receive data.

					Output
						output:    A tensor with the same shape as `tensor` and same value as
								   `tensor` on root rank.
					)doc");
	}//namespace tensorflow
}//namespce bcube