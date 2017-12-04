#ifndef __BCUBE__MESSAGE__
#define __BCUBE__MESSAGE__

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <unordered_map>
#include <tuple>

#ifdef __tensorflow__
typedef std::function<Status&> Status_callback;
#endif // __tensorflow__

typedef enum
{
    T_INIT8 = 0, T_INT32 = 1, T_INIT64 = 2, T_FLOAT32 = 3, T_FLOAT64 = 4
} TENSOR_TYPE;



typedef enum
{
	OPS_ERROR=0,ALLREDUCE=1, ALLGATHER=2,BROADCAST=3, OPS_HELLO=4
} TENSOR_OPS;

typedef struct
{
	int rank;/*是哪一个rank发送的*/
	int start_pos;/*在原始tensor的起始位置*/
	int nums;/*tensor 元素个数*/
	int msg_length;
    int name_len;/*tensor name length*/
	TENSOR_TYPE tensor_type;/*tensor type*/
    TENSOR_OPS t_ops;/*当前tensor需要进行的操作*/
    char data[1];/*data store tensor_name and tensor_data,offset by name_length and tensor_len*/
} msg_struct;

/*
存放具体tensor的表项
*/
typedef int Tensor_Shape;

typedef struct
{
	Tensor_Shape tensor_shape;
	void* tensor_ptr;
}Tensor_Info;
typedef struct
{
    std::vector<bool> step;/*use less*/
    std::vector<int> block_in_step;/*use less*/


    std::string tensor_name;/*store tensor name*/
    TENSOR_TYPE tensor_type;/*current tensor type*/
	std::size_t available_nums;/*available nums, for alignment in reduce.*/
	std::size_t block_size;/*element number in each block*/

    void* tensor_data = nullptr; /*store tensor data*/
	int tensor_size;
#ifdef __tensorflow__
    OpKernelContext* context;/*context for tensor*/
    Tensor tensor;/*Input tensor.*/
    Tensor* output;/* Pre-allocated output tensor. */
    Status_callback callback;
#endif
	/*index as order=rand,tensorshape describe a tensor like size, void* for tensor ptr.*/
	std::vector<Tensor_Info> gather_tensor;
	TENSOR_OPS tensor_ops;/*tensor operation like allreduce,allgather,broadcast...*/
} tensor_table_entry;

typedef struct
{
	std::string tensor_name;/*tensor name*/
	std::size_t tensor_nums;/*number of element*/
	std::size_t start_position;/*start position*/
	TENSOR_TYPE tensor_type;/*tensor type, int, char or other*/
	TENSOR_OPS tensor_ops;/*tensor ops allreduce,allgather,broadcast*/
	char* receive_ptr =nullptr;/*point to the memory access for tensor data*/
	std::vector<Tensor_Info> gather_ptr;
}received_tensor_entry;

typedef std::unordered_map<std::string, tensor_table_entry> Tensor_table;

class tensor_msg
{
public:
    tensor_msg() {}; /*nothing to do*/
    ~tensor_msg() {};

static void encode(tensor_table_entry&, void**, int, int, int*);/*encode to send*/
static void decode(received_tensor_entry&, void* );/*decode msg to tensor entery*/
};
received_tensor_entry msg_deocde(void* msg);
#endif // __BCUBE__MESSAGE__
