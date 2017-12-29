#ifndef __TENSORFLOW_BCUBE_RDMA_H__
#define __TENSORFLOW_BCUBE_RDMA_H__


#if HAVE_RDMA

#include <vector>
#include <string>
#include <rdma/rdma_cma.h>
#include <thread>
#include <iostream>

#include "bcube_message.h"
#include "bcube_comm.h"
#include "bcube_utils.h"
#include "bcube_ops.h"
#include "bcube_message.h"


void rc_die(const char *reason);

const size_t BUFFER_SIZE = 512 * 1024 * 1024 + 1;
#define TIMEOUT_IN_MS 500
#define TEST_NZ(x) do { if ( (x)) rc_die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) rc_die("error: " #x " failed (returned zero/null)."); } while (0)
#define MIN_CQE 10

enum message_id
{
	MSG_INVALID = 0,
	MSG_MR,
	MSG_READY,
	MSG_DONE
};
struct message
{
	int id;
	union
	{
		struct
		{
			uint64_t addr;
			uint32_t rkey;
		} mr;
	} data;
};

struct context
{
	char *buffer;
	struct ibv_context *ibv_ctx;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_comp_channel *comp_channel;
	struct ibv_mr *buffer_mr;
	struct message *msg;
	struct ibv_mr *msg_mr;
	std::thread  cq_poller_thread;
	uint64_t peer_addr;
	uint32_t peer_rkey;
	bool remote_idle;
};


struct _recv_chain
{
	void* data_ptr;
	_recv_chain* next;
};
bool rdma_all_init(bcube_struct& bcube_s);
bool bcube_send_by_rdma(tensor_table_entry& e, bcube_struct& bs, int stage);

#endif // HAVE_RDMA
#endif // __TENSORFLOW_BCUBE_RDMA_H__
