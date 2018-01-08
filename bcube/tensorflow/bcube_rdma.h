#ifndef __TENSORFLOW_BCUBE_RDMA_H__
#define __TENSORFLOW_BCUBE_RDMA_H__
#include <vector>
#include <string>
#include <pthread.h>
#if HAVE_RDMA
#include <rdma/rdma_cma.h>

#define MAX_CONCURRENCY 2

typedef struct _data_list_
{
	char* data_ptr;
	struct _data_list_* next;
	uint32_t data_len;
} node_item;

typedef struct _rdma_pack_
{
	struct rdma_cm_id* rdma_id;
	node_item* nit;
} _rdma_thread_pack_;

enum message_id
{
	MSG_INVALID = 0,
	MSG_MR,
	MSG_READY,
	MSG_DONE
};

/******
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
********/

typedef struct _key_exchange_
{
	int id;
	uint64_t md5;
	struct
	{
		uint64_t addr;
		uint32_t rkey;
	} key_info[MAX_CONCURRENCY];

} _key_exch;

typedef struct _ack
{
	int index;
} _ack_;

struct context
{
	struct ibv_context *ibv_ctx;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_comp_channel *comp_channel;

	pthread_t cq_poller_thread;

	//register buffer for remote to write
	char *           buffer[MAX_CONCURRENCY];
	struct ibv_mr *  buffer_mr[MAX_CONCURRENCY];

	//register ack mem is used for write to remote
	_ack_*			 ack[MAX_CONCURRENCY];
	struct ibv_mr *  ack_mr[MAX_CONCURRENCY];

	//indicate current status of each peer exchange
	bool 			 is_busy[MAX_CONCURRENCY];
	// index 0: store for local as tx
	// index 1: used to recv the remote info
	_key_exch*       k_exch[2];
	struct ibv_mr*   k_exch_mr[2];

	/*store the peer addr and rkey*/
	uint64_t 		 peer_addr[MAX_CONCURRENCY];
	uint32_t 		 peer_rkey[MAX_CONCURRENCY];
};

/***************
struct _context
{
	char *buffer;
	struct ibv_context *ibv_ctx;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_comp_channel *comp_channel;
	struct ibv_mr *buffer_mr;
	struct message *msg;
	struct ibv_mr *msg_mr;
	pthread_t cq_poller_thread;
	uint64_t peer_addr;
	uint32_t peer_rkey;
	bool remote_idle;
};
*************/

struct bcube_global_struct;
struct bcube_struct;
#include "bcube_message.h"

void rdma_bcube_init(bcube_struct&, bcube_global_struct&);
void rdma_bcube_send(tensor_table_entry& , bcube_struct& , int );
#include <stdarg.h>
void log_info(const char *format, ...);
#endif
#endif