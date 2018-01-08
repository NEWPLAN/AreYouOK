#include "bcube_rdma.h"

#define __RDMA_SLOW__ 1

#if HAVE_RDMA
#include <rdma/rdma_cma.h>

//50 M for default size;
const size_t BUFFER_SIZE = 50 * 1024 * 1024 + 1;
#define TIMEOUT_IN_MS 500
#define TEST_NZ(x) do { if ( (x)) rc_die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) rc_die("error: " #x " failed (returned zero/null)."); } while (0)
#define MIN_CQE 10

#endif // RDMA_SUPPORT

#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
#include <unordered_map>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include "bcube_utils.h"
#include "bcube_ops.h"
#include "bcube_comm.h"
#include "bcube_message.h"

#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <sys/time.h>

#define IS_CLIENT false
#define IS_SERVER true

static std::atomic_bool rdma_server_establisted(false);
static std::atomic_bool rdma_client_establisted(false);

static bool enable_padding = false;

//static std::mutex rdma_send_mutex;
//static std::mutex rdma_recv_mutex;
static node_item* out_send_list [20] = {nullptr};

static void rc_die(const char *reason)
{
	extern int errno;
	fprintf(stderr, "%s\nstrerror= %s\n", reason, strerror(errno));
	exit(-1);
}

static void show_msg(void* row_data)
{
	return;
	msg_struct* msg = (msg_struct*)row_data;
	printf("msg info:\n");
	printf("msg_length: %d\n", msg->msg_length);
	printf("name_length: %d\n", msg->name_len);
	printf("start position: %d\n", msg->start_pos);
	printf("msg.data[0]: %c\n", msg->data[0]);
	char* name = (char*)msg + sizeof(msg_struct);
	char* data = name + msg->name_len;
	char tmp = *data;
	*data = 0;
	printf("msg_name: %s\n", name);
	*data = tmp;
	for (int ii = 0; ii < 3; ii++)
		printf("%d ", ((int*)data)[ii]);
	printf("\n");
}

void log_info(const char *format, ...)
{
	char now_time[32];
	char s[1024];
	char content[1024];
	//char *ptr = content;
	struct tm *tmnow;
	struct timeval tv;
	bzero(content, 1024);
	va_list arg;
	va_start (arg, format);
	vsprintf (s, format, arg);
	va_end (arg);

	gettimeofday(&tv, NULL);
	tmnow = localtime(&tv.tv_sec);

	sprintf(now_time, "%04d/%02d/%02d %02d:%02d:%02d:%06ld ", \
	        tmnow->tm_year + 1900, tmnow->tm_mon + 1, tmnow->tm_mday, tmnow->tm_hour, \
	        tmnow->tm_min, tmnow->tm_sec, tv.tv_usec);

	sprintf(content, "%s %s", now_time, s);
	printf("%s", content);

}

#if HAVE_RDMA
static void node_counts(bcube_struct& bcube_s)
{
	bcube_s.bcube_node_count = 1;
	for (int lev = 0; lev < bcube_s.bcube_level; lev++)
	{
		bcube_s.bcube_node_count *= bcube_s.bcube0_size;
	}
}

static node_item* get_new_node(void)
{
	node_item* nit = (node_item*)std::malloc(sizeof(node_item));
	if (nit == nullptr)
	{
		printf("fatal error : malloc node_item error\n");
		exit(-1);
	}
	nit->next = nullptr;
	nit->data_ptr = nullptr;
	return nit;
}

static _rdma_thread_pack_* get_new_thread_pack(struct rdma_cm_id* id, node_item* nit)
{
	_rdma_thread_pack_* rtp = (_rdma_thread_pack_*)std::malloc(sizeof(_rdma_thread_pack_));
	if (rtp == nullptr)
	{
		printf("fatal error : malloc _rdma_thread_pack_ error\n");
		exit(-1);
	}
	rtp->rdma_id = id;
	rtp->nit = nit;
	return rtp;
}

/*确定当前bcube所处的节点和信息*/
int current_node_rank;
static void setup_node(bcube_struct& bcube_s)
{
	char* __rank__ = getenv("BCUBE_RANK");
	if (__rank__ != NULL)bcube_s.rank = atoi(__rank__);
	else
	{
		std::cerr << "error in get env rank, you must set up it before run." << std::endl;
		exit(-1);
	}
	current_node_rank = bcube_s.rank;
	bcube_s.local_info.node_index = bcube_s.rank;
	for (size_t lev = 0; lev < bcube_s.topo.size(); lev++)/*add myself ip into mynodes*/
		bcube_s.local_info.myip.push_back(bcube_s.topo[lev][bcube_s.rank].ip);

	/*发现邻居节点，加入到neighbor_info中*/
	for (int lev = 0; lev < bcube_s.bcube_level; lev++)/*each level*/
	{
		int * tmp_neigh = new int[bcube_s.bcube0_size - 1];
		std::vector<node> grp;
		Utils::getOneHopNeighbour(bcube_s.rank, lev, bcube_s.bcube0_size, 1, tmp_neigh);
		for (int neigh_index = 0; neigh_index < bcube_s.bcube0_size - 1; neigh_index++)
			grp.push_back(bcube_s.topo[lev][tmp_neigh[neigh_index]]);
		bcube_s.neighbor_info.push_back(grp);
		grp.clear();
		delete[] tmp_neigh;
	}
	return;

}

/*
12.12.10.XXX
12.12.11.XXX
*/
static void topology_init(bcube_struct& bcube_s)
{
	printf("constructing a BCube(%d,%d) topology\n", bcube_s.bcube0_size, bcube_s.bcube_level);
	node_counts(bcube_s);
	for (int leve = 0; leve < bcube_s.bcube_level; leve++)
	{
		std::string ip_addr = "12.12.";
		std::string leve_str = std::to_string((leve + 10));
		std::vector<node> tp;/*each level*/
		node tmp_node;
		ip_addr += leve_str + std::string(".");

		for (int nodenum = 0; nodenum < bcube_s.bcube_node_count; nodenum++)
		{
			tmp_node.ip = ip_addr + std::to_string(nodenum + 11);
			tmp_node.node_index = nodenum;
			tp.push_back(tmp_node);
		}
		bcube_s.topo.push_back(tp);
	}
	printf("BCube(%d,%d) is constructed done!\n", bcube_s.bcube0_size, bcube_s.bcube_level);
}
struct bcube_global_struct;
static void insert_to_recv_queue(bcube_global_struct& bgs, received_tensor_entry& rs_e)
{
	auto& bs = bgs.bcube_s;
	auto& recv_queue = bgs.receiv_tmp_tensor;
	string name = rs_e.tensor_name;/*point to get tensor_name*/
	auto it = recv_queue.find(name);
	if (it != recv_queue.end())/*exist before and insert into.*/
	{
		//printf("%s exit before, append it behinds, it size is %d\n", name.c_str(), it->second.size());
		auto& vec_msg = it->second;
		vec_msg.push_back(std::move(rs_e));
		if (vec_msg.size() == (size_t)(bs.bcube0_size - 1)*bs.bcube_level)
		{
			/*if all received done, move tensor to received tensor.*/
			//printf("tensor %s is ready to reduce, move to received tensor buf.\n", it->first.c_str());
			{
				std::lock_guard<std::mutex> recv_lock(bgs.tensor_recv_mutex);
				bgs.receiv_tensor.emplace(std::make_pair(name, std::move(vec_msg)));
			}
			recv_queue.erase(it);
		}
	}
	else
	{
		//printf("%s not exist... create one...\n",name.c_str());
		std::vector<received_tensor_entry> msg_record;
		//std::this_thread::sleep_for(std::chrono::seconds(1));
		msg_record.push_back(std::move(rs_e));
		recv_queue.emplace(std::make_pair(name, std::move(msg_record)));
		//printf("insert receive done\n");
	}
	return;
}

#ifdef HAVE_RDMA
/**********************
static void send_message(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

	memset(&wr, 0, sizeof(wr));

	wr.wr_id = (uintptr_t)id;
	wr.opcode = IBV_WR_SEND;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_SIGNALED;

	sge.addr = (uintptr_t)ctx->msg;
	sge.length = sizeof(*ctx->msg);
	sge.lkey = ctx->msg_mr->lkey;

	TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
}
*********************/
/********************
static void send_tensor(struct rdma_cm_id *id, char* buff, uint32_t len)
{
	struct context *ctx = (struct context *)id->context;
	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;
	if (buff)
	{
		memcpy(ctx->buffer, buff, len);
		std::free(buff);
	}
	else
	{
		printf("fatal error in send out data can not be empty\n");
		exit(-1);
	}
	{

		msg_struct* msg = (msg_struct*)(ctx->buffer);
		msg->rank = current_node_rank;
	}

	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)id;
	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = htonl(len);
	wr.wr.rdma.remote_addr = ctx->peer_addr;
	wr.wr.rdma.rkey = ctx->peer_rkey;
	if (len)
	{
		wr.sg_list = &sge;
		wr.num_sge = 1;

		sge.addr = (uintptr_t)ctx->buffer;
		sge.length = len;
		sge.lkey = ctx->buffer_mr->lkey;
	}
	TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
}
*****************/
static void _write_remote(struct rdma_cm_id *id, uint32_t len, uint32_t index)
{
	struct context *new_ctx = (struct context *)id->context;

	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

	memset(&wr, 0, sizeof(wr));

	wr.wr_id = (uintptr_t)id;

	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = index;//htonl(index);
	wr.wr.rdma.remote_addr = new_ctx->peer_addr[index];
	wr.wr.rdma.rkey = new_ctx->peer_rkey[index];

	if (len)
	{
		wr.sg_list = &sge;
		wr.num_sge = 1;

		sge.addr = (uintptr_t)new_ctx->buffer[index];
		sge.length = len;
		sge.lkey = new_ctx->buffer_mr[index]->lkey;
	}

	TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
}

static void _post_receive(struct rdma_cm_id *id, uint32_t index)
{
	struct ibv_recv_wr wr, *bad_wr = NULL;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uint64_t)id;
	wr.sg_list = NULL;
	wr.num_sge = 0;

	TEST_NZ(ibv_post_recv(id->qp, &wr, &bad_wr));
}
static void _ack_remote(struct rdma_cm_id *id, uint32_t index)
{
	struct context *new_ctx = (struct context *)id->context;

	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

	memset(&wr, 0, sizeof(wr));

	wr.wr_id = (uintptr_t)id;

	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = index;//htonl(index);
	wr.wr.rdma.remote_addr = new_ctx->peer_addr[index];
	wr.wr.rdma.rkey = new_ctx->peer_rkey[index];

	new_ctx->ack[index]->index = index;

	{
		wr.sg_list = &sge;
		wr.num_sge = 1;

		sge.addr = (uintptr_t)new_ctx->ack[index];
		sge.length = sizeof(_ack_);
		sge.lkey = new_ctx->ack_mr[index]->lkey;
	}

	TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
}

/*************
static void batch_send(struct rdma_cm_id * id, node_item*& nit)
{
	struct context *ctx = (struct context *)id->context;
	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

	node_item* tp_nit = nit;
	char* _buff = ctx->buffer;

	uint32_t send_byte = 0;

	while (nit->next == nullptr)
		std::this_thread::sleep_for(std::chrono::nanoseconds(100));

	while (nit->next != nullptr)
	{
		node_item* free_tp_node = nit;
		nit = nit->next;
		msg_struct* msg = (msg_struct*)(nit->data_ptr);
		if ((send_byte + msg->msg_length) > BUFFER_SIZE)
		{
			//too much data... should send next batch...
			if (send_byte == 0)
			{
				printf("fatal error : one tensor is too huge...\n");
				exit(-1);
			}
			nit = free_tp_node;
		}
		else
		{
			msg->rank = current_node_rank;
			send_byte += msg->msg_length;
			std::free(free_tp_node);
			std::memcpy(_buff, nit->data_ptr, msg->msg_length);
			_buff += msg->msg_length;
			std::free(nit->data_ptr);
		}
	}//padding all available data...

	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)id;
	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = htonl(send_byte);
	wr.wr.rdma.remote_addr = ctx->peer_addr;
	wr.wr.rdma.rkey = ctx->peer_rkey;
	if (send_byte)
	{
		wr.sg_list = &sge;
		wr.num_sge = 1;
		sge.addr = (uintptr_t)ctx->buffer;
		sge.length = send_byte;
		sge.lkey = ctx->buffer_mr->lkey;
	}
	TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
}

static node_item* bacth_send_by_RDMA(struct ibv_wc *wc, node_item* nit)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;

	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		printf("send thread %ld will never be here!!!!!\n", pthread_self());
		exit(0);
	}
	else if (wc->opcode & IBV_WC_RECV)
	{
		switch (ctx->msg->id)
		{
			case MSG_MR:
				{
					ctx->peer_addr = ctx->msg->data.mr.addr;
					ctx->peer_rkey = ctx->msg->data.mr.rkey;
					printf("received remote memory address and key\n");
					ctx->remote_idle = true;
					batch_send(id, nit);
				}
				break;
			case MSG_READY:
				{
					ctx->remote_idle = true;
					batch_send(id, nit);
				}
				break;
			case MSG_DONE:
				{
					printf("received DONE, disconnecting\n");
					rdma_disconnect(id);
					return nit;
				}
				break;
			default:
				break;
		}
		post_receive_client(id);
	}
	return nit;
}

static void* batch_recv_by_RDMA(struct ibv_wc *wc, uint32_t& recv_len)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;
	void* _data = nullptr;

	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		uint32_t size = ntohl(wc->imm_data);
		recv_len = size;
		_data = (void*)std::malloc(sizeof(char) * size);

		if (_data == nullptr)
		{
			printf("fatal error in recv data malloc!!!!\n");
			exit(-1);
		}
		std::memcpy(_data, ctx->buffer, size);

		post_receive_server(id);
		ctx->msg->id = MSG_READY;
		send_message(id);
	}
	else if (wc->opcode & IBV_WC_RECV)
	{
		printf("recv thread %ld will never be here!!!!!\n", pthread_self());
		exit(0);
	}
	return _data;
}
**********/
static void* corcurency_recv_by_RDMA(struct ibv_wc *wc, uint32_t& recv_len)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;
	void* _data = nullptr;

	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		log_info("recv with IBV_WC_RECV_RDMA_WITH_IMM\n");
		log_info("imm_data is %d\n", wc->imm_data);
		//uint32_t size = ntohl(wc->imm_data);
		uint32_t index = wc->imm_data;
		uint32_t size = *((uint32_t*)(ctx->buffer[index]));
		char* recv_data_ptr = ctx->buffer[index] + sizeof(uint32_t);

		recv_len = size;
		_data = (void*)std::malloc(sizeof(char) * size);

		if (_data == nullptr)
		{
			printf("fatal error in recv data malloc!!!!\n");
			exit(-1);
		}
		std::memcpy(_data, recv_data_ptr, size);

		_post_receive(id, wc->imm_data);
		_ack_remote(id, wc->imm_data);

	}
	else if (wc->opcode == IBV_WC_RECV)
	{
		switch (ctx->k_exch[1]->id)
		{
			case MSG_MR:
				{
					log_info("recv MD5 is %llu\n", ctx->k_exch[1]->md5);
					log_info("imm_data is %d\n", wc->imm_data);
					for (int index = 0; index < MAX_CONCURRENCY; index++)
					{
						ctx->peer_addr[index] = ctx->k_exch[1]->key_info[index].addr;
						ctx->peer_rkey[index] = ctx->k_exch[1]->key_info[index].rkey;
					}
				} break;
			default:
				break;
		}
	}
	return _data;
}

static node_item* send_tensor(struct rdma_cm_id *id, node_item* nit, int index)
{
	struct context *ctx = (struct context *)id->context;


	while (nit->next == nullptr)
		std::this_thread::sleep_for(std::chrono::nanoseconds(10));
	{
		/*release the old source*/
		node_item* free_tp_node;
		free_tp_node = nit;
		nit = nit->next;
		std::free(free_tp_node);
	}
	/*encode msg_length and buffer*/
	uint32_t msg_len = ((msg_struct*)(nit->data_ptr))->msg_length;

	if ((msg_len + sizeof(uint32_t)) > BUFFER_SIZE)
	{
		perror("fatal error, send msg length is too long\n");
		exit(-1);
	}

	char* _buff = ctx->buffer[index];
	std::memcpy(_buff, (char*)(&msg_len), sizeof(uint32_t));
	_buff += sizeof(uint32_t);
	std::memcpy(_buff, (char*)(nit->data_ptr), msg_len);
	_write_remote(id, msg_len, index);

	return nit;
}

static node_item* concurrency_send_by_RDMA(struct ibv_wc *wc, node_item* nit, int& mem_used)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;

	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		log_info("recv with IBV_WC_RECV_RDMA_WITH_IMM\n");
		log_info("imm_data is %d\n", wc->imm_data);
		_post_receive(id, wc->imm_data);
		nit = send_tensor(id, nit, wc->imm_data);
	}
	else if (wc->opcode == IBV_WC_RECV)
	{
		switch (ctx->k_exch[1]->id)
		{
			case MSG_MR:
				{
					log_info("recv MD5 is %llu\n", ctx->k_exch[1]->md5);
					for (int index = 0; index < MAX_CONCURRENCY; index++)
					{
						//reserved the (buffer)key info from server.
						ctx->peer_addr[index] = ctx->k_exch[1]->key_info[index].addr;
						ctx->peer_rkey[index] = ctx->k_exch[1]->key_info[index].rkey;
					}
					/**send one tensor...**/
					nit = send_tensor(id, nit, 0);
					mem_used++;
				}
				break;
			default:
				break;
		}
	}
	return nit;
}

/*******************
static node_item* send_by_RDMA(struct ibv_wc *wc, node_item* nit)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;

	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		printf("send thread %ld will never be here!!!!!\n", pthread_self());
		exit(0);
	}
	else if (wc->opcode & IBV_WC_RECV)
	{
		if (ctx->msg->id == MSG_MR)
		{
			ctx->peer_addr = ctx->msg->data.mr.addr;
			ctx->peer_rkey = ctx->msg->data.mr.rkey;
			printf("received remote memory address and key\n");
			ctx->remote_idle = true;

			while (nit->next == nullptr)
				std::this_thread::sleep_for(std::chrono::nanoseconds(1));
			//std::this_thread::sleep_for(std::chrono::milliseconds(1));
			node_item* free_tp_node;
			{
				//std::lock_guard<std::mutex> send_lock(rdma_send_mutex);
				free_tp_node = nit;
				nit = nit->next;
			}
			std::free(free_tp_node);
			send_tensor(id, (char*)(nit->data_ptr), ((msg_struct*)(nit->data_ptr))->msg_length);

		}
		else if (ctx->msg->id == MSG_DONE)
		{
			printf("received DONE, disconnecting\n");
			rdma_disconnect(id);
			return nit;
		}
		else if (ctx->msg->id == MSG_READY)
		{
			ctx->remote_idle = true;

			while (nit->next == nullptr)
				std::this_thread::sleep_for(std::chrono::nanoseconds(100));

			//std::this_thread::sleep_for(std::chrono::milliseconds(1));
			node_item* free_tp_node;
			{
				//std::lock_guard<std::mutex> send_lock(rdma_send_mutex);
				free_tp_node = nit;
				nit = nit->next;
			}
			std::free(free_tp_node);

			//node_item* next_node = nit->next;
			//std::free(nit);
			//nit = next_node;
			send_tensor(id, (char*)(nit->data_ptr), ((msg_struct*)(nit->data_ptr))->msg_length);

		}
		post_receive_client(id);
	}
	return nit;
}
***************************/

/**************************
static void* recv_by_RDMA(struct ibv_wc *wc, node_item* nit)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;
	void* _data = nullptr;

	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		uint32_t size = ntohl(wc->imm_data);
		struct sockaddr_in* client_addr = (struct sockaddr_in *)rdma_get_peer_addr(id);
		static int64_t lpop = 0;
		//if (lpop % 100 == 0)
		//printf("thread: %ld received %i bytes from client %s!!!!!!!!!!!!!%p!!!!!!!!!!!!!!!!!!!!!!!\n", pthread_self(), size, inet_ntoa(client_addr->sin_addr), nit);
		lpop++;
		//printf("%s\n",ctx->buffer);
		msg_struct* msg = (msg_struct*)(ctx->buffer);
		//printf("recv from node %d count: %d\n", msg->rank, ++recvcount[msg->rank]);
		_data = (void*)std::malloc(sizeof(char) * size);
		if (size != ((msg_struct*)(ctx->buffer))->msg_length)
		{
			printf("fatal error: recv send_message length is not equal...\n");
			exit(0);
		}
		if (_data == nullptr)
		{
			printf("fatal error in recv data malloc!!!!\n");
			exit(-1);
		}
		std::memcpy(_data, ctx->buffer, size);

		post_receive_server(id);
		ctx->msg->id = MSG_READY;
		send_message(id);
	}
	else if (wc->opcode & IBV_WC_RECV)
	{
		printf("recv thread %ld will never be here!!!!!\n", pthread_self());
		exit(0);
	}
	return _data;
}
****************/

static void *recv_poll_cq(void *rtp)
{
	struct ibv_cq *cq = NULL;
	struct ibv_wc wc[MAX_CONCURRENCY * 2];
	struct rdma_cm_id *id = ((_rdma_thread_pack_ *)rtp)->rdma_id;
	node_item* nit = ((_rdma_thread_pack_ *)rtp)->nit;

	struct context *ctx = (struct context *)id->context;
	void *ev_ctx = NULL;

	std::free((_rdma_thread_pack_ *)rtp);

	while (true)
	{
		TEST_NZ(ibv_get_cq_event(ctx->comp_channel, &cq, &ev_ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));

		int wc_num = ibv_poll_cq(cq, MAX_CONCURRENCY * 2, wc);

		for (int index = 0; index < wc_num; index++)
		{
			if (wc[index].status == IBV_WC_SUCCESS)
			{
				/*****here to modified recv* wc---->wc[index]****/
				void* recv_data = nullptr;
				uint32_t recv_len;
				/*
								if (enable_padding)
									recv_data = batch_recv_by_RDMA(&wc, recv_len);
								else
									recv_data = recv_by_RDMA(&wc, nit);
				*/
				recv_data = corcurency_recv_by_RDMA(&wc[index], recv_len);
				if (recv_data != nullptr)
				{
					//received data, will append to recv_chain...
					auto new_node = get_new_node();
					new_node->data_ptr = (char*)recv_data;
					new_node->data_len = recv_len;
					nit->next = new_node;
					nit = new_node;
				}
			}
			else
			{
				printf("\nwc = %s\n", ibv_wc_status_str(wc[index].status));
				rc_die("poll_cq: status is not IBV_WC_SUCCESS");
			}
		}
	}
	return NULL;
}
static void *send_poll_cq(void *rtp)
{
	struct ibv_cq *cq = NULL;
	struct ibv_wc wc[MAX_CONCURRENCY * 2];
	struct rdma_cm_id *id = ((_rdma_thread_pack_ *)rtp)->rdma_id;
	node_item* nit = ((_rdma_thread_pack_ *)rtp)->nit;

	std::free((_rdma_thread_pack_ *)rtp);

	struct context *ctx = (struct context *)id->context;
	void *ev_ctx = NULL;

	int mem_used = 0;

	while (1)
	{
		TEST_NZ(ibv_get_cq_event(ctx->comp_channel, &cq, &ev_ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));

		int wc_num = ibv_poll_cq(cq, MAX_CONCURRENCY * 2, wc);

		if (wc_num < 0)
		{
			perror("fatal error in ibv_poll_cq, -1");
			exit(-1);
		}

		for (int index = 0; index < wc_num; index++)
		{
			if (wc[index].status == IBV_WC_SUCCESS)
			{
				nit = concurrency_send_by_RDMA(&wc[index], nit, mem_used);
			}
			else
			{
				printf("\nwc = %s\n", ibv_wc_status_str(wc[index].status));
				rc_die("poll_cq: status is not IBV_WC_SUCCESS");
			}
		}
		if (mem_used)
		{
			struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc[index]->wr_id;
			struct context *ctx = (struct context *)id->context;
			for (mem_used; mem_used < MAX_CONCURRENCY; mem_used++)
			{
				if (nit->next == nullptr) break;
				nit = send_tensor(id, nit, mem_used);
			}/*send used next buffer*/
		}
	}
	return NULL;
}

static void unlock_insert_to_recv_queue(bcube_global_struct& bgs, received_tensor_entry& rs_e);





static struct ibv_pd * rc_get_pd(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	return ctx->pd;
}

static void _build_params(struct rdma_conn_param *params)
{
	memset(params, 0, sizeof(*params));

	params->initiator_depth = params->responder_resources = 1;
	params->rnr_retry_count = 7; /* infinite retry */
	params->retry_count = 7;
}

static void _build_context(struct rdma_cm_id *id, bool is_server, node_item* nit)
{
	struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
	s_ctx->ibv_ctx = id->verbs;
	TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ibv_ctx));
	TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ibv_ctx));
	TEST_Z(s_ctx->cq = ibv_create_cq(s_ctx->ibv_ctx, MAX_CONCURRENCY * 2 + 10, NULL, s_ctx->comp_channel, 0));
	TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));
	id->context = (void*)s_ctx;
	if (is_server)
	{
		_rdma_thread_pack_* rtp = get_new_thread_pack(id, nit);
		TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, recv_poll_cq, (void*)rtp));
		id->context = (void*)s_ctx;
	}
}

static void _build_qp_attr(struct ibv_qp_init_attr *qp_attr, struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	memset(qp_attr, 0, sizeof(*qp_attr));
	qp_attr->send_cq = ctx->cq;
	qp_attr->recv_cq = ctx->cq;
	qp_attr->qp_type = IBV_QPT_RC;

	qp_attr->cap.max_send_wr = MAX_CONCURRENCY + 2;
	qp_attr->cap.max_recv_wr = MAX_CONCURRENCY + 2;
	qp_attr->cap.max_send_sge = 1;
	qp_attr->cap.max_recv_sge = 1;
}

static void _build_connection(struct rdma_cm_id *id, bool is_server, node_item* nit)
{
	struct ibv_qp_init_attr qp_attr;
	_build_context(id, is_server, nit);
	_build_qp_attr(&qp_attr, id);

	struct context *ctx = (struct context *)id->context;
	TEST_NZ(rdma_create_qp(id, ctx->pd, &qp_attr));
}

static void _on_pre_conn(struct rdma_cm_id *id)
{
	struct context *new_ctx = (struct context *)id->context;


	for (int index = 0; index < MAX_CONCURRENCY; index++)
	{
		posix_memalign((void **)(&(new_ctx->buffer[index])), sysconf(_SC_PAGESIZE), BUFFER_SIZE);
		TEST_Z(new_ctx->buffer_mr[index] = ibv_reg_mr(rc_get_pd(), new_ctx->buffer[index], BUFFER_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

		posix_memalign((void **)(&(new_ctx->ack[index])), sysconf(_SC_PAGESIZE), sizeof(_ack_));
		TEST_Z(new_ctx->ack_mr[index] = ibv_reg_mr(rc_get_pd(), new_ctx->ack[index],
		                                sizeof(_ack_), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
	}
	log_info("register %d tx_buffer and rx_ack\n", index);

	{
		posix_memalign((void **)(&(new_ctx->k_exch[0])), sysconf(_SC_PAGESIZE), sizeof(_key_exch));
		TEST_Z(new_ctx->k_exch_mr[0] = ibv_reg_mr(rc_get_pd(), new_ctx->k_exch[0], sizeof(_key_exch), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

		posix_memalign((void **)(&(new_ctx->k_exch[1])), sysconf(_SC_PAGESIZE), sizeof(_key_exch));
		TEST_Z(new_ctx->k_exch_mr[1] = ibv_reg_mr(rc_get_pd(), new_ctx->k_exch[1], sizeof(_key_exch), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

	}
	log_info("register rx_k_exch (index:0) and tx_k_exch (index:1)\n");

	struct ibv_recv_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

	memset(&wr, 0, sizeof(wr));

	wr.wr_id = (uintptr_t)id;
	wr.sg_list = &sge;
	wr.num_sge = 1;

	sge.addr = (uintptr_t)(new_ctx->k_exch[1]);
	sge.length = sizeof(_key_exch);
	sge.lkey = new_ctx->k_exch_mr[1]->lkey;



	TEST_NZ(ibv_post_recv(id->qp, &wr, &bad_wr));


	for (uint32_t index = 0; index < MAX_CONCURRENCY; index++)
	{
		log_info("post recv index : %u\n", index);
		_post_receive(id, index);
	}
}

/**server on connection***/
static void _on_connection(struct rdma_cm_id *id, , bool is_server)
{
	struct context *new_ctx = (struct context *)id->context;

	int index = 0;

	new_ctx->k_exch[0]->id = MSG_MR;
	if (is_server)
		new_ctx->k_exch[0]->md5 = 6666;
	else
		new_ctx->k_exch[0]->md5 = 5555;

	log_info("k_exch md5 is %llu\n", new_ctx->k_exch[0]->md5);
	if (is_server)
	{
		for (index = 0; index < MAX_CONCURRENCY; index++)
		{
			new_ctx->k_exch[0]->key_info[index].addr = (uintptr_t)(new_ctx->buffer_mr[index]->addr);
			new_ctx->k_exch[0]->key_info[index].rkey = (new_ctx->buffer_mr[index]->rkey);
		}

	}
	else
	{
		for (index = 0; index < MAX_CONCURRENCY; index++)
		{
			new_ctx->k_exch[0]->key_info[index].addr = (uintptr_t)(new_ctx->ack_mr[index]->addr);
			new_ctx->k_exch[0]->key_info[index].rkey = (new_ctx->ack_mr[index]->rkey);
		}

	}


	//send to myself info to peer
	{
		struct ibv_send_wr wr, *bad_wr = NULL;
		struct ibv_sge sge;

		memset(&wr, 0, sizeof(wr));

		wr.wr_id = (uintptr_t)id;
		wr.opcode = IBV_WR_SEND;
		wr.sg_list = &sge;
		wr.num_sge = 1;
		wr.send_flags = IBV_SEND_SIGNALED;

		sge.addr = (uintptr_t)(new_ctx->k_exch[0]);
		sge.length = sizeof(_key_exch);
		sge.lkey = new_ctx->k_exch_mr[0]->lkey;

		TEST_NZ(ibv_post_send(id->qp, &wr, &bad_wr));
	}
	log_info("share my registed mem rx_buffer for peer write to\n");
}



static void _on_disconnect(struct rdma_cm_id *id)
{
	struct context *new_ctx = (struct context *)id->context;

	for (int index = 0; index < MAX_CONCURRENCY; index++)
	{
		ibv_dereg_mr(new_ctx->buffer_mr[index]);
		ibv_dereg_mr(new_ctx->ack_mr[index]);

		free(new_ctx->buffer[index]);
		free(new_ctx->ack[index]);
	}

	{
		ibv_dereg_mr(new_ctx->k_exch_mr[0]);
		ibv_dereg_mr(new_ctx->k_exch_mr[1]);

		free(new_ctx->k_exch[0]);
		free(new_ctx->k_exch[1]);
	}

	free(new_ctx);
}

void recv_tensor_from_list(bcube_global_struct& bgs, std::vector<node_item*>& _recv_chain)
{
	auto& recv_chain = _recv_chain;

	msg_struct msg_buf;
	while (true)
	{
		for (auto& recv_list : recv_chain)
		{
			if (recv_list == nullptr)
			{
				printf("fatal error in malloc recv_list！！！\n");
				exit(-1);
			}
			if (recv_list->next != nullptr)
			{
				{
					node_item* free_tmp = recv_list;
					recv_list = free_tmp->next;
					std::free(free_tmp);
					free_tmp = nullptr;
				}
				if (enable_padding)
				{
					uint32_t data_len = 0;
					void* new_msg = nullptr;
					while (data_len < recv_list->data_len)
					{
						new_msg = recv_list->data_ptr + data_len;
						received_tensor_entry e;
						show_msg(new_msg);
						tensor_msg::decode(e, new_msg);
						unlock_insert_to_recv_queue(bgs, e);
						//insert_to_recv_queue(bgs, e);
						data_len += ((msg_struct*)new_msg)->msg_length;
					}
				}
				else
				{
					//insert into recv_tensor...
					void* new_msg = recv_list->data_ptr;
					received_tensor_entry e;
					show_msg(new_msg);
					tensor_msg::decode(e, new_msg);
					//insert_to_recv_queue(bgs, e);
					unlock_insert_to_recv_queue(bgs, e);
					new_msg = nullptr;

				}
				std::free((char*)(recv_list->data_ptr));
				recv_list->data_ptr = nullptr;
			}
		}
	}
}

static void recv_RDMA(bcube_global_struct& bgs)
{
	bcube_struct& bs = bgs.bcube_s;
	struct rdma_cm_event *event = NULL;
	struct rdma_conn_param cm_params;
	int connecting_client_cnt = 0;
	int client_counts = (bs.bcube0_size - 1) * bs.bcube_level;
	printf("server is inited done (RDMA), waiting for %d client connecting....:)\n", client_counts);
	build_params(&cm_params);
	//std::vector<node_item*> recv_chain;
	auto& recv_chain = bgs.recv_chain;

	while (rdma_get_cm_event(bs.event_channel, &event) == 0)
	{
		struct rdma_cm_event event_copy;

		memcpy(&event_copy, event, sizeof(*event));
		rdma_ack_cm_event(event);


		if (event_copy.event == RDMA_CM_EVENT_CONNECT_REQUEST)
		{
			node_item* nit = get_new_node();
			recv_chain.push_back(nit);
			build_connection(event_copy.id, IS_SERVER, nit);
			_on_pre_conn(event_copy.id);
			TEST_NZ(rdma_accept(event_copy.id, &cm_params));
		}
		else if (event_copy.event == RDMA_CM_EVENT_ESTABLISHED)
		{
			_on_connection(event_copy.id, true);
			bs.recv_rdma_cm_id.push_back(event_copy.id);

			struct sockaddr_in* client_addr = (struct sockaddr_in *)rdma_get_peer_addr(event_copy.id);
			printf("client[%s,%d] is connecting (RDMA) now... \n", inet_ntoa(client_addr->sin_addr), client_addr->sin_port);
			connecting_client_cnt++;
			if (connecting_client_cnt == client_counts)
				break;
		}
		else if (event_copy.event == RDMA_CM_EVENT_DISCONNECTED)
		{
			rdma_destroy_qp(event_copy.id);
			_on_disconnect(event_copy.id);
			rdma_destroy_id(event_copy.id);
			connecting_client_cnt--;
			if (connecting_client_cnt == 0)
				break;
		}
		else
		{
			rc_die("unknown event server\n");
		}
	}
	printf("%d clients have connected to my node (RDMA), ready to receiving loops\n", client_counts);

	//int msg_len = sizeof(msg_struct);
	//auto& fd_vect = bgs.bcube_s.recv_rdma_cm_id;
	//int fd_num = fd_vect.size();
	rdma_server_establisted = true;
	recv_tensor_from_list(bgs, recv_chain);


//
	printf("RDMA recv loops exit now...\n");
	return;
}
#endif
extern bcube_global_struct bcube_gs;

static void rdma_server_init(bcube_struct& bs)
{
	int init_loops = 0;
	struct sockaddr_in sin;
	printf("init a server (RDMA)....\n");
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;/*ipv4*/
	sin.sin_port = htons(bs.server_port);/*server listen public ports*/
	sin.sin_addr.s_addr = INADDR_ANY;/*listen any connects*/

	TEST_Z(bs.event_channel = rdma_create_event_channel());
	TEST_NZ(rdma_create_id(bs.event_channel, &bs.listener, NULL, RDMA_PS_TCP));

	while (rdma_bind_addr(bs.listener, (struct sockaddr *)&sin))
	{
		std::cerr << "server init failed (RDMA): error in bind socket, will try it again in 2 seconds..." << std::endl;
		if (init_loops > 10)
		{
			rdma_destroy_id(bs.listener);
			rdma_destroy_event_channel(bs.event_channel);
			exit(-1);
		}
		std::this_thread::sleep_for(std::chrono::seconds(2));
		init_loops++;
	}

	int client_counts = (bs.bcube0_size - 1) * bs.bcube_level;
	if (rdma_listen(bs.listener, client_counts))
	{
		std::cerr << "server init failed (RDMA): error in server listening" << std::endl;
		rdma_destroy_id(bs.listener);
		rdma_destroy_event_channel(bs.event_channel);
		exit(-1);
	}

	bcube_gs.bg_thread.push_back(std::thread(recv_RDMA, std::ref(bcube_gs)));

	std::this_thread::sleep_for(std::chrono::seconds(1));
	return;
}



static void rdma_client_init(bcube_struct& bs)
{
	std::cout << "client inited (RDMA) start" << std::endl;
	for (size_t lev = 0; lev < bs.neighbor_info.size(); lev++)
	{
		for (size_t index = 0; index < bs.neighbor_info[lev].size(); index++)
		{
			struct rdma_cm_id *conn = NULL;
			struct rdma_event_channel *ec = NULL;
			std::string local_eth = bs.local_info.myip[lev];/*get each lev ip*/
			struct sockaddr_in ser_in, local_in;/*server ip and local ip*/
			int connect_count = 0;
			memset(&ser_in, 0, sizeof(ser_in));
			memset(&local_in, 0, sizeof(local_in));

			/*bind remote socket*/
			ser_in.sin_family = AF_INET;
			ser_in.sin_port = htons(bs.server_port);/*connect to public port remote*/
			inet_pton(AF_INET, bs.neighbor_info[lev][index].ip.c_str(), &ser_in.sin_addr);

			/*bind local part*/
			local_in.sin_family = AF_INET;
			std::cout << local_eth.c_str() << "----->" << bs.neighbor_info[lev][index].ip.c_str() << std::endl;
			inet_pton(AF_INET, local_eth.c_str(), &local_in.sin_addr);

			TEST_Z(ec = rdma_create_event_channel());
			TEST_NZ(rdma_create_id(ec, &conn, NULL, RDMA_PS_TCP));
			TEST_NZ(rdma_resolve_addr(conn, (struct sockaddr*)(&local_in), (struct sockaddr*)(&ser_in), TIMEOUT_IN_MS));

			struct rdma_cm_event *event = NULL;
			struct rdma_conn_param cm_params;

			build_params(&cm_params);
			while (rdma_get_cm_event(ec, &event) == 0)
			{
				struct rdma_cm_event event_copy;
				memcpy(&event_copy, event, sizeof(*event));
				rdma_ack_cm_event(event);
				if (event_copy.event == RDMA_CM_EVENT_ADDR_RESOLVED)
				{
					_build_connection(event_copy.id, IS_CLIENT, nullptr);
					_on_pre_conn(event_copy.id);
					TEST_NZ(rdma_resolve_route(event_copy.id, TIMEOUT_IN_MS));
				}
				else if (event_copy.event == RDMA_CM_EVENT_ROUTE_RESOLVED)
				{
					TEST_NZ(rdma_connect(event_copy.id, &cm_params));
				}
				else if (event_copy.event == RDMA_CM_EVENT_ESTABLISHED)
				{
					struct context *ctx = (struct context *)event_copy.id->context;
					node_item* nit = get_new_node();
					_rdma_thread_pack_* rtp = get_new_thread_pack(event_copy.id, nit);
					//bs.neighbor_info[lev][index].send_list = nit;
					bs.topo[0][bs.neighbor_info[lev][index].node_index].send_list = nit;
					printf("out node id is %d\n", bs.neighbor_info[lev][index].node_index);
					out_send_list[bs.neighbor_info[lev][index].node_index] = nit;
					TEST_NZ(pthread_create(&ctx->cq_poller_thread, NULL, send_poll_cq, (void*)rtp));
					std::cout << local_eth << " has connected to server[ " << bs.neighbor_info[lev][index].ip << " , " << bs.server_port << " ]" << std::endl;

					{
						_on_connection(event_copy.id, false);
					}
					break;
				}
				else if (event_copy.event == RDMA_CM_EVENT_REJECTED)
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
					connect_count++;
					_on_disconnect(event_copy.id);
					rdma_destroy_qp(event_copy.id);
					rdma_destroy_id(event_copy.id);
					rdma_destroy_event_channel(ec);
					if (connect_count > 10 * 600)/*after 600 seconds, it will exit.*/
					{
						std::cerr << 600 << "seconds is passed, error in connect to server" << bs.neighbor_info[lev][index].ip << ", check your network condition" << std::endl;
						exit(-1);
					}
					else
					{
						TEST_Z(ec = rdma_create_event_channel());
						TEST_NZ(rdma_create_id(ec, &conn, NULL, RDMA_PS_TCP));
						TEST_NZ(rdma_resolve_addr(conn, (struct sockaddr*)(&local_in), (struct sockaddr*)(&ser_in), TIMEOUT_IN_MS));
					}
				}
				else
				{
					printf("event = %d\n", event_copy.event);
					rc_die("unknown event client\n");
				}
			}
			bs.topo[lev][bs.neighbor_info[lev][index].node_index].send_rdma_event_channel = ec;
			bs.topo[lev][bs.neighbor_info[lev][index].node_index].send_rdma_cm_id = conn;
			bs.neighbor_info[lev][index].send_rdma_event_channel = ec;
			bs.neighbor_info[lev][index].send_rdma_cm_id = conn;
		}
	}
	rdma_client_establisted = true;
	std::cout << "client inited done" << std::endl;
}



static void outputs(bcube_struct& bs, int** sendMatrix, int N, int para_num, int p, int s)
{
#if __outs_all_
	printf("para_num = %d\n", para_num);
#endif
	for (int i = 0; i < N; i++)
	{
		auto& strategy = bs.nodes_send_strategy[i][s][p];
		for (int j = 0; j < N; j++)
		{
			if (-1 != sendMatrix[i][j])
			{
				send_to_one to_one_node;
				to_one_node.node_id = j;
#if __outs_all_
				printf("\nnode%d->node%d, send paraID: %d", i, j, sendMatrix[i][j]);
				for (int cou = 1; cou < para_num; cou++)printf(", %d", sendMatrix[i][j] + cou);
#endif
				for (int col = 0; col < para_num; col++)to_one_node.paraid.push_back(sendMatrix[i][j] + col);
				to_one_node.block_num = para_num;
				strategy.push_back(to_one_node);
			}

		}
#if __outs_all_
		printf("\n");
#endif
	}
}
static void set_sock_fd(bcube_struct& bs)
{
	for (size_t step_index = 0; step_index < bs.my_strategy.size(); step_index++)
	{
		for (size_t proc_index = 0; proc_index < bs.my_strategy[step_index].size(); proc_index++)
		{
			for (size_t node_index = 0; node_index < bs.my_strategy[step_index][proc_index].size(); node_index++)
			{
				auto& each_node = bs.my_strategy[step_index][proc_index][node_index];
				for (size_t lev_index = 0; lev_index < bs.neighbor_info.size(); lev_index++)
				{
					for (size_t neigh_index = 0; neigh_index < bs.neighbor_info[lev_index].size(); neigh_index++)
					{
						if (each_node.node_id == bs.neighbor_info[lev_index][neigh_index].node_index)
						{
#if HAVE_RDMA
							each_node.send_list = bs.neighbor_info[lev_index][neigh_index].send_list;
#endif
							each_node.socket_fd = bs.neighbor_info[lev_index][neigh_index].remote_fd;
						}
					}
				}
			}
		}
	}
}

static void get_send_strategy(bcube_struct& bs)
{
	int n = bs.bcube0_size;
	int k = bs.bcube_level - 1;
	int N = bs.bcube_node_count;
	int para_num;
	int **sendMatrix = new int*[N];

	for (int i = 0; i < N; i++)sendMatrix[i] = new int[N];

	{
		/*resize strategy vector*/
		bs.nodes_send_strategy.resize(N);
		for (size_t node_index = 0; node_index < bs.nodes_send_strategy.size(); node_index++)
		{
			bs.nodes_send_strategy[node_index].resize((k + 1) * 2);
			for (size_t step_index = 0; step_index < bs.nodes_send_strategy[node_index].size(); step_index++)
			{
				bs.nodes_send_strategy[node_index][step_index].resize((k + 1));
			}
		}
	}
	for (int s = 0; s <= k; s++)
	{
		for (int p = 0; p <= k; p++)
		{
#if __outs_all_
			printf("\n\nin scatter stage: %d\t using p %d\n", s, p);
#endif
			Utils::getScatterMatrix(p, s, n, k, N, sendMatrix, para_num);
			outputs(bs, sendMatrix, N, para_num, p, s);
		}

	}
	for (int s = 0; s <= k; s++)
	{
		for (int p = 0; p <= k; p++)
		{
#if __outs_all_
			printf("\n\nin gather stage: %d\t using p %d\n", s, p);
#endif
			Utils::getGatherMatrix(p, s, n, k, N, sendMatrix, para_num);
			outputs(bs, sendMatrix, N, para_num, p, s + k + 1);
		}
	}
	for (int row = 0; row < N; row++)
		delete[] sendMatrix[row];
	delete[] sendMatrix;
	bs.my_strategy = bs.nodes_send_strategy[bs.rank];
	set_sock_fd(bs);
	bs.nodes_send_strategy[bs.rank] = bs.my_strategy;
	//show_each_node(bs, bs.rank);
}


static bool rdma_check_bcube_is_inited_done(bcube_struct& bs)
{
	std::cout << "check bcube inite status, not yet finished" << std::endl;
	return rdma_server_establisted && rdma_client_establisted;
}

#include "stdlib.h"
#include "stdio.h"
#include "signal.h"
#include "execinfo.h"

void fun_dump( int no)
{
	char _signal[64][32] =
	{
		"1: SIGHUP", "2: SIGINT", "3: SIGQUIT", "4: SIGILL",
		"5: SIGTRAP", "6: SIGABRT", "7: SIGBUS", "8: SIGFPE",
		"9: SIGKILL", "10: SIGUSR1", "11: SIGSEGV", "12: SIGUSR2",
		"13: SIGPIPE", "14: SIGALRM", "15: SIGTERM", "16: SIGSTKFLT",
		"17: SIGCHLD", "18: SIGCONT", "19: SIGSTOP", "20: SIGTSTP",
		"21: SIGTTIN", "22: SIGTTOU", "23: SIGURG", "24: SIGXCPU",
		"25: SIGXFSZ", "26: SIGVTALRM", "27: SIGPROF", "28: SIGWINCH",
		"29: SIGIO", "30: SIGPWR", "31: SIGSYS", "34: SIGRTMIN",
		"35: SIGRTMIN+1", "36: SIGRTMIN+2", "37: SIGRTMIN+3", "38: SIGRTMIN+4",
		"39: SIGRTMIN+5", "40: SIGRTMIN+6", "41: SIGRTMIN+7", "42: SIGRTMIN+8",
		"43: SIGRTMIN+9", "44: SIGRTMIN+10", "45: SIGRTMIN+11", "46: SIGRTMIN+12",
		"47: SIGRTMIN+13", "48: SIGRTMIN+14", "49: SIGRTMIN+15", "50: SIGRTMAX-14",
		"51: SIGRTMAX-13", "52: SIGRTMAX-12", "53: SIGRTMAX-11", "54: SIGRTMAX-10",
		"55: SIGRTMAX-9", "56: SIGRTMAX-8", "57: SIGRTMAX-7", "58: SIGRTMAX-6",
		"59: SIGRTMAX-5", "60: SIGRTMAX-4", "61: SIGRTMAX-3", "62: SIGRTMAX-2",
		"63: SIGRTMAX-1", "64: SIGRTMAX"
	};

	void *stack_p[10];
	char **stack_info;
	int size;

	size = backtrace( stack_p, sizeof(stack_p));
	stack_info = backtrace_symbols( stack_p, size);

	if ( no >= 1 && no <= 64)

		printf("[%s] %d stack frames.\n", _signal[no - 1], size);

	else

		printf("[No infomation %d] %d stack frames.\n", no, size);

	int i = 0;
	for ( ; i < size; i++)
		printf("%s\n", stack_info[i]);

	//free( stack_info);

	//free anything
	std::this_thread::sleep_for(std::chrono::seconds(100));
	exit(0);

	//fflush(NULL);
	//exit(0);
}


/*default is BCube(3,2)*/
void rdma_bcube_init(bcube_struct& bcube_s, bcube_global_struct& bgs)
{
	//signal(SIGINT, sig_handler);
	signal(SIGSEGV, fun_dump);
	printf("in rdma bcube init...\n");
	bcube_s.bcube0_size = 3;
	bcube_s.bcube_level = 2;

	topology_init(bcube_s);
	setup_node(bcube_s);
	rdma_server_init(bcube_s);
	rdma_client_init(bcube_s);
	get_send_strategy(bcube_s);
	while (!rdma_check_bcube_is_inited_done(bcube_s))
		std::this_thread::sleep_for(std::chrono::seconds(1));
	return;
}

extern bcube_global_struct bcube_gs;



void rdma_bcube_send(tensor_table_entry& e, bcube_struct& bs, int stage)
{
	auto& send_strategy = bs.my_strategy;
	assert((size_t)stage < send_strategy.size());
	auto & steps = send_strategy[stage];

	for (auto& link_proc : steps)
	{
		msg_struct* tmp_msg = nullptr;
		for (auto& to_one_node : link_proc)
		{
			int encode_len = 0;
			tensor_msg::encode(e, (void**)&tmp_msg, to_one_node.paraid[0], to_one_node.block_num, &encode_len);
			show_msg((void*)tmp_msg);

			/*
			append to each node list here...
			*/
			node_item* nit = get_new_node();
			nit->data_ptr = (char*)tmp_msg;
			tmp_msg->rank = to_one_node.node_id;
			{
				bs.topo[0][to_one_node.node_id].send_list->next = nit;
				bs.topo[0][to_one_node.node_id].send_list = nit;
			}
			tmp_msg = nullptr;
		}
	}


}


static void unlock_insert_to_recv_queue(bcube_global_struct& bgs, received_tensor_entry& rs_e)
{

	auto& tailer = bgs.tail;
	auto new_node = new unlock_recv_tensor(rs_e);
	if (new_node == nullptr)
	{
		printf("fatal error in get unlocked recv_tensor handler\n");
		exit(-1);
	}
	static int index = 0;
	//printf("%d insert into list....\n", index++);
	tailer->next = new_node;
	tailer = new_node;

	return;
}

#endif