#include "bcube_rdma.h"
#include "bcube_comm.h"
#include "bcube_message.h"
#include "bcube_ops.h"

#if HAVE_RDMA

#include <errno.h>
#include <iostream>
#include <arpa/inet.h>
#define IS_SERVER true
#define IS_CLIENT false

extern std::atomic_bool server_establisted;
extern std::atomic_bool client_establisted;
extern void show_msg(void*);

void rc_die(const char* reason)
{
	fprintf(stderr, "%s\n", reason);
	exit(-1);
}

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

static void send_tensor(struct rdma_cm_id *id, char* buff, uint32_t len)
{
	struct context *ctx = (struct context *)id->context;
	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;
	if (!buff)throw std::runtime_error("send buf can not be empty!");


	std::memcpy(ctx->buffer, buff, len);
	delete buff;

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

static void post_receive_client(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	struct ibv_recv_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)id;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	sge.addr = (uintptr_t)ctx->msg;
	sge.length = sizeof(*ctx->msg);
	sge.lkey = ctx->msg_mr->lkey;
	TEST_NZ(ibv_post_recv(id->qp, &wr, &bad_wr));
}

static void post_receive_server(struct rdma_cm_id *id)
{
	struct ibv_recv_wr wr, *bad_wr = NULL;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)id;
	wr.sg_list = NULL;
	wr.num_sge = 0;
	TEST_NZ(ibv_post_recv(id->qp, &wr, &bad_wr));
}


static void* recv_data(struct ibv_wc* wc)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;
	void* _data = NULL;
	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)//as server only
	{
		uint32_t size = ntohl(wc->imm_data);
		struct sockaddr_in* client_addr = (struct sockaddr_in *)rdma_get_peer_addr(id);
		_data = (void*) (new char[size]);
		std::memcpy(_data, ctx->buffer, size);
		post_receive_server(id);
		ctx->msg->id = MSG_READY;
		send_message(id);
	}
	else
	{
		std::cout << "op code is " << wc->opcode << std::endl;
	}
	return _data;
}

static void* send_data(struct ibv_wc* wc, void* data)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct context *ctx = (struct context *)id->context;

	if (wc->opcode & IBV_WC_RECV)
	{
		if (ctx->msg->id == MSG_MR)
		{
			ctx->peer_addr = ctx->msg->data.mr.addr;
			ctx->peer_rkey = ctx->msg->data.mr.rkey;
			printf("received remote memory address and key\n");
			ctx->remote_idle = true;
			send_tensor(id, (char*)data, ((msg_struct *) data)->msg_length);
		}
		else if (ctx->msg->id == MSG_DONE)
		{
			printf("received DONE, disconnecting\n");
			rdma_disconnect(id);
			return NULL;
		}
		else if (ctx->msg->id == MSG_READY)
		{
			ctx->remote_idle = true;
			send_tensor(id, (char*)data, ((msg_struct *) data)->msg_length);
		}
		post_receive_client(id);
	}
	else
	{
		std::cout << "op code is " << wc->opcode << std::endl;
	}
	return NULL;
}

void rcv_poll_cq(void *tmp_id, _recv_chain* chain_header)
{
	struct ibv_cq *cq = NULL;
	struct ibv_wc wc;
	struct rdma_cm_id *id = (struct rdma_cm_id *)tmp_id;
	struct context *ctx = (struct context *)id->context;
	void *ev_ctx = NULL;

	_recv_chain* rcv_tail = chain_header;

	while (true)
	{
		TEST_NZ(ibv_get_cq_event(ctx->comp_channel, &cq, &ev_ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));

		while (ibv_poll_cq(cq, 1, &wc))
		{
			if (wc.status == IBV_WC_SUCCESS)
			{
				auto tp_node = new _recv_chain;
				tp_node->data_ptr = recv_data(&wc);
				tp_node->next = NULL;
				rcv_tail->next = tp_node;
				rcv_tail = tp_node;
			}
			else
			{
				printf("\nwc = %s\n", ibv_wc_status_str(wc.status));
				rc_die("poll_cq: status is not IBV_WC_SUCCESS");
			}
		}
	}
	return ;
}

void send_poll_cq(void * tmp_id, _recv_chain* chain_header)
{
	struct ibv_cq *cq = NULL;
	struct ibv_wc wc;
	struct rdma_cm_id *id = (struct rdma_cm_id *)tmp_id;
	struct context *ctx = (struct context *)id->context;
	void *ev_ctx = NULL;

	_recv_chain* rcv_header = chain_header;

	while (true)
	{
		TEST_NZ(ibv_get_cq_event(ctx->comp_channel, &cq, &ev_ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));
		while (!(rcv_header->next))//no data is ready... will sleep
			std::this_thread::sleep_for(std::chrono::nanoseconds(10));

		while (ibv_poll_cq(cq, 1, &wc))
		{
			if (wc.status == IBV_WC_SUCCESS)
			{
				auto tp_node = rcv_header->next;
				send_data(&wc, tp_node->data_ptr);
				delete rcv_header;
				rcv_header = tp_node;
			}
			else
			{
				printf("\nwc = %s\n", ibv_wc_status_str(wc.status));
				rc_die("poll_cq: status is not IBV_WC_SUCCESS");
			}
		}
	}
	return ;
}

struct ibv_pd * rc_get_pd(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	return ctx->pd;
}

void build_params(struct rdma_conn_param *params)
{
	memset(params, 0, sizeof(*params));

	params->initiator_depth = params->responder_resources = 1;
	params->rnr_retry_count = 7; /* infinite retry */
	params->retry_count = 7;
}

void build_context(struct rdma_cm_id *id, bool is_server, _recv_chain* chain_header)
{
	struct context *s_ctx = (struct context *)malloc(sizeof(struct context));
	s_ctx->ibv_ctx = id->verbs;
	TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ibv_ctx));
	TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ibv_ctx));
	TEST_Z(s_ctx->cq = ibv_create_cq(s_ctx->ibv_ctx, MIN_CQE, NULL, s_ctx->comp_channel, 0));
	TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));
	id->context = (void*)s_ctx;
	if (is_server)
	{
		//TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, poll_cq, id));/*create recv threads*/
		s_ctx->cq_poller_thread = std::thread(rcv_poll_cq, id, chain_header);
		id->context = (void*)s_ctx;
	}
}

void build_qp_attr(struct ibv_qp_init_attr *qp_attr, struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;
	memset(qp_attr, 0, sizeof(*qp_attr));
	qp_attr->send_cq = ctx->cq;
	qp_attr->recv_cq = ctx->cq;
	qp_attr->qp_type = IBV_QPT_RC;

	qp_attr->cap.max_send_wr = 10;
	qp_attr->cap.max_recv_wr = 10;
	qp_attr->cap.max_send_sge = 1;
	qp_attr->cap.max_recv_sge = 1;
}

void build_connection(struct rdma_cm_id *id, bool is_server, _recv_chain* chain_header)
{
	struct ibv_qp_init_attr qp_attr;
	build_context(id, is_server, chain_header);
	build_qp_attr(&qp_attr, id);

	struct context *ctx = (struct context *)id->context;
	TEST_NZ(rdma_create_qp(id, ctx->pd, &qp_attr));
}

static void on_pre_conn(struct rdma_cm_id *id, bool is_server)
{
	struct context *ctx = (struct context *)id->context;
	posix_memalign((void **)&ctx->buffer, sysconf(_SC_PAGESIZE), BUFFER_SIZE);
	TEST_Z(ctx->buffer_mr = ibv_reg_mr(rc_get_pd(id), ctx->buffer, BUFFER_SIZE,
	                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

	posix_memalign((void **)&ctx->msg, sysconf(_SC_PAGESIZE), sizeof(*ctx->msg));
	TEST_Z(ctx->msg_mr = ibv_reg_mr(rc_get_pd(id), ctx->msg, sizeof(*ctx->msg),
	                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

	if (is_server)
		post_receive_server(id);
	else
		post_receive_client(id);
}

static void on_connection(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;

	ctx->msg->id = MSG_MR;
	ctx->msg->data.mr.addr = (uintptr_t)ctx->buffer_mr->addr;
	ctx->msg->data.mr.rkey = ctx->buffer_mr->rkey;

	send_message(id);
}

static void on_disconnect(struct rdma_cm_id *id)
{
	struct context *ctx = (struct context *)id->context;

	ibv_dereg_mr(ctx->buffer_mr);
	ibv_dereg_mr(ctx->msg_mr);

	free(ctx->buffer);
	free(ctx->msg);
	free(ctx);
}


static void rdma_recv_loops(bcube_global_struct& bgs)
{
	bcube_struct& bs = bgs.bcube_s;
	struct rdma_cm_event *event = NULL;
	struct rdma_conn_param cm_params;
	int connecting_client_cnt = 0;
	int client_counts = (bs.bcube0_size - 1) * bs.bcube_level;
	printf("server is inited done (RDMA), waiting for %d client connecting....:)\n", client_counts);
	build_params(&cm_params);
	std::vector<_recv_chain*> _recv_vec;

	while (rdma_get_cm_event(bs.event_channel, &event) == 0)
	{
		struct rdma_cm_event event_copy;

		memcpy(&event_copy, event, sizeof(*event));
		rdma_ack_cm_event(event);

		if (event_copy.event == RDMA_CM_EVENT_CONNECT_REQUEST)
		{
			_recv_chain* rc_tp = new _recv_chain;
			rc_tp->data_ptr = rc_tp->next = NULL;
			build_connection(event_copy.id, IS_SERVER, rc_tp);
			_recv_vec.push_back(rc_tp);
			on_pre_conn(event_copy.id, IS_SERVER);
			TEST_NZ(rdma_accept(event_copy.id, &cm_params));
		}
		else if (event_copy.event == RDMA_CM_EVENT_ESTABLISHED)
		{
			on_connection(event_copy.id);
			bs.recv_rdma_cm_id.push_back(event_copy.id);

			struct sockaddr_in* client_addr = (struct sockaddr_in *)rdma_get_peer_addr(event_copy.id);
			printf("client[%s,%d] is connecting (RDMA) now... \n",
			       inet_ntoa(client_addr->sin_addr), client_addr->sin_port);
			connecting_client_cnt++;
			if (connecting_client_cnt == client_counts)
				break;
		}
		else if (event_copy.event == RDMA_CM_EVENT_DISCONNECTED)
		{
			rdma_destroy_qp(event_copy.id);
			on_disconnect(event_copy.id);
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

	int msg_len = sizeof(msg_struct);
	auto& fd_vect = bgs.bcube_s.recv_rdma_cm_id;
	int fd_num = fd_vect.size();
	server_establisted = true;
	msg_struct msg_buf;
	while (true)
	{
		for (auto& _rc_header : _recv_vec)
		{
			if (!_rc_header) {throw std::runtime_error("_recv_vec header should not be NULL");}
			if (!_rc_header->next) continue;
			if (!_rc_header->data_ptr) {throw std::runtime_error("_recvd data not be NULL");}
			received_tensor_entry e;
			show_msg(new_msg);
			tensor_msg::decode(e, new_msg);
			insert_to_recv_queue(bgs, e);
			auto tmp_header = _rc_header->next;
			delete _rc_header;
			_rc_header = tmp_header;
		}
	}
	return;
}

extern bcube_global_struct bcube_gs;

static void rdma_server_init(bcube_struct & bs)
{
	int init_loops = 0;
	struct sockaddr_in sin;
	printf("init a server with RDMA ....\n");
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;/*ipv4*/
	sin.sin_port = htons(bs.server_port);/*server listen public ports*/
	sin.sin_addr.s_addr = INADDR_ANY;/*listen any connects*/

	TEST_Z(bs.event_channel = rdma_create_event_channel());
	TEST_NZ(rdma_create_id(bs.event_channel, &bs.listener, NULL, RDMA_PS_TCP));

	while (rdma_bind_addr(bs.listener, (struct sockaddr *)&sin))
	{
		std::cerr << "server init failed (RDMA): error in bind socket, will try it again in 2 seconds..." << std::endl;
		if (init_loops > 30)
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

	bcube_gs.bg_thread.push_back(std::thread(rdma_recv_loops, std::ref(bcube_gs)));
	std::this_thread::sleep_for(std::chrono::seconds(1));
	return;
}
static void rdma_client_init(bcube_struct& bs)
{
	std::cout << "client with RDMA is initing" << std::endl;
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
			TEST_NZ(rdma_resolve_addr(conn, (struct sockaddr*)(&local_in),
			                          (struct sockaddr*)(&ser_in), TIMEOUT_IN_MS));

			struct rdma_cm_event *event = NULL;
			struct rdma_conn_param cm_params;

			build_params(&cm_params);

			_recv_chain* send_chain = new _recv_chain;
			send_chain->data_ptr = send_chain->next = NULL;

			while (rdma_get_cm_event(ec, &event) == 0)
			{
				struct rdma_cm_event event_copy;
				memcpy(&event_copy, event, sizeof(*event));
				rdma_ack_cm_event(event);
				if (event_copy.event == RDMA_CM_EVENT_ADDR_RESOLVED)
				{
					build_connection(event_copy.id, IS_CLIENT);
					on_pre_conn(event_copy.id, IS_CLIENT);
					TEST_NZ(rdma_resolve_route(event_copy.id, TIMEOUT_IN_MS));
				}
				else if (event_copy.event == RDMA_CM_EVENT_ROUTE_RESOLVED)
				{
					TEST_NZ(rdma_connect(event_copy.id, &cm_params));
				}
				else if (event_copy.event == RDMA_CM_EVENT_ESTABLISHED)
				{
					struct context *ctx = (struct context *)event_copy.id->context;
					ctx->cq_poller_thread = std::thread(send_poll_cq, event_copy.id, send_chain);
					//TEST_NZ(pthread_create(&ctx->cq_poller_thread, NULL, poll_cq, event_copy.id));
					std::cout << local_eth << " has connected to server[ "
					          << bs.neighbor_info[lev][index].ip << " , " << bs.server_port << " ]" << std::endl;
					break;
				}
				else if (event_copy.event == RDMA_CM_EVENT_REJECTED)
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
					connect_count++;
					struct context *ctx = (struct context *)event_copy.id->context;
					ibv_dereg_mr(ctx->buffer_mr);
					ibv_dereg_mr(ctx->msg_mr);
					free(ctx->buffer);
					free(ctx->msg);
					free(ctx);
					rdma_destroy_qp(event_copy.id);
					rdma_destroy_id(event_copy.id);
					rdma_destroy_event_channel(ec);
					if (connect_count > 10 * 300)/*after 300 seconds, it will exit.*/
					{
						std::cerr << 300 << "seconds is passed, error in connect to server "
						          << bs.neighbor_info[lev][index].ip << ", check your network condition" << std::endl;
						exit(-1);
					}
					else
					{
						TEST_Z(ec = rdma_create_event_channel());
						TEST_NZ(rdma_create_id(ec, &conn, NULL, RDMA_PS_TCP));
						TEST_NZ(rdma_resolve_addr(conn, (struct sockaddr*)(&local_in),
						                          (struct sockaddr*)(&ser_in), TIMEOUT_IN_MS));
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
			bs.topo[lev][bs.neighbor_info[lev][index].node_index].send_list = send_chain;
			bs.neighbor_info[lev][index].send_rdma_event_channel = ec;
			bs.neighbor_info[lev][index].send_rdma_cm_id = conn;
			bs.neighbor_info[lev][index].send_list = send_chain;
		}
	}
	client_establisted = true;
	std::cout << "client inited done" << std::endl;
}

bool rdma_all_init(bcube_struct& bcube_s)
{
	rdma_server_init(bcube_s);
	rdma_client_init(bcube_s);
	printf("rdma all inited done\n");
	return true;
}

bool bcube_send_by_rdma(tensor_table_entry& e, bcube_struct& bs, int stage)
{
	auto& send_strategy = bs.my_strategy;
	assert((size_t)stage < send_strategy.size());
	auto & steps = send_strategy[stage];

	for (int process_index = 0; process_index < 2; process_index++)
	{
		msg_struct* tmp_msg = nullptr;
		for (auto& it : steps[process_index])
		{
			int len = 0;
			tensor_msg::encode(a_tensor, (void**)&tmp_msg, it.paraid[0], it.block_num, &len);
			//printf("send out: %s,\t send len=%d\n",a_tensor.tensor_name.c_str(),len);
			show_msg((void*)tmp_msg);

			auto send_node_data = new _recv_chain;
			send_node_data->data_ptr = (void*)tmp_msg;
			send_node_data->next = nullptr;
			it.send_list->next = send_node_data;
			tmp_msg = nullptr;
		}
	}
	return true;
}
//#endif //RDMA
#endif