#include <fcntl.h>
#include <libgen.h>
#include <stdio.h>
#include "common.h"
#include "messages.h"
#include <time.h>
#include <assert.h>


struct client_context
{
	char *buffer;
	struct ibv_mr *buffer_mr;

	struct message *msg;
	struct ibv_mr *msg_mr;

	uint64_t peer_addr;
	uint32_t peer_rkey;

	FILE* fd;
	const char *file_name;
};

static void write_remote(struct rdma_cm_id *id, uint32_t len)
{
	struct client_context *ctx = (struct client_context *)id->context;

	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

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

static void post_receive(struct rdma_cm_id *id)
{
	struct client_context *ctx = (struct client_context *)id->context;

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

static int fisrt_init = 0;
static char* __send_str = NULL;

static char* data_gene(int size)
{
	char* _data = (char*)malloc(size * sizeof(char) + 1);
	_data[size] = 0;
	char padding[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
	for (int index = 0; index < size; index++)
		_data[index] = padding[index % 10];
	return _data;
}

#include <time.h>

static void send_next_chunk(struct rdma_cm_id *id)
{
	struct client_context *ctx = (struct client_context *)id->context;

	//size_t size = 0;
	size_t size = BUFFER_SIZE - 1;
	if (!fisrt_init)
	{
		srand((int)time(NULL));
		__send_str = data_gene(size);
		memcpy(ctx->buffer, __send_str, size);
		fisrt_init++;
	}

	memcpy(ctx->buffer, __send_str, size);
	ctx->buffer[10] = (rand() % ('z' - 'a' + 1)) + 'a';
	ctx->buffer[11] = (rand() % ('z' - 'a' + 1)) + 'a';
	ctx->buffer[12] = (rand() % ('z' - 'a' + 1)) + 'a';
	ctx->buffer[13] = (rand() % ('z' - 'a' + 1)) + 'a';
	ctx->buffer[14] = (rand() % ('z' - 'a' + 1)) + 'a';
	/*if (fisrt_init <= 100)
	{
	  //strcpy(ctx->buffer,__send_str);
	  //memcpy(ctx->buffer,__send_str,size);
	  fisrt_init++;
	}
	else
	{
	  size = 0;
	  memcpy(ctx->buffer, "\0", size);
	}
	*/
	//size = read(ctx->fd, ctx->buffer, BUFFER_SIZE);

	if (size == -1)
		rc_die("read() failed\n");

	write_remote(id, size);
}

static void send_file_name(struct rdma_cm_id *id)
{
	struct client_context *ctx = (struct client_context *)id->context;

	strcpy(ctx->buffer, ctx->file_name);

	write_remote(id, strlen(ctx->file_name) + 1);
}

static void on_pre_conn(struct rdma_cm_id *id)
{
	struct client_context *ctx = (struct client_context *)id->context;

	posix_memalign((void **)&ctx->buffer, sysconf(_SC_PAGESIZE), BUFFER_SIZE);
	TEST_Z(ctx->buffer_mr = ibv_reg_mr(rc_get_pd(), ctx->buffer, BUFFER_SIZE, IBV_ACCESS_LOCAL_WRITE));

	posix_memalign((void **)&ctx->msg, sysconf(_SC_PAGESIZE), sizeof(*ctx->msg));
	TEST_Z(ctx->msg_mr = ibv_reg_mr(rc_get_pd(), ctx->msg,
	                                sizeof(*ctx->msg), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

	post_receive(id);
}
static float limited2zero = 0.00000001f;
int first_statistic = 1;
static void on_completion(struct ibv_wc *wc)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)(wc->wr_id);
	struct client_context *ctx = (struct client_context *)id->context;

	if (wc->opcode & IBV_WC_RECV)
	{
		if (ctx->msg->id == MSG_MR)
		{
			ctx->peer_addr = ctx->msg->data.mr.addr;
			ctx->peer_rkey = ctx->msg->data.mr.rkey;

			printf("received MR, sending file name\n");
			send_file_name(id);
		}
		else if (ctx->msg->id == MSG_READY)
		{
			static int counts = 0;
			//if (counts % 5000 == 0)
			//printf("received READY, sending chunk\n");
			counts++;
			if (counts < 2 * 1024 * 1000)
			{
				send_next_chunk(id);
				if (counts % 9 == 0)
				{
					static time_t start_t, end_t;

					if (first_statistic)
					{
						first_statistic = 0;
						start_t = time(NULL);
					}

					//start_t = end_t;
					end_t = time(NULL);

					printf("send info (rates = %f Gbps): \n",
					       (BUFFER_SIZE * (counts + 1) * 8) / 1024.0 / 1024 / 1024 / (difftime(end_t, start_t) + limited2zero));

					printf("%ld bits, %ld Bytes, i.e. %f KB, i.e. %f MB, i.e. %f GB, i.e. %f TB.\n",
					       BUFFER_SIZE * (counts + 1) * 8,                             // bits
					       BUFFER_SIZE * (counts + 1),                                 //Bytes
					       BUFFER_SIZE * (counts + 1) / 1024.0,                        //killo Bytes
					       BUFFER_SIZE * (counts + 1) / 1024.0 / 1024.0,               //mega Bytes
					       BUFFER_SIZE * (counts + 1) / 1024.0 / 1024 / 1024,          //giga bytes
					       BUFFER_SIZE * (counts + 1) / 1024.0 / 1024 / 1024 / 1024    //tela byte
					      );
				}
			}
			else
			{
				printf("shuai is done!\n");
				rc_disconnect(id);
				return ;
			}
		}
		else if (ctx->msg->id == MSG_DONE)
		{
			printf("received DONE, disconnecting\n");
			rc_disconnect(id);
			return;
		}

		post_receive(id);
	}
}

int main(int argc, char **argv)
{
	struct client_context ctx;

	if (argc == 3)
	{
		printf("make sure you are going to transfer a file to the remote?\n");
		fprintf(stdout, "usage: %s <server-address> <file-name>\n", argv[0]);
		ctx.file_name = basename(argv[2]);
		ctx.fd = fopen(argv[2], "ab");

		if (ctx.fd == NULL)
		{
			fprintf(stderr, "unable to open input file \"%s\"\n", ctx.file_name);
			return 1;
		}
	}
	else
	{
		assert(argc < 3);
		ctx.file_name = "empty.file";
		ctx.fd = NULL;

	}
	rc_init(
	    on_pre_conn,
	    NULL, // on connect
	    on_completion,
	    NULL// on disconnect
	);

	rc_client_loop(argv[1], DEFAULT_PORT, &ctx);

	fclose(ctx.fd);

	return 0;
}

