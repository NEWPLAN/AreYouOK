#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "common.h"
#include "messages.h"

#define MAX_FILE_NAME 256
static float limited2zero = 0.00000001f;

struct conn_context
{
	char *buffer;
	struct ibv_mr *buffer_mr;

	struct message *msg;
	struct ibv_mr *msg_mr;

	FILE* fd;
	char file_name[MAX_FILE_NAME];
};

static void send_message(struct rdma_cm_id *id)
{
	struct conn_context *ctx = (struct conn_context *)id->context;

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

static void post_receive(struct rdma_cm_id *id)
{
	struct ibv_recv_wr wr, *bad_wr = NULL;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)id;
	wr.sg_list = NULL;
	wr.num_sge = 0;

	TEST_NZ(ibv_post_recv(id->qp, &wr, &bad_wr));
}

static void on_pre_conn(struct rdma_cm_id *id)
{
	struct conn_context *ctx = (struct conn_context *)malloc(sizeof(struct conn_context));

	id->context = ctx;

	ctx->file_name[0] = '\0'; // take this to mean we don't have the file name

	posix_memalign((void **)&ctx->buffer, sysconf(_SC_PAGESIZE), BUFFER_SIZE);
	TEST_Z(ctx->buffer_mr = ibv_reg_mr(rc_get_pd(), ctx->buffer, BUFFER_SIZE,
	                                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

	posix_memalign((void **)&ctx->msg, sysconf(_SC_PAGESIZE), sizeof(*ctx->msg));
	TEST_Z(ctx->msg_mr = ibv_reg_mr(rc_get_pd(), ctx->msg,
	                                sizeof(*ctx->msg), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));

	post_receive(id);
}

static void on_connection(struct rdma_cm_id *id)
{
	struct conn_context *ctx = (struct conn_context *)id->context;

	ctx->msg->id = MSG_MR;
	ctx->msg->data.mr.addr = (uintptr_t)ctx->buffer_mr->addr;
	ctx->msg->data.mr.rkey = ctx->buffer_mr->rkey;

	send_message(id);
}
int first_statistic = 1;
static void on_completion(struct ibv_wc *wc)
{
	struct rdma_cm_id *id = (struct rdma_cm_id *)(uintptr_t)wc->wr_id;
	struct conn_context *ctx = (struct conn_context *)id->context;

	if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
	{
		uint32_t size = ntohl(wc->imm_data);

		if (size == 0)
		{
			ctx->msg->id = MSG_DONE;
			send_message(id);

			// don't need post_receive() since we're done with this connection

		}
		else if (ctx->file_name[0])
		{
			//ssize_t ret;
			static int64_t loopppp = 0;
			if (!(loopppp % 9))
			{
				static time_t start_t, end_t;
				if (first_statistic)
				{
					first_statistic = 0;
					start_t = time(NULL);
				}

				//start_t = end_t;
				end_t = time(NULL);

				printf("Received msg info (rates = %f Gbps): \n",
				       (size * (loopppp + 1) * 8) / 1024.0 / 1024 / 1024 / (difftime(end_t, start_t) + limited2zero));

				printf("%ld bits, %ld Bytes, i.e. %f KB, i.e. %f MB, i.e. %f GB, i.e. %f TB.\n",
				       size * (loopppp + 1) * 8, // bits
				       size * (loopppp + 1), //Bytes
				       size * (loopppp + 1) / 1024.0,  //killo Bytes
				       size * (loopppp + 1) / 1024.0 / 1024.0, //mega Bytes
				       size * (loopppp + 1) / 1024.0 / 1024 / 1024, //giga bytes
				       size * (loopppp + 1) / 1024.0 / 1024 / 1024 / 1024 //tela byte
				      );
				if (size < 100)
					ctx->buffer[size] = 0;
				ctx->buffer[100] = 0;
				printf("recv: %s\n", ctx->buffer);
			}
			loopppp++;
			/*
			ret = write(ctx->fd, ctx->buffer, size);
			if (ret != size)rc_die("write() failed");
			*/
			post_receive(id);
			ctx->msg->id = MSG_READY;
			send_message(id);

		}
		else
		{
			memcpy(ctx->file_name, ctx->buffer, (size > MAX_FILE_NAME) ? MAX_FILE_NAME : size);
			ctx->file_name[size - 1] = '\0';

			printf("opening file %s\n", ctx->file_name);
			if (strncmp(ctx->file_name, "empty.file", strlen("empty.file")))
			{
				ctx->fd = fopen(ctx->file_name, "ab");
				if (ctx->fd == NULL) rc_die("open() failed");
			}
			else
			{
				ctx->fd = NULL;
			}

			post_receive(id);
			ctx->msg->id = MSG_READY;
			send_message(id);
		}
	}
}

static void on_disconnect(struct rdma_cm_id *id)
{
	struct conn_context *ctx = (struct conn_context *)id->context;

	fclose(ctx->fd);

	ibv_dereg_mr(ctx->buffer_mr);
	ibv_dereg_mr(ctx->msg_mr);

	free(ctx->buffer);
	free(ctx->msg);

	printf("finished transferring %s\n", ctx->file_name);

	free(ctx);
}

int main(int argc, char **argv)
{
	rc_init(
	    on_pre_conn,
	    on_connection,
	    on_completion,
	    on_disconnect
	);

	printf("waiting for connections. interrupt (^C) to exit.\n");

	rc_server_loop(DEFAULT_PORT);

	return 0;
}
