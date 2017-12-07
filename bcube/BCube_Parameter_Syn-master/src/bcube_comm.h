#ifndef __TENSOTFLOW_BCBUE__
#define __TENSOTFLOW_BCBUE__

#include <vector>
#include <string>
struct bcube_global_struct;
struct node
{
	int remote_fd;
	int node_index;
	std::string ip;
	/*below is for server*/
	std::vector<std::string> myip;

};

typedef struct
{
	int socket_fd;
	int node_id;
	int block_num;/*���ͼ���block*/
	int block_size;/*ÿ��block�Ĵ�С*/
	std::vector<int> paraid;
}send_to_one;

/*
D1:node index;
D2:stage index;
D3��process index;
*/
typedef std::vector<send_to_one> send_strategy;
typedef std::vector<send_strategy> process;
typedef std::vector<process> step;


struct bcube_struct
{
	int bcube0_size;/*node count in bcube0*/
	int bcube_level;/*level count in bcube*/
	int bcube_node_count;

	int rank;

	int server_fd;/*server listen fd*/
	int server_port=9610;/*public port*/

	std::vector<int> recv_fd;/*�����׽���*/
	std::vector<std::vector<node>> topo,neighbor_info;

	node local_info;/*���صķ���˽����׽���,��ʼ����ɺ�䲻��ʹ��*/

	std::vector<step> nodes_send_strategy;/*ȫ���ķ��Ͳ���*/
	std::vector<process> my_strategy;/*��ǰ�ڵ�ķ��Ͳ���*/
};
#include "bcube_message.h"
void bcube_init(bcube_struct&, bcube_global_struct&);
void bcube_send(tensor_table_entry& , bcube_struct& , int );
void bcube_test(void);
#endif
