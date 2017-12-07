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
	int block_num;/*发送几个block*/
	int block_size;/*每个block的大小*/
	std::vector<int> paraid;
}send_to_one;

/*
D1:node index;
D2:stage index;
D3：process index;
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

	std::vector<int> recv_fd;/*接收套接字*/
	std::vector<std::vector<node>> topo,neighbor_info;

	node local_info;/*本地的服务端接收套接字,初始化完成后变不再使用*/

	std::vector<step> nodes_send_strategy;/*全部的发送步骤*/
	std::vector<process> my_strategy;/*当前节点的发送策略*/
};
#include "bcube_message.h"
void bcube_init(bcube_struct&, bcube_global_struct&);
void bcube_send(tensor_table_entry& , bcube_struct& , int );
void bcube_test(void);
#endif
