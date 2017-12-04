#ifndef UTILS_H
#define UTILS_H
#include <cstdio>

#define MAX_N 1000
using namespace std;



class Utils
{
public:
	Utils();
	static int Convert(int temp[], int len, int k);
	static void Convert2Array(int val, int k, int*ans);
	static void getOneHopNeighbour(int serv_id, int l, int n, int k, int*one_hop_neighours);
	static void getScatterMatrix(int p, int s, int n, int k, int N, int**sendMatrix, int&para_num);
	static void getGatherMatrix(int p, int s, int n, int k, int N, int**sendMatrix, int&para_num);
	~Utils();
private:

};
#endif
