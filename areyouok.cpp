#include <iostream>
#include <string>
#include <vector>

bool question(std::string ques)
{
	std::string u_ans;
	std::cout << ques << std::endl;
	std::cin >> u_ans;
	if (u_ans != "yes")
		return true;
	return false;
}

void enqueue_ques(std::vector<std::string>& ques_pool)
{
	ques_pool.push_back("How are you");
	ques_pool.push_back("Are you OK?");
	ques_pool.push_back("Do you like Mi4i?");
	ques_pool.push_back("I'm very happy to be in China (emm, in Indina)..., do you like me?");
	ques_pool.push_back("I will give you everybody a gift, a Mi band, are you happy?");
	ques_pool.push_back("Are you OK?");
}

int main(int argc, char** argv)
{
	bool res;
	std::vector<std::string> ques_pool;

	enqueue_ques(ques_pool);

	for (auto& it : ques_pool)
	{
		if (question(it))goto exit_fina;
	}

	std::cout << "You are a good man..., Bye." << std::endl;
exit_fina:
	std::cout << "Your answer is not my expected..., leaving you now!" << std::endl;
	return -1;
}
