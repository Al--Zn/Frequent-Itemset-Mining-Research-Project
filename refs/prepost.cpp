#include <iostream>
#include <time.h>
#include <windows.h>
#include <psapi.h>
#pragma comment(lib,"psapi.lib")

using namespace std;

int **bf;
int bf_cursor;
int bf_size;
int bf_col;
int bf_currentSize;

int numOfFItem;			// the total number of frequent items
int Min_Support;		// the minimum support provided

struct Item
{
	int index;
	int num;
};

Item *item;				// the array to store all the items and their support

struct NodeListTreeNode
{
	int label;
	NodeListTreeNode* firstChild;
	NodeListTreeNode* next;
	int support;
	int NLStartinBf;
	int NLLength;
	int NLCol;
};

struct PPCTreeNode
{
	int label;
	PPCTreeNode* firstChild;
	PPCTreeNode* rightSibling;
	PPCTreeNode* labelSibling;
	PPCTreeNode* father;
	int count;
	int foreIndex;
	int backIndex;
};

FILE *out;
int dump;
int *result;
int resultLen = 0;
int resultCount = 0;
long nlLenSum = 0;

int comp(const void *a,const void *b)
{
	return (*(Item *)b).num - (*(Item *)a).num;
}

void showMemoryInfo(void)
{
    HANDLE handle=GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(handle,&pmc,sizeof(pmc));
    cout<<pmc.PeakWorkingSetSize/1000<<endl;
}


void getData(FILE * in, char *filename, double support)
{
	if((in = fopen(filename,"r")) == NULL)
	{
		cout<<"read wrong!"<<endl;
		fclose(in);
		exit(1);
	}

	char str[500];
	Item **tempItem = new Item*[10];
	tempItem[0] = new Item[10000];	
	for(int i = 0; i < 10000; i++)
	{
		tempItem[0][i].index = i;
		tempItem[0][i].num = 0;
	}
	
	int numOfTrans = 0;		// total number of transactions
	int size = 1;			// the current upper bound of col
	int numOfItem = 0;		// total number of items
	int num = 0;			// the current number in the transaction
	int col = 0;			// equals to num / 10000, used to store the support of num into 2-D array

	// To save memory space in a deceptive way
	// the support of num is stored in a dynamically allocated 2-d array tempItem[num/10000][num%10000]
	// which is redundantly intricate and obscure 
	while(fgets(str,500,in))
	{
		if(feof(in)) break;
		numOfTrans++;
		num = 0;
		for(int i = 0; i < 500 && str[i] != '\0'; i++)
		{			
			if(str[i] != ' ')	num = num * 10 + str[i] - '0';			
			else
			{				
				col = num / 10000;
				if(col >= size)				
				{
					for(int j = size; j <= col; j++)
					{
						tempItem[j] = new Item[10000];
						for(int p = 0; p < 10000; p++)
						{
							tempItem[j][p].index = j*10000+p;
							tempItem[j][p].num = 0;
						}
					}
					size = col + 1;
				}
				if(0 == tempItem[col][num % 10000].num++) numOfItem++;
				num = 0;
			}
		}	
	}
	fclose(in);
	// now the read process is finished
	// we collect the items into a 1-D array
	Min_Support = int(support * numOfTrans);
	item = new Item[numOfItem];	
	for(int i = 0, p = 0; i < size; i++)
		for(int j = 0;j < 10000; j++)
			if(tempItem[i][j].num != 0)
				item[p++] = tempItem[i][j];
	// and delete the 2-D array we've created to SAVE THE HUAKING MEMORY SPACE
	for(int i = 0; i < size; i++) 
		delete[] tempItem[i];
	delete[] tempItem;
	// finally we sort the item array in descending order of the support
	qsort(item, numOfItem, sizeof(Item), comp);
	for(numOfFItem = 0; numOfFItem < numOfItem; numOfFItem++)
		if(item[numOfFItem].num < Min_Support) 
			break;
}

PPCTreeNode ppcRoot;		// the root of the fp tree (PPCTree in the paper)
NodeListTreeNode nlRoot;
PPCTreeNode **headTable;
int *headTableLen;
int *itemsetCount;
int *sameItems;
int nlNodeCount = 0;

void buildTree(FILE *in, char * filename)
{
	// To build the fp tree (PPCTree), we need another access to the dataset
	if((in = fopen(filename,"r")) == NULL)
	{
		cout<<"read wrong!"<<endl;
		fclose(in);
		exit(1);
	}

	ppcRoot.label = -1;
	char str[500];
	Item transaction[1000];
	// clean the transaction
	for(int i = 0; i < 1000; i++)
	{
		transaction[i].index = 0;
		transaction[i].num = 0;
	}
	
	int num = 0, tLen = 0;
	while(fgets(str,500,in))
	{		
		if(feof(in))
			break;
		num = 0; tLen = 0;
		for(int i = 0; i < 500 && str[i] != '\0'; i++)
		{			
			if(str[i] != ' ')	num = num * 10 + str[i] - '0';			
			else
			{
				// traverse the item array to find the number
				// transaction[tLen].index stores the number
				// transaction[tLen].num stores the negative of the position of the number in the item array
				// 		i.e. the smaller transaction[tLen].num is, the smaller its support is
				for(int j = 0; j < numOfFItem; j++)
				{
					if(num == item[j].index)
					{
						transaction[tLen].index = num;
						transaction[tLen].num = 0 - j;					
						tLen++;					
						break;
					}
				}
				num = 0;
			}
		}
		// sort the transaction in the descending order of the item support
		qsort(transaction, tLen, sizeof(Item), comp);
		int curPos = 0;
		PPCTreeNode *curRoot =&(ppcRoot);
		PPCTreeNode *rightSibling = NULL;
		// insert the item into the fp tree (PPCTree)
		while(curPos != tLen)
		{
			PPCTreeNode *child = curRoot->firstChild;
			while(child != NULL)
			{				
				if(child -> label == 0 - transaction[curPos].num)
				{					
					curPos++;
					child->count++;
					curRoot = child;
					break;
				}
				if(child -> rightSibling == NULL)
				{
					rightSibling = child;
					child = NULL;
					break;
				}
				child = child -> rightSibling;
			}			
			if(child == NULL) break;
		}
		for(int j = curPos; j < tLen; j++)
		{
			PPCTreeNode *ppcNode = new PPCTreeNode;
			ppcNode->label = 0 - transaction[j].num;
			if(rightSibling != NULL)
			{
				rightSibling->rightSibling = ppcNode;
				rightSibling = NULL;
			}
			else
			{
				curRoot->firstChild = ppcNode;	
			}
			ppcNode->rightSibling = NULL;
			ppcNode->firstChild = NULL;
			ppcNode->father = curRoot;
			ppcNode->labelSibling = NULL;
			ppcNode->count = 1;
			curRoot = ppcNode;
		}
	}
	fclose(in);
	
	headTable = new PPCTreeNode*[numOfFItem];
	memset(headTable, 0, sizeof(int*) * numOfFItem);
	headTableLen = new int[numOfFItem];
	memset(headTableLen, 0, sizeof(int) * numOfFItem);
	PPCTreeNode **tempHead = new PPCTreeNode*[numOfFItem];
	
	itemsetCount = new int[(numOfFItem-1) * numOfFItem / 2];
	memset(itemsetCount, 0, sizeof(int) * (numOfFItem-1) * numOfFItem / 2);
	
	PPCTreeNode *root = ppcRoot.firstChild;
	int pre = 0, last = 0;
	while(root != NULL)
	{
		root->foreIndex = pre;
		pre++;

		if(headTable[root->label] == NULL)
		{	
			headTable[root->label] = root;
			tempHead[root->label] = root;
		}
		else
		{
			tempHead[root->label]->labelSibling = root;
			tempHead[root->label] = root;		
		}
		headTableLen[root->label]++;

		PPCTreeNode *temp = root->father;
		// count the support of the 2nd frequent itemset (root->num, temp->num) from the leaf to the root
		while(temp->label != -1)
		{
			// temp is root's father, so its label must smaller than root's label
			// itemsetCount[root->label * (root->label - 1) / 2 + temp->label] denotes the support of the 2nd frequent itemset (root->num, temp->num)
			itemsetCount[root->label * (root->label - 1) / 2 + temp->label] += root->count;
			temp = temp->father;
		}
		if(root->firstChild != NULL)
			root = root->firstChild;
		else
		{
			//backvist
			root->backIndex=last;
			last++;
			if(root->rightSibling != NULL)
				root = root->rightSibling;
			else
			{
				root = root->father;
				while(root != NULL)
				{	
					//backvisit
					root->backIndex=last;
					last++;
					if(root->rightSibling != NULL)
					{
						root = root->rightSibling;
						break;
					}
					root = root->father;
				}
			}
		}
	}
	delete[] tempHead;
}

void initializeTree()
{
	NodeListTreeNode *lastChild = NULL;
	for(int t = numOfFItem - 1; t >= 0; t--)
	{
		if(bf_cursor > bf_currentSize - headTableLen[t] * 3)
		{
			bf_col++;
			bf_cursor = 0;
			bf_currentSize = 10 * bf_size;
			bf[bf_col] = new int[bf_currentSize];
		}

		NodeListTreeNode *nlNode = new NodeListTreeNode;
		nlNode->label = t;
		nlNode->support = 0;
		nlNode->NLStartinBf = bf_cursor;
		nlNode->NLLength = 0;
		nlNode->NLCol = bf_col;
		nlNode->firstChild = NULL;
		nlNode->next = NULL;
		PPCTreeNode *ni = headTable[t];
		while(ni != NULL)
		{
			nlNode->support+= ni->count;
			bf[bf_col][bf_cursor++] =  ni->foreIndex;
			bf[bf_col][bf_cursor++] =  ni->backIndex;
			bf[bf_col][bf_cursor++] =  ni->count;
			nlNode->NLLength++;
			ni = ni->labelSibling;
		}
		if(nlRoot.firstChild == NULL)
		{
			nlRoot.firstChild = nlNode;
			lastChild = nlNode;
		}
		else
		{
			lastChild->next = nlNode;
			lastChild = nlNode;
		}
	}
}

NodeListTreeNode *isk_itemSetFreq(NodeListTreeNode* ni, NodeListTreeNode* nj, int level, NodeListTreeNode *lastChild, int &sameCount)
{
	if(bf_cursor + ni->NLLength * 3 > bf_currentSize)
	{
		bf_col++;
		bf_cursor = 0;
		bf_currentSize = bf_size > ni->NLLength * 1000 ? bf_size : ni->NLLength * 1000;
		bf[bf_col] = new int[bf_currentSize];
	}
		
	NodeListTreeNode *nlNode = new NodeListTreeNode;
	nlNode->support = 0;
	nlNode->NLStartinBf = bf_cursor;
	nlNode->NLCol = bf_col;
	nlNode->NLLength = 0;
	
	int cursor_i = ni->NLStartinBf;
	int cursor_j = nj->NLStartinBf;
	int col_i = ni->NLCol;
	int col_j = nj->NLCol;
	int last_cur = -1;
	while(cursor_i < ni->NLStartinBf + ni->NLLength * 3 && cursor_j < nj->NLStartinBf + nj->NLLength * 3)
	{
		if(bf[col_i][cursor_i] > bf[col_j][cursor_j] && bf[col_i][cursor_i + 1] < bf[col_j][cursor_j + 1])
		{
			if(last_cur == cursor_j)
			{
				bf[bf_col][bf_cursor - 1] += bf[col_i][cursor_i + 2];
			}
			else
			{
				bf[bf_col][bf_cursor++] =  bf[col_j][cursor_j];
				bf[bf_col][bf_cursor++] =  bf[col_j][cursor_j + 1];
				bf[bf_col][bf_cursor++] =  bf[col_i][cursor_i + 2];
				nlNode->NLLength++;
			}
			nlNode->support += bf[col_i][cursor_i + 2];
			last_cur = cursor_j;
			cursor_i += 3;
		}
		else if(bf[col_i][cursor_i] < bf[col_j][cursor_j])
		{
			cursor_i += 3;
		}
		else if(bf[col_i][cursor_i + 1] > bf[col_j][cursor_j + 1])
		{
			cursor_j += 3;
		}
	}
	if(nlNode->support >= Min_Support)
	{
		if(ni->support == nlNode->support && nlNode->NLLength == 1)
		{
			sameItems[sameCount++] = nj->label;
			bf_cursor = nlNode->NLStartinBf;
			delete nlNode;
		}
		else
		{
			nlNode->label = nj->label;
			nlNode->firstChild = NULL;
			nlNode->next = NULL;
			if(ni->firstChild == NULL)
			{
				ni->firstChild = nlNode;
				lastChild = nlNode;
			}
			else
			{
				lastChild->next = nlNode;
				lastChild = nlNode;
			}
		}
		return lastChild;
	}
	else
	{
		bf_cursor = nlNode->NLStartinBf;
		delete nlNode;
	}
	return lastChild;
}

void traverse(NodeListTreeNode *curNode, NodeListTreeNode *curRoot, int level, int sameCount)
{
	NodeListTreeNode *sibling = curNode->next;
	NodeListTreeNode *lastChild = NULL;
	while(sibling != NULL)
	{	
		if(level >1 || (level == 1 && itemsetCount[(curNode->label-1) * curNode->label/2 + sibling->label] >= Min_Support))
		lastChild = isk_itemSetFreq(curNode, sibling, level, lastChild, sameCount);
		sibling = sibling->next;
	}
	
	resultCount += pow(2.0, sameCount);
	nlLenSum += pow(2.0, sameCount) * curNode->NLLength;

	if(dump == 1)
	{
		result[resultLen++] = curNode->label;
		for(int i = 0; i < resultLen; i++)
			fprintf(out, "%d ", item[result[i]].index);
		fprintf(out, "(%d %d)", curNode->support, curNode->NLLength);
		for(int i = 0; i < sameCount; i++)
			fprintf(out, " %d", item[sameItems[i]].index);
		fprintf(out, "\n");
	}
	nlNodeCount++;
	
	int from_cursor = bf_cursor;
	int from_col = bf_col;
	int from_size = bf_currentSize;
	NodeListTreeNode *child = curNode->firstChild;
	NodeListTreeNode *next = NULL;
	while(child != NULL)
	{
		next = child->next;
		traverse(child, curNode, level+1, sameCount);
		for(int c = bf_col; c > from_col; c--)
			delete[] bf[c];
		bf_col = from_col;
		bf_cursor = from_cursor;
		bf_currentSize = from_size;
		child = next;
	}
	if(dump == 1)
		resultLen--;
	delete curNode;
}

void deleteFPTree()
{
	ppcRoot.father = NULL;
	ppcRoot.rightSibling = NULL;
	PPCTreeNode *root = ppcRoot.firstChild;
	PPCTreeNode *next = NULL;
	while(root != NULL)
	{
		if(root->firstChild != NULL)
			root = root->firstChild;
		else
		{
			if(root->rightSibling != NULL)
			{
				next = root->rightSibling;
				delete root;
				root = next;
			}
			else
			{
				next = root->father;
				delete root;
				root = next;
				while(root != NULL)
				{	
					if(root->rightSibling != NULL)
					{
						next = root->rightSibling;
						delete root;
						root = next;
						break;
					}
					next = root->father;
					delete root;
					root = next;
				}
			}
		}
	}
}

void deleteNLTree(NodeListTreeNode *root)
{
	NodeListTreeNode *cur = root->firstChild;;
	NodeListTreeNode *next = NULL;
	while(cur != NULL)
	{
		next = cur->next;
		deleteNLTree(cur);
		cur = next;
	}
	delete root;
}

void run(FILE *in, char* filename)
{
	// if the dump flag is set, create a file to save the result
	if(1 == dump)
	{
		out = fopen("sdp.txt","wt");
		result = new int[numOfFItem];
		resultLen = 0;
	}
	// firstly we have to build the fp tree
	buildTree(in, filename);
	//deleteFPTree();

	nlRoot.label = numOfFItem;
	nlRoot.firstChild = NULL;
	nlRoot.next = NULL;
	
	initializeTree();
	sameItems = new int[numOfFItem];

	int from_cursor = bf_cursor;
	int from_col = bf_col;
	int from_size = bf_currentSize;
	
	NodeListTreeNode *curNode = nlRoot.firstChild;
	NodeListTreeNode *next = NULL;
	while(curNode != NULL)
	{
		next = curNode->next;
		traverse(curNode, &nlRoot, 1, 0);
		for(int c = bf_col; c > from_col; c--)
			delete[] bf[c];
		bf_col = from_col;
		bf_cursor = from_cursor;
		bf_currentSize = from_size;
		curNode = next;
	}
	
	delete[] sameItems;
	printf("%d %d %.2f ", nlNodeCount, resultCount, nlLenSum /((float)resultCount));
	if(1 == dump)
		fclose(out);
	
}

int main(int argc, char **argv)
{
	if (argc < 2)
	{
	  cout << "usage: fmi <DATAFILE> <MINSUP(0~1)> <ISOUT>\n";
	  exit(1);
	}

	char *filename = argv[1];
	double THRESHOLD = atof(argv[2]);
	dump = atoi(argv[3]);
	FILE *in = NULL;
	bf_size = 1000000;
	bf = new int*[100000];
	bf_currentSize = bf_size * 10;
	bf[0] = new int[bf_currentSize];
	
	bf_cursor = 0;
	bf_col = 0;
	clock_t start, end;
	
	//Read Dataset
	start =clock();
	getData(in, filename, THRESHOLD);
	run(in, filename);
	end = clock();
	printf("%.3f ", (double)(end - start) / CLOCKS_PER_SEC);
	showMemoryInfo();
	return 0;
}