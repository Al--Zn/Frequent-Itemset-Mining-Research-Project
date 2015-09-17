#include <iostream>
#include <cstdio>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>

#define MAX_CHAR_PER_LINE 500
#define MAX_ITEM_PER_TRANS 1000
#define MAX_DEPTH 16
#define MAX_THREAD 1024

using namespace std;

struct Item {
	int index;
	int num;
};

Item *items;

struct NodeListTreeNode {
	int label;
	NodeListTreeNode *firstChild;
	NodeListTreeNode *next;
	int support;
	// TODO
};

struct PPCTreeNode {
	int label;
	PPCTreeNode *firstChild;
	PPCTreeNode *rightSibling;
	PPCTreeNode *labelSibling;
	PPCTreeNode *father;
	int count;
	int foreIndex;
	int backIndex;
};

struct NList {

	int label[MAX_DEPTH];
	int support;	
	int len;		// len of NList
	int idx;
};

int numOfTotalItems = 0;
int numOfFreqItems = 0;
int numOfTrans = 0;
int minSupport;
double supportRate;

char *fileName;
map<int, int> itemCntMap;
map<int, int> itemIdxMap;

PPCTreeNode ppcRoot;
NodeListTreeNode nlRoot;
PPCTreeNode **headTable;
int *headTableLen;
int *itemsetCount;

int hCurNListNum;
int *dCurNListNum;
int hCurMaxLen;
int *dCurMaxLen;
struct NList *hCurNLists;
struct NList *dCurNLists;
struct NList *dNextNLists;
int *hPres;
int *hPosts;
int *hCounts;
int *dCurPres;
int *dCurPosts;
int *dCurCounts;
int *dNextPres;
int *dNextPosts;
int *dNextCounts;

int totalNListLen = 0;



bool comp(Item a, Item b) {
	return b.num < a.num;
}
void readFile() {
	FILE *in;
	char str[MAX_CHAR_PER_LINE];
	int num;

	if ((in = fopen(fileName, "r")) == NULL) {
		printf("read wrong\n");
		exit(1);
	}

	while (fgets(str, MAX_CHAR_PER_LINE, in)) {

		numOfTrans++;
		num = 0;
		for (int i = 0; i < MAX_CHAR_PER_LINE && str[i] != '\0'; ++i) {
			if (str[i] != ' ' && str[i] != '\n')
				num = num * 10 + str[i] - '0';
			else {
				if (0 == itemCntMap[num]++)
					numOfTotalItems++;
				num = 0;
			}
		}
	}
	fclose(in);
	minSupport = ceil(supportRate * numOfTrans);
	items = (Item *) malloc(sizeof(Item) * numOfTotalItems);
	for (map<int, int>::iterator it = itemCntMap.begin(); it != itemCntMap.end(); ++it) {
		if (it->second >= minSupport) {
			items[numOfFreqItems].index = it->first;
			items[numOfFreqItems++].num = it->second;
		}
	}
	sort(items, items + numOfFreqItems, comp);
	for (int i = 0; i < numOfFreqItems; ++i) {
		itemIdxMap[items[i].index] = i;
	}
}

void initNList() {
	int n = 0;
	int curIdx = 0;
	hCurNLists = new NList[numOfFreqItems];
	hPres = new int[totalNListLen];
	hPosts = new int[totalNListLen];
	hCounts = new int[totalNListLen];

	for (int i = numOfFreqItems - 1; i >= 0; i--) {
		PPCTreeNode *curNode = headTable[i];
		hCurNLists[n].label[0] = items[i].index;
		hCurNLists[n].len = headTableLen[i];
		if (hCurNLists[n].len > hCurMaxLen)
			hCurMaxLen = hCurNLists[n].len;
		hCurNLists[n].support = 0;
		hCurNLists[n].idx = curIdx;
		for (int j = 0; j < headTableLen[i]; ++j) {
			hPres[curIdx] = curNode->foreIndex;
			hPosts[curIdx] = curNode->backIndex;
			hCounts[curIdx] = curNode->count;
			hCurNLists[n].support += curNode->count;
			curNode = curNode->labelSibling;
			curIdx++;
		}

		printf("Item: %d Support: %d Len: %d Idx: %d\n", hCurNLists[n].label[0], hCurNLists[n].support, hCurNLists[n].len, hCurNLists[n].idx);
		n++;
	}

}

void initDeviceMem() {
	cudaMalloc(&dCurNLists, sizeof(NList) * numOfFreqItems);
	cudaMalloc(&dCurPres, sizeof(int) * totalNListLen);
	cudaMalloc(&dCurPosts, sizeof(int) * totalNListLen);
	cudaMalloc(&dCurCounts, sizeof(int) * totalNListLen);
	cudaMalloc(&dCurNListNum, sizeof(int));
	cudaMemcpy(dCurPres, hPres, sizeof(int) * totalNListLen, cudaMemcpyHostToDevice);
	cudaMemcpy(dCurPosts, hPosts, sizeof(int) * totalNListLen, cudaMemcpyHostToDevice);
	cudaMemcpy(dCurCounts, hCounts, sizeof(int) * totalNListLen, cudaMemcpyHostToDevice);

	delete[] hCurNLists;
	delete[] hPres;
	delete[] hPosts;
	delete[] hCounts;
}

void buildPPCTree() {
	FILE *in;
	char str[MAX_CHAR_PER_LINE];
	Item transaction[MAX_ITEM_PER_TRANS];

	if ((in = fopen(fileName, "r")) == NULL) {
		printf("read wrong\n");
		exit(1);
	}

	ppcRoot.label = -1;
	memset(transaction, 0, sizeof(transaction));
	int num = 0, tLen = 0;
	while (fgets(str, MAX_CHAR_PER_LINE, in)) {

		num = 0;
		tLen = 0;
		for (int i = 0; i < MAX_CHAR_PER_LINE && str[i] != '\0'; ++i) {
			if (str[i] != ' ' && str[i] != '\n')
				num = num * 10 + str[i] - '0';
			else {
				map<int, int>::iterator it = itemIdxMap.find(num);
				if (it != itemIdxMap.end()) {
					transaction[tLen].index = num;
					transaction[tLen++].num = 0 - it->second;
				}
				num = 0;
			}
		}

		// sort the transaction in descending order
		sort(transaction, transaction + tLen, comp);
		int curPos = 0;
		PPCTreeNode *curRoot = &(ppcRoot);
		PPCTreeNode *rightSibling = NULL;

		while (curPos != tLen) {
			PPCTreeNode *child = curRoot->firstChild;
			while (child != NULL) {
				if (child->label == 0 - transaction[curPos].num) {
					curPos++;
					child->count++;
					curRoot = child;
					break;
				}
				if (child->rightSibling == NULL) {
					rightSibling = child;
					child = NULL;
					break;
				}
				child = child->rightSibling;
			}
			if (child == NULL)
				break;
		}

		for (int j = curPos; j < tLen; ++j) {
			PPCTreeNode *ppcNode = new PPCTreeNode;
			totalNListLen++;
			ppcNode->label = 0 - transaction[j].num;

			if (rightSibling != NULL) {
				rightSibling->rightSibling = ppcNode;
				rightSibling = NULL;
			}
			else {
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

	headTable = new PPCTreeNode*[numOfFreqItems];
	memset(headTable, 0, sizeof(PPCTreeNode*) * numOfFreqItems);
	headTableLen = new int[numOfFreqItems];
	memset(headTableLen, 0, sizeof(int) * numOfFreqItems);
	PPCTreeNode **tempHead = new PPCTreeNode*[numOfFreqItems];

	itemsetCount = new int[(numOfFreqItems - 1) * numOfFreqItems / 2];
	memset(itemsetCount, 0, sizeof(int) * (numOfFreqItems - 1) * numOfFreqItems / 2);

	PPCTreeNode *root = ppcRoot.firstChild;
	int pre = 1, last = 0;
	while (root != NULL) {
		root->foreIndex = pre;
		pre++;

		// insert into the headTable
		if (headTable[root->label] == NULL) {
			headTable[root->label] = root;
			tempHead[root->label] = root;
		}
		else {
			tempHead[root->label]->labelSibling = root;
			tempHead[root->label] = root;
		}
		headTableLen[root->label]++;

		// count the support of the 2nd frequent itemset (root->num, temp->num) from the leaf to the root
		PPCTreeNode *temp = root->father;
		while (temp->label != -1) {
			itemsetCount[root->label * (root->label - 1) / 2 + temp->label] += root->count;
			temp = temp->father;
		}

		if (root->firstChild != NULL)
			root = root->firstChild;
		else {
			root->backIndex = last;
			last++;
			if (root->rightSibling != NULL)
				root = root->rightSibling;
			else {
				root = root->father;
				while (root != NULL) {
					root->backIndex = last;
					last++;
					if (root->rightSibling != NULL) {
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

__global__ void generateNextLevel(struct NList *dCurNLists, int *dCurPres, int *dCurPosts, int *dCurCounts, struct NList *dNextNLists, int *dNextPres, int *dNextPosts, int *dNextCounts, int *dCurNListNum, int *dCurMaxLen) {
	*dCurNListNum = 1;
	*dCurMaxLen = 1;
}

void mining() {
	hCurNListNum = numOfFreqItems;
	for (int depth = 1; depth < MAX_DEPTH; ++depth) {
		int maxNListNum = hCurNListNum * (hCurNListNum - 1) / 2;
		cudaMemcpy(dCurNLists, hCurNLists, sizeof(NList) * hCurNListNum, cudaMemcpyHostToDevice);
		cudaMalloc(&dNextNLists, sizeof(NList) * maxNListNum);
		cudaMalloc(&dNextPres, sizeof(int) * maxNListNum * hCurMaxLen);
		cudaMalloc(&dNextPosts, sizeof(int) * maxNListNum * hCurMaxLen);
		cudaMalloc(&dNextCounts, sizeof(int) * maxNListNum * hCurMaxLen);
		cudaMemset(&dCurNListNum, 0, sizeof(int));
		cudaMemset(&dCurMaxLen, 0, sizeof(int));
		generateNextLevel<<<max(hCurNListNum/MAX_THREAD, 1), min(MAX_THREAD, hCurNListNum)>>>
		(dCurNLists, dCurPres, dCurPosts, dCurCounts, dNextNLists, dNextPres, dNextPosts, dNextCounts, dCurNListNum, dCurMaxLen);
		cudaMemcpy(dCurNListNum, &hCurNListNum, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(dCurMaxLen, &hCurMaxLen, sizeof(int), cudaMemcpyDeviceToHost);
		hCurNLists = new NList[hCurNListNum];
		cudaMemcpy(dNextNLists, hCurNLists, sizeof(NList) * hCurNListNum, cudaMemcpyDeviceToHost);
		cudaFree(dCurNLists);
		dCurNLists = dNextNLists;
	}
}
int main(int argc, char **argv) {
	supportRate = 0.4;
	fileName = "/home/manycore/users/jtyuan/GPUApriori-master/mushroom.dat";
	readFile();
	buildPPCTree();
	initNList();
	initDeviceMem();
	mining();
	// test the correctness of prepost value
	for (int i = 0; i < totalNListLen; ++i) {
		printf("(%d, %d, %d)", hPres[i], hPosts[i], hCounts[i]);
	}
	printf("Minsup: %d TransNum: %d FreqItemNum: %d TotalItemNum: %d NodeNum: %d", minSupport, numOfTrans, numOfFreqItems, numOfTotalItems, totalNListLen);


}