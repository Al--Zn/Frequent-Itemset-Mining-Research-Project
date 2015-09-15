#include <iostream>
#include <cstdio>
#include <map>
#include <vector>
#include <algorithm>

#define MAX_CHAR_PER_LINE 500
#define MAX_ITEM_PER_TRANS 1000

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
		if (feof(in))
			break;
		numOfTrans++;
		num = 0;
		for (int i = 0; i < MAX_CHAR_PER_LINE && str[i] != '\0'; ++i) {
			if (str[i] != ' ')
				num = num * 10 + str[i] - '0';
			else {
				if (0 == itemCntMap[num]++)
					numOfTotalItems++;
				num = 0;
			}
		}
	}
	fclose(in);
	minSupport = int(supportRate * numOfTrans);
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
		if (feof(in))
			break;
		num = 0;
		tLen = 0;
		for (int i = 0; i < MAX_CHAR_PER_LINE && str[i] != '\0'; ++i) {
			if (str[i] != ' ')
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
//		for (int i = 0; i < tLen; ++i) {
//			printf("%d %d\n", transaction[i].index, transaction[i].num);
//		}
//		system("pause");
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
	memset(headTableLen, 0, sizeof(int*) * numOfFreqItems);
	PPCTreeNode **tempHead = new PPCTreeNode*[numOfFreqItems];

	itemsetCount = new int[(numOfFreqItems - 1) * numOfFreqItems / 2];
	memset(itemsetCount, 0, sizeof(int) * (numOfFreqItems - 1) * numOfFreqItems / 2);

	PPCTreeNode *root = ppcRoot.firstChild;
	int pre = 0, last = 0;
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

int main(int argc, char **argv) {
	supportRate = 0.2;
	fileName = "mushroom.dat";
	readFile();
	buildPPCTree();
	printf("%d %d %d %d", minSupport, numOfTrans, numOfFreqItems, numOfTotalItems);


}