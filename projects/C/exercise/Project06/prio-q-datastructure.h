#define NULL ((void *)0)
typedef struct node {
  char *data;
  int p;
  struct node *next;
} Node;

typedef struct queue {

  unsigned int count;
  Node *head;
  
} Prio_que;
