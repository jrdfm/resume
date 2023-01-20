#if !defined(SIMPLE_SSERVER_DATASTRUCTURE)
#define SIMPLE_SSERVER_DATASTRUCTURE

/* CMSC 216, Spring 2021, Project #9
 * simple-sserver-datastructure.h
 *
 * I pledge on my honor that I have not given or received any unauthorized
 * assistance on this assignment.
 *
 * Yared Fikremariam
 * TerpConnect ID: yfikrema
 * Section: 0302 UID: 116945769
 *
 * Definition for the Commands data structure, which is two singly linked lists
 * , containing compilation commands and test commands.
 *
 */

#define NULL ((void *)0)

typedef struct node {
  char *data;
  struct node *next;
} Node;

typedef struct command {

  Node *cmp_head;
  Node *tst_head;

} Commands;

#endif
