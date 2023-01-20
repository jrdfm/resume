/* CMSC 216, Spring 2021, Project #6
 * prio-q-datastructure.h
 *
 * I pledge on my honor that I have not given or received any unauthorized
 * assistance on this assignment.
 *
 * Yared Fikremariam
 * TerpConnect ID: yfikrema
 * Section: 0302 UID: 116945769
 *
 * Definition for the priority queue data structure, which is
 * a singly linked list. 
 * Each node of this list is a structure that has a data 
 * field of a character array, an int priority and a
 * pointer to the next node.
 *
 * The structure Prio_que has an unsigned int that is count
 * and a pointer to its head.
 * 
 */

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
