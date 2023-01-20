#include "prio-q.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CMSC 216, Spring 2021, Project #6
 * prio-q.c
 *
 * I pledge on my honor that I have not given or received any unauthorized
 * assistance on this assignment.
 *
 * Yared Fikremariam
 * TerpConnect ID: yfikrema
 * Section: 0302 UID: 116945769
 *
 * This program contains functions defined in prio-q.h to manage the priority
 * queue defined in prio-q-datastructure.h.
 * init() initializes the queue, enqueue() and dequeue() add and remove elements
 * from the queue. is_empty() checks if queue has no elements and size() returns
 * the number of elements while get_all_elements() returns dynamically allocated
 * array of all elements of the priority queue.
 *
 *
 */

/* This function initializes a priority queue Prio_que that was already created
 * elsewhere it sets the count of the queue to 0 and sets the head pointer to
 * NULL
 */
static int check_count(Prio_que *prio_q, char element[]);

void init(Prio_que *const prio_q) {

  if (prio_q != NULL) {

    prio_q->count = 0;
    prio_q->head = NULL;
  }
}

/* This function adds an element with a given priority and data field to the
 * queue. With a unique priority for each element. It returns 1 if it
 * successfully stored the new element in dynamically allocated array of
 * matching size or else it returns 0 if element with priority already exists or
 * priority is negative
 *
 */

unsigned int enqueue(Prio_que *const prio_q, const char new_element[],
                     unsigned int priority) {
  Node *curr, *new, *prev;
  prev = NULL;

  /* If the queue pointer is null return 0 */
  if (prio_q == NULL) {
    return 0;
  }

  if ((priority >= 0)) {

    curr = prio_q->head;

    /* find first node in list that has a priority less than the
     * one we want to insert
     */

    while (curr != NULL && curr->p >= priority) {

      if (curr->p == priority) {
        return 0;
      }

      prev = curr;
      curr = curr->next;
    }

    /* create new node and if memory allocation succeeded
     * allocate the appropriate amount of memory for data field
     * and copy contents
     */

    new = malloc(sizeof(*new));
    if (new != NULL) {

      new->data = malloc(sizeof(char) * (strlen(new_element) + 1));
      new->p = priority;

      strcpy(new->data, new_element);

      new->next = curr;

      /* inserting new first element */
      if (prev == NULL) {
        prio_q->head = new;
        prio_q->count++;
      } else
      /* inserting in the middle of the queue */
      {
        prev->next = new;
        prio_q->count++;
      }

      return 1;
    }
  }
  return 0;
}

/* This function checks if the priority queue is empty
 * it returns 1 if it is and 0 otherwise
 */

unsigned int is_empty(const Prio_que prio_q) { return (prio_q.count == 0); }

/* This function returns the size of the priority queue
 */

unsigned int size(const Prio_que prio_q) { return (prio_q.count); }

/* This function returns a pointer to a dynamically allocated
 * array that is a copy of the element with the top priority.
 * If the queue is empty it returns NULL
 */

char *peek(Prio_que prio_q) {

  if (!(is_empty(prio_q))) {

    /* allocate the appropriate amount of memory for the string */

    char *data = malloc(strlen(prio_q.head->data) + 1);
    strcpy(data, prio_q.head->data);

    return data;
  }

  else
    return NULL;
}

/* This function removes and returns a pointer to a dynamically
 * allocated array that is a copy of the element with the top
 * priority.
 * If the queue is empty it just returns NULL
 */

char *dequeue(Prio_que *const prio_q) {

  if (prio_q == NULL) {
    return 0;
  }

  if (prio_q != NULL && (!(is_empty(*prio_q)))) {
    Node *temp;
    char *data = malloc(strlen(prio_q->head->data) + 1);
    strcpy(data, prio_q->head->data);

    /* pop the first element and return its data field */

    temp = prio_q->head;
    prio_q->head = prio_q->head->next;
    prio_q->count--;
    free(temp->data);
    free(temp);

    return data;
  }

  else
    return NULL;
}

/* This function returns a pointer to a dynamically allocated
 * array of character pointers. Containing each element of the
 * priority queue in order of decreasing priority follwed by
 * NULL element at the end.
 */

char **get_all_elements(Prio_que prio_q) {

  if (prio_q.head != NULL) {

    /* allocate the appropriate amount of memory for the array */

    char **arr = malloc((prio_q.count + 1) * sizeof(char *));
    Node *curr;
    int i, n;

    curr = prio_q.head;
    n = prio_q.count;
    /* allocate memory for the last element and set it NULL */
    /*arr[n] = malloc(sizeof(char *));*/
    arr[n] = NULL;

    /* allocate the appropriate amount of memory for each elements
     * data field and copy the string
     */
    for (i = 0; i < n; i++) {

      arr[i] = malloc(sizeof(char) * (strlen((curr->data)) + 1));

      strcpy(arr[i], curr->data);

      curr = curr->next;
    }

    return arr;
  }
  return NULL;
}

void free_name_list(char *name_list[]) {

  if (name_list != NULL) {
    int i;
    i = 0;
    while (name_list[i] != NULL) {
      free(name_list[i]);
      i++;
    }
    free(name_list[i]);
    free(name_list);
  }
}

void clear(Prio_que *const prio_q) {
  Node *curr, *pre;

  if (prio_q != NULL && prio_q->head) {
    curr = prio_q->head;
    while (curr->next != NULL) {
      pre = curr;
      curr = curr->next;
      free(pre->data);
      free(pre);
    }
    free(curr->data);
    free(curr);
  }
}

int get_priority(Prio_que prio_q, char element[]) {

  Node *curr;
  if (prio_q.head == NULL) {
    return 0;
  }

  curr = prio_q.head;
  while (curr->next != NULL && strcmp(element, curr->data) != 0) {
    curr = curr->next;
  }

  if (strcmp(element, curr->data) == 0) {
    return curr->p;
  }

  return -1;
}

unsigned int remove_elements_between(Prio_que *const prio_q, unsigned int low,
                                     unsigned int high) {
  Node *curr, *prev, *temp;
  int count;

  if (prio_q == NULL) {
    return 0;
  }

  curr = prio_q->head;
  count = 0;
  while (curr->next != NULL && curr->p > high) {
    prev = curr;
    curr = curr->next;
  }

while (curr != NULL && curr->p >= low)
{
  /* code */
}




  while (curr != NULL && curr->p >= low) {
    /*head case*/
    if (curr == prio_q->head) {
      if (dequeue(prio_q) != NULL ) {
        count++;
      }
    }
   else {
    temp = curr;
    prev->next = curr->next;
    prio_q->count--;
    count++;
    free(temp->data);
    free(temp);
   }
   if (curr != NULL && curr->next != NULL)
   {
    curr = curr->next;
   }
   
    
  }

  return count;
}

unsigned int change_priority(Prio_que *prio_q, char element[],
                             unsigned int new_priority) {
  Node *curr;
  if (prio_q == NULL) {
    return 0;
  }

  curr = prio_q->head;
  while (curr->next != NULL && strcmp(element, curr->data) != 0 &&
         curr->p != new_priority) {
    curr = curr->next;
  }
  if ((curr->p == new_priority) || (curr->next == NULL) ||
      check_count(prio_q, element) != 1) {
    return 0;
  }

  if (strcmp(element, curr->data) == 0) {

    curr->p = new_priority;
    return 1;
  }

  return 0;
}

int check_count(Prio_que *prio_q, char element[]) {
  Node *curr;
  int count;
  if (prio_q == NULL) {
    return 0;
  }

  curr = prio_q->head;
  count = 0;
  while (curr->next != NULL) {

    if (strcmp(element, curr->data) == 0) {
      count++;
    }
    curr = curr->next;
  }
  if (curr->next != NULL) {
    count++;
  }

  return count;
}