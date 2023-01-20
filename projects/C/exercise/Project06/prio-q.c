#include "prio-q.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init(Prio_que *const prio_q) {

  prio_q->count = 0;
  prio_q->head = NULL;
}

unsigned int enqueue(Prio_que *const prio_q, const char new_element[],
                     unsigned int priority) {

  Node *curr, *new, *prev;
  prev = NULL;

  if (prio_q == NULL) {
    return 0;
  }

  if ((priority >= 0)) {
    /* If the queue is empty the new node is head and tail*/

    curr = prio_q->head;

    /* find first node in list that's greater than or equal to the one we
       want to insert */
    while (curr != NULL && curr->p >= priority) {

      if (curr->p == priority) {
        return 0;
      }

      prev = curr;
      curr = curr->next;
    }

    /* create new node and if memory allocation succeeded fill in its
       fields */

    new = malloc(sizeof(*new));
    if (new != NULL) {

      new->data = malloc(sizeof(char) * (strlen(new_element) + 1));
      new->p = priority;

      strcpy(new->data, new_element);

      new->next = curr;

      if (prev == NULL) {
        prio_q->head = new; /* special case- inserting new first element */
        prio_q->count++;
      } else

      {
        prev->next = new; /* general case- inserting elsewhere */
        prio_q->count++;
      }

      return 1;
    }
  }

  return 0;
}

int is_empty(const Prio_que priq) { return (priq.count == 0); }

int size(const Prio_que prio_q) { return prio_q.count; }

char *peek(Prio_que prio_q) {

  if (!(is_empty(prio_q))) {

    char *data = malloc(strlen(prio_q.head->data) + 1);

    strcpy(data, prio_q.head->data);

    return data;
  }

  else
    return NULL;
}

char *dequeue(Prio_que *const prio_q) {

  if (prio_q == NULL) {
    return 0;
  }

  if (prio_q != NULL && (!(is_empty(*prio_q)))) {
    Node *temp;

    temp = prio_q->head;
    prio_q->head = prio_q->head->next;
    prio_q->count--;

    return temp->data;
  }

  else
    return NULL;
}

char **get_all_elements(Prio_que prio_q) {

  char **re = malloc(sizeof(char *));
  re[0] = NULL;

  if (prio_q.head != NULL) {
    char **arr = malloc((prio_q.count + 1) * sizeof(char *));
    Node *curr;
    int i, n;

    curr = prio_q.head;
    n = prio_q.count;

    arr[n] = malloc(sizeof(char *));
    arr[n] = NULL;
    for (i = 0; i < n; i++) {

      arr[i] = malloc(sizeof(char) * (strlen((curr->data)) + 1));

      strcpy(arr[i], curr->data);

      curr = curr->next;
    }

    return arr;
  }
  return re;
}
