#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Secret test 2 (secret02.c)
 *
 * Tests that duplicate elements can be added to a priority queue (not ones
 * with duplicate priorities).
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;
  char *element;
  int i;

  init(&prio_q);

  enqueue(&prio_q, "Sally Salamander", 9);
  enqueue(&prio_q, "Sally Salamander", 2);
  enqueue(&prio_q, "Sally Salamander", 8);
  enqueue(&prio_q, "Sally Salamander", 3);
  enqueue(&prio_q, "Sally Salamander", 1);
  enqueue(&prio_q, "Sally Salamander", 5);
  enqueue(&prio_q, "Sally Salamander", 7);
  enqueue(&prio_q, "Sally Salamander", 4);

  for (i= 1; i <= 8; i++) {
    element= dequeue(&prio_q);
    assert(strcmp(element, "Sally Salamander") == 0);
  }

  assert(is_empty(prio_q));
  assert(size(prio_q) == 0);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}
