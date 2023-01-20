#include <stdio.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Public test 7 (public07.c)
 *
 * Tests calling dequeue() on an empty priority queue.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;

  init(&prio_q);

  /* this should have no effect, and definitely not crash */
  assert(dequeue(&prio_q) == NULL);

  assert(is_empty(prio_q));
  assert(size(prio_q) == 0);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}
