#include <stdio.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Public test 10 (public10.c)
 *
 * Tests passing NULL into some of the functions' parameters.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;

  init(&prio_q);

  assert(enqueue(NULL, "Sally Salamander", 50) == 0);
  assert(dequeue(NULL) == 0);

  assert(is_empty(prio_q));
  assert(size(prio_q) == 0);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}
