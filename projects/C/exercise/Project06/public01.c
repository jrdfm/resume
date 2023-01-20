#include <stdio.h>
#include <assert.h>
#include "prio-q.h"


/* CMSC 216, Spring 2021, Project #6
 * Public test 1 (public01.c)
 *
 * Tests that a newly declared and initialized priority queue has no
 * elements and its size is 0.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que priq;

  init(&priq);

  assert(is_empty(priq));
  assert(size(priq) == 0);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}
