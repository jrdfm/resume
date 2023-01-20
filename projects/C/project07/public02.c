#include <stdio.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 2 (public02.c)
 *
 * Tests calling clear() on an empty priority queue.
 *
 * This test will always unavoidably have memory leaks if you run it under
 * valgrind, so fix any other problems that valgrind identifies, but ignore
 * any memory leaks (or just add the --leak-check=no argument when running
 * this test with valgrind).
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;

  init(&prio_q);

  assert(size(prio_q) == 0);

  /* this should have no effect, and definitely not crash */
  clear(&prio_q);
  init(&prio_q);

  assert(is_empty(prio_q));
  assert(size(prio_q) == 0);

  printf("Yet another test worked right!\n");

  return 0;
}
