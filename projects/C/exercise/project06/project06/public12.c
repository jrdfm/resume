#include <stdio.h>
#include <assert.h>
#include "prio-q.h"
#include "compare-name-lists.h"

/* CMSC 216, Spring 2021, Project #6
 * Public test 12 (public12.c)
 *
 * Tests calling get_all_elements() on a priority queue that has no
 * elements.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;
  char *expected_elements[]= {NULL};

  init(&prio_q);

  assert(compare_name_lists(get_all_elements(prio_q), expected_elements) == 1);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}
