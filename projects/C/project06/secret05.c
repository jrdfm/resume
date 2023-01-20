#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Secret test 5 (secret05.c)
 *
 * Tests adding a large number of elements to a priority queue and then
 * removing them.  If you fail this test you may be using an inefficient
 * queue representation (for example where some operations are quadratic
 * time O(n^2)).
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

#define MAX_ELT 30000

int main(void) {
  Prio_que prio_q;
  char *element, *result;
  int i;

  init(&prio_q);

  /* add the elements 1..MAX_ELT to the queue but they are not in sequential
     order- they are in the order 1, MAX_ELT, 2, (MAX_ELT - 1), 3, (MAX_ELT
     - 2), etc. (note that this test is assuming that MAX_ELT is even) */
  for (i= 1; i <= MAX_ELT / 2; i++) {
    /* note that this test is allocating new memory for every element being
       added- which should not be necessary because enqueue() should do it
       itself- but we want to allow you to pass this test even if your
       enqueue() is not doing that (you will fail another secret test anyway
       if your enqueue() is not allocating new memory for elements being
       stored) */
    element= malloc(6);
    sprintf(element, "%d", i);
    enqueue(&prio_q, element, i);
    element= malloc(6);
    sprintf(element, "%d", MAX_ELT + 1 - i);
    enqueue(&prio_q, element, MAX_ELT + 1 - i);
  }

  assert(size(prio_q) == MAX_ELT);

  i= MAX_ELT;
  while (!is_empty(prio_q)) {
    result= dequeue(&prio_q);
    sprintf(element, "%d", i--);
    assert(strcmp(element, result) == 0);
  }

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}
