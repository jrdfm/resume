#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "prio-q.h"

/* CMSC 216, Spring 2021, Project #6
 * Secret test 6 (secret06.c)
 *
 * Tests that very long strings can be added as elements to a priority
 * queue.  (If this test fails you may be allocating the wrong amount of
 * memory for queue elements.)
 *
 * This test uses the library function memset(), which was not explained in
 * class but is covered in the Reek text.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

#define SZ 2000

int main(void) {
  Prio_que prio_q;
  char arr1[SZ], arr2[SZ], arr3[SZ];
  char *element;

  init(&prio_q);

  memset(arr1, 'B', SZ - 1);
  arr1[SZ - 1]= '\0';  /* properly null terminate */
  memset(arr2, 'D', SZ - 1);
  arr2[SZ - 1]= '\0';  /* properly null terminate */
  memset(arr3, 'F', SZ - 1);
  arr3[SZ - 1]= '\0';  /* properly null terminate */

  enqueue(&prio_q, "A", 80);
  enqueue(&prio_q, arr1, 60);
  enqueue(&prio_q, "C", 10);
  enqueue(&prio_q, arr2, 20);
  enqueue(&prio_q, "E", 30);
  enqueue(&prio_q, arr3, 40);
  enqueue(&prio_q, "G", 50);

  element= dequeue(&prio_q);
  assert(strcmp(element, "A") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, arr1) == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "G") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, arr3) == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "E") == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, arr2) == 0);
  element= dequeue(&prio_q);
  assert(strcmp(element, "C") == 0);

  assert(is_empty(prio_q));
  assert(size(prio_q) == 0);

  printf("Every assertion succeeded resoundingly!\n");

  return 0;
}
