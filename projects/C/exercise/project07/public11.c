#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "prio-q.h"
#include "memory-checking.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 11 (public11.c)
 *
 * Tests that very long strings can be added as elements to a priority
 * queue.  (If this test fails you may be allocating the wrong amount of
 * memory for queue elements.)  This test is basically the same as the sixth
 * Project #6 secret test, with memory checking added.
 *
 * This test uses the library function memset(), which was not explained in
 * class but is covered in the Reek text.
 *
 * Compile this test with the -D ENABLE_VALGRIND option to be able to use
 * valgrind with it (see the project assignment for explanation).
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

#define SZ 2000

int main(void) {
  Prio_que prio_q;
  char arr1[SZ], arr2[SZ], arr3[SZ];
  char *element;

  #if !defined(ENABLE_VALGRIND)
  setup_memory_checking();
  #endif

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
  free(element);  /* now clean up after ourselves */
  element= dequeue(&prio_q);
  assert(strcmp(element, arr1) == 0);
  free(element);
  element= dequeue(&prio_q);
  assert(strcmp(element, "G") == 0);
  free(element);
  element= dequeue(&prio_q);
  assert(strcmp(element, arr3) == 0);
  free(element);
  element= dequeue(&prio_q);
  assert(strcmp(element, "E") == 0);
  free(element);
  element= dequeue(&prio_q);
  assert(strcmp(element, arr2) == 0);
  free(element);
  element= dequeue(&prio_q);
  assert(strcmp(element, "C") == 0);
  free(element);

  assert(is_empty(prio_q));
  assert(size(prio_q) == 0);

  clear(&prio_q);

  #if !defined(ENABLE_VALGRIND)
  check_heap();
  #endif

  printf("Yet another test worked right!\n");

  return 0;
}
