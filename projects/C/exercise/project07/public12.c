#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "prio-q.h"
#include "memory-checking.h"
#include "compare-name-lists.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 12 (public12.c)
 *
 * Tests that get_all_elements() is allocating new memory for elements being
 * stored in the returned array, and is not just performing pointer
 * aliasing.  This test is basically the same as the eighth Project #6
 * secret test, with memory checking added.
 *
 * Compile this test with the -D ENABLE_VALGRIND option to be able to use
 * valgrind with it (see the project assignment for explanation).
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Prio_que prio_q;
  char *expected_elements[]= {"Myrtle Turtle", "Hammy Hamster", "Ryan Lion",
                              NULL};
  char name[20]= "Hammy Hamster";
  char **result1, **result2;

  #if !defined(ENABLE_VALGRIND)
  setup_memory_checking();
  #endif

  init(&prio_q);

  enqueue(&prio_q, "Myrtle Turtle", 35);
  enqueue(&prio_q, name, 25);
  enqueue(&prio_q, "Ryan Lion", 15);

  result1= get_all_elements(prio_q);  /* get array of elements being stored */
  assert(compare_name_lists(result1, expected_elements) == 1);

  /* now change one element of result, which should not change the names
     being stored in the queue unless the array of strings being returned by
     get_all_elements() incorrectly just has aliases to the elements in the
     queue */
  strcpy(result1[0], "Horace Horse");

  result2= get_all_elements(prio_q);
  assert(compare_name_lists(result2, expected_elements) == 1);

  /* now clean up after ourselves */
  free_name_list(result1);
  free_name_list(result2);
  clear(&prio_q);

  #if !defined(ENABLE_VALGRIND)
  check_heap();
  #endif

  printf("Yet another test worked right!\n");

  return 0;
}
