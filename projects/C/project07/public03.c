#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "prio-q.h"
#include "memory-checking.h"

/* CMSC 216, Spring 2021, Project #7
 * Public test 3 (public03.c)
 *
 * Tests free_name_list() and ensures that there are no memory leaks or
 * errors in the heap.
 *
 * Compile this test with the -D ENABLE_VALGRIND option to be able to use
 * valgrind with it (see the project assignment for explanation).
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  char *strings[]= {"Earl Squrrel", "Sally Salamander", "Shelly Sheep",
                    "Robin Robin", "Giselle Gazelle", "Neil Seal",
                    "Placido Flamingo"};
  char **ptr;
  int i;

  #if !defined(ENABLE_VALGRIND)
  setup_memory_checking();
  #endif

  /* create a dynamically-allocated list of strings of the form that
     free_name_list() is expecting; note that we CANNOT call free_name_list
     on the array strings above because it is not dynamically allocated */
  ptr= malloc((1 + ARRSIZE(strings)) * sizeof(char *));  /* "1 +" for NULL */
  if (ptr == NULL) {
    puts("Goodbye cruel world!  Memory could not be allocated. :-(\n");
    exit(-1);
  } else {
    /* allocate space for and copy values to each element of the array of
       strings */
    for (i= 0; i < ARRSIZE(strings); i++) {
      ptr[i]= malloc(strlen(strings[i]) + 1);
      strcpy(ptr[i], strings[i]);
    }
    ptr[i]= NULL;  /* terminate the list of strings with NULL */

    /* now free the whole darn thing (it should use no memory afterwards) */
    free_name_list(ptr);
  }

  #if !defined(ENABLE_VALGRIND)
  check_heap();
  #endif

  printf("Yet another test worked right!\n");

  return 0;
}
