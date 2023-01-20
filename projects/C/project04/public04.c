#include <stdio.h>
#include "student.h"

/* CMSC 216, Spring 2021, Project #4
 * Public test 4 (public04.c)
 *
 * Tests calling create_student() and change_first_name().
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Student *student= create_student("Kora", "Koala");

  /* "student" is already a pointer */
  change_first_name(student, "Kourtney");

  /* note we must dereference the pointer to pass the structure by value */
  print_student(*student);
  printf("\n");

  return 0;
}
