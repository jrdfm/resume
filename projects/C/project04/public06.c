#include <stdio.h>
#include "student.h"

/* CMSC 216, Spring 2021, Project #4
 * Public test 6 (public06.c)
 *
 * Tests add_nickname().
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Student student= {NULL, NULL};

  init_student(&student, "Horace", "Horse");
  add_nickname(&student, "Horty");

  print_student(student);
  printf("\n");

  return 0;
}
