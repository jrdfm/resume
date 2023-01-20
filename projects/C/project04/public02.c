#include <stdio.h>
#include "student.h"

/* CMSC 216, Spring 2021, Project #4
 * Public test 2 (public02.c)
 *
 * Tests calling init_student() and print_student() to check that the
 * student's data was stored correctly.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Student student= {NULL, NULL};

  init_student(&student, "Ryan", "Lion");

  print_student(student);
  printf("\n");

  return 0;
}
