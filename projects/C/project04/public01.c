#include <stdio.h>
#include "student.h"

/* CMSC 216, Spring 2021, Project #4
 * Public test 1 (public01.c)
 *
 * Tests calling print_student() on a student who just has NULL names, to
 * ensure that it doesn't print anything and doesn't crash.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  Student student= {NULL, NULL};

  print_student(student);

  printf("If this is the only thing printed by this test- without anything ");
  printf("above this-\nthen it worked successfully!\n");

  return 0;
}
