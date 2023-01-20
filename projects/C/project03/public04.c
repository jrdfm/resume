#include <stdio.h>
#include <assert.h>
#include "machine.h"

/* CMSC 216, Spring 2021, Project #3
 * Public test 4 (public04.c)
 *
 * Tests the return value of print_instruction().
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

int main(void) {
  assert(print_instruction(0x00298000) == 1);
  printf("\n");
  /* this one should not print anything */
  assert(print_instruction(0xffffffff) == 0);

  printf("The assertions worked successfully!\n");

  return 0;
}
