#include <stdio.h>
#include "machine.h"

/* CMSC 216, Spring 2021, Project #3
 * Public test 1 (public01.c)
 *
 * Tests calling print_instruction() on a few types of instructions.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

#define NUM_INSTRUCTIONS 5

int main(void) {
  Wrd instructions[NUM_INSTRUCTIONS]= {0x00298000, 0x084e0000, 0x10728000,
                                       0x68b18000, 0x702e8000};
  int i;

  for (i= 0; i < NUM_INSTRUCTIONS; i++) {
    print_instruction(instructions[i]);
    printf("\n");
  }

  return 0;
}
