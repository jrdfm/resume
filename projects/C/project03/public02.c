#include <stdio.h>
#include "machine.h"

/* CMSC 216, Spring 2021, Project #3
 * Public test 2 (public02.c)
 *
 * Tests calling print_instruction() on some other types of instructions.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

#define NUM_INSTRUCTIONS 7

int main(void) {
  Wrd instructions[NUM_INSTRUCTIONS]= {0x304c0000, 0x80282710, 0xb0280000,
                                       0xc8203034, 0xb8207d64, 0xd1400000,
                                       0xd4000000};
  int i;

  for (i= 0; i < NUM_INSTRUCTIONS; i++) {
    print_instruction(instructions[i]);
    printf("\n");
  }

  return 0;
}
