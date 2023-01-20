#include <stdio.h>
#include <assert.h>
#include "machine.h"

/* CMSC 216, Spring 2021, Project #3
 * Public test 12 (public12.c)
 *
 * Tests calling where_set() on a register that is not set in the program.
 * Call your disassemble() function on the program, to see what its
 * instructions are.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

#define PROGRAM_SIZE 8

int main(void) {
  Wrd program[PROGRAM_SIZE]= {0x304c0000, 0x80282710, 0xb0280000, 0xc8203034,
                              0xb8207d64, 0xb8602800, 0xd1400000, 0xd4000000};

  assert(where_set(program, PROGRAM_SIZE, R4) == -1);

  printf("The assertion worked successfully!\n");

  return 0;
}
