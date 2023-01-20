#include <stdio.h>
#include <assert.h>
#include "machine.h"

/* CMSC 216, Spring 2021, Project #3
 * Public test 9 (public09.c)
 *
 * Tests that disassemble() returns 0 without printing anything when an
 * instruction in the program tries to modify the special program counter
 * register R6.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

#define PROGRAM_SIZE 7

int main(void) {
  Wrd program[PROGRAM_SIZE]= {0x30cc0000, 0x80282710, 0xb0280000, 0xc8203034,
                              0xb8207d64, 0xd1400000, 0xd4000000};
  unsigned short location= -1;

  assert(disassemble(program, PROGRAM_SIZE, &location) == 0);

  printf("The assertion worked successfully!\n");

  return 0;
}
