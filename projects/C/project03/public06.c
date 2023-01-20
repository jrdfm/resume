#include <stdio.h>
#include "machine.h"

/* CMSC 216, Spring 2021, Project #3
 * Public test 6 (public06.c)
 *
 * Tests calling disassemble() on a program with some other types of valid
 * instructions.
 *
 * (c) Larry Herman, 2021.  You are allowed to use this code yourself, but
 * not to provide it to anyone else.
 */

#define PROGRAM_SIZE 7

int main(void) {
  Wrd program[PROGRAM_SIZE]= {0x304c0000, 0x80282710, 0xb0280000, 0xc8203035,
                              0xb8207d64, 0xd1400000, 0xd4000000};
  unsigned short location= -1;

  disassemble(program, PROGRAM_SIZE, &location);

  return 0;
}
